"""Tool-calling Ask Margin: the model decides what to look up, in a bounded loop.

Design (see docs/companion.md, "Agent mode"):
  - Start from the same evidence the single-pass path uses, so most questions are
    answered in one model call at today's speed. Tools are for when that's not enough.
  - Every model call streams. If the model answers with text, that text is the answer;
    there is no separate "now write the answer" call.
  - Lookups the question obviously needs (notes, library) are started speculatively in
    parallel with the first model call, and reused if the model asks for them.
  - Several tool calls in one step run in parallel. Hard cap of MAX_STEPS model calls
    and a total timeout; then the model answers with what it has and says so.
  - Citations may only reference ids shown this turn; others are stripped.
  - propose_note never writes. It returns a pending proposal the reader approves in
    the UI, which saves through the normal notes path.
  - If tool calling fails, fall back to the single-pass path.
"""

import asyncio
import json
import os
import re
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from time import perf_counter
from typing import Any
from uuid import uuid4

import httpx
from fastapi import HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

import companion as core

AGENT_MODEL = os.getenv("MARGIN_AGENT_MODEL", core.MODEL)
MAX_STEPS = 4  # model calls per answer, including the one that writes the answer
TOTAL_TIMEOUT = float(os.getenv("MARGIN_AGENT_TIMEOUT", "30"))
TOOL_TIMEOUT = 10.0
MAX_CALLS_PER_STEP = 4
MAX_SOURCES = 40
MODEL_TEXT_LIMIT = 1800  # characters of each passage shown to the model in tool results
HOLD_CHARS = 48  # text held back until we know the model isn't calling a tool

TOOL_GUIDE = (
    "\n\nLooking things up:\n"
    "- The passages below may already be enough. If they are, answer straight away "
    "without calling a tool.\n"
    "- Call a tool only when you need something they do not contain: search_notes for "
    "what the reader thought, noted or wrote (use days for phrases like 'last week'); "
    "search_library for other articles they have read; search_article for elsewhere in "
    "this article; get_passage to read around a numbered passage.\n"
    "- You may call several tools at once. Keep queries short and specific.\n"
    "- Tool results add numbered passages. Cite them like the others, and never cite a "
    "number you have not been shown.\n"
    "- Use propose_note only when the reader asks you to save, note or remember "
    "something. It shows them a draft to approve; never say a note was saved.\n"
    "- If a lookup finds nothing useful, say so plainly rather than guessing."
)

FINAL_NUDGE = (
    "(Margin: no more lookups are available for this answer. Answer now with what you "
    "have. If you could not check something you wanted to, say so in one short sentence.)"
)


def _query(description: str) -> dict[str, Any]:
    return {"type": "string", "description": description}


TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_article",
            "description": "Search the article the reader is reading for passages.",
            "parameters": {
                "type": "object",
                "properties": {"query": _query("What to look for, in a few words.")},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_library",
            "description": "Search every article in the reader's library for passages.",
            "parameters": {
                "type": "object",
                "properties": {"query": _query("What to look for, in a few words.")},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_notes",
            "description": "Search the reader's own saved notes (thoughts) on what they read.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": _query("What the notes might be about, in a few words."),
                    "days": {
                        "type": "integer",
                        "description": "Only notes from the last N days, if the reader "
                        "mentions a time.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_passage",
            "description": "Read a numbered passage together with the passages around it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "passage_id": {
                        "type": "integer",
                        "description": "The passage number, as shown in brackets.",
                    }
                },
                "required": ["passage_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "propose_note",
            "description": "Draft a note on a passage of the current article for the reader "
            "to approve. Does not save anything by itself.",
            "parameters": {
                "type": "object",
                "properties": {
                    "passage_id": {
                        "type": "integer",
                        "description": "The passage number the note belongs to. Leave out "
                        "for the passage the reader is on.",
                    },
                    "text": {"type": "string", "description": "The note, in plain prose."},
                },
                "required": ["text"],
            },
        },
    },
]
TOOL_NAMES = {tool["function"]["name"] for tool in TOOLS}


class Args(BaseModel):
    model_config = ConfigDict(extra="ignore")


class QueryArgs(Args):
    query: str = Field(min_length=2, max_length=300)


class NotesArgs(QueryArgs):
    days: int | None = Field(default=None, ge=1, le=3650)


class PassageArgs(Args):
    passage_id: int = Field(ge=1)


class NoteArgs(Args):
    passage_id: int | None = Field(default=None, ge=1)
    text: str = Field(min_length=1, max_length=800)


ARGS: dict[str, type[Args]] = {
    "search_article": QueryArgs,
    "search_library": QueryArgs,
    "search_notes": NotesArgs,
    "get_passage": PassageArgs,
    "propose_note": NoteArgs,
}


class ToolError(Exception):
    """A problem the model should hear about and can recover from."""


class ModelError(Exception):
    def __init__(self, status: int) -> None:
        super().__init__(f"model returned HTTP {status}")
        self.status = status


def anchor(source: dict[str, Any]) -> tuple:
    return (source["kind"], source["url"], source["paragraphIndex"])


class Ledger:
    """Every source shown this turn, with stable citation ids."""

    def __init__(self, sources: list[dict[str, Any]]) -> None:
        self.sources = list(sources)
        self.by_anchor = {anchor(source): source for source in self.sources}

    def get(self, source_id: int) -> dict[str, Any] | None:
        if 1 <= source_id <= len(self.sources):
            return self.sources[source_id - 1]
        return None

    @property
    def ids(self) -> set[int]:
        return {source["id"] for source in self.sources}

    def add(self, found: list[dict[str, Any]]) -> list[tuple[dict[str, Any], bool]]:
        """Returns (source, is_new) pairs; known anchors keep their original id."""
        added = []
        for source in found:
            existing = self.by_anchor.get(anchor(source))
            if existing is not None:
                added.append((existing, False))
                continue
            if len(self.sources) >= MAX_SOURCES:
                break
            numbered = dict(source, text=source["text"][:4000], id=len(self.sources) + 1)
            self.sources.append(numbered)
            self.by_anchor[anchor(numbered)] = numbered
            added.append((numbered, True))
        return added


class Turn:
    """State for one answer."""

    def __init__(self, question: core.Question, key: str, ledger: Ledger) -> None:
        self.question = question
        self.key = key
        self.ledger = ledger
        self.proposal: dict[str, Any] | None = None
        self.prefetch: dict[str, asyncio.Task] = {}
        self.prefetch_hits = 0
        # The citation number of the passage the reader is on, if it was supplied.
        self.selected_id = next(
            (
                s["id"]
                for s in ledger.sources
                if s["kind"] == "article"
                and s["url"] == question.article_url
                and s["paragraphIndex"] == question.paragraph_index
            ),
            None,
        )


# --- tools ---------------------------------------------------------------------------


async def search_tool(
    turn: Turn, name: str, query: str, days: int | None = None
) -> list[dict[str, Any]]:
    """Hybrid search, scoped to this reader and, for search_article, this article."""
    vector, _ = await core.query_vector(query, turn.key, turn.question.client_id)
    await run_in_threadpool(core.ensure_search_index)
    options: dict[str, Any]
    if name == "search_article":
        options = {
            "article_url": turn.question.article_url,
            "thoughts": False,
            "max_passages": 6,
            "per_article": 6,
        }
    elif name == "search_library":
        options = {"thoughts": False, "max_passages": 8}
    else:
        options = {
            "passages": False,
            "max_thoughts": 6,
            "thoughts_per_article": 3,
            "since": datetime.now(UTC) - timedelta(days=days) if days else None,
        }
    return await run_in_threadpool(
        lambda: core.search(turn.question.client_id, vector, query, **options)
    )


def neighbours(turn: Turn, source: dict[str, Any]) -> list[dict[str, Any]]:
    """The passage plus one either side, from the reader's index of that article."""
    position = source["paragraphIndex"]
    with Session(core.engine) as session:
        rows = session.scalars(
            select(core.Passage)
            .where(
                core.Passage.client_id == turn.question.client_id,
                core.Passage.article_url == source["url"],
                core.Passage.position.between(position - 1, position + 1),
            )
            .order_by(core.Passage.position, core.Passage.fragment)
        ).all()
    found: dict[int, dict[str, Any]] = {}
    for row in rows:
        found.setdefault(
            row.position,
            {
                "kind": "article",
                "title": row.title,
                "url": row.article_url,
                "createdAt": None,
                "paragraphIndex": row.position,
                "text": row.content,
            },
        )
    if not found and source["url"] == turn.question.article_url:
        # Not indexed yet: fall back to the passage window the reader sent.
        start = turn.question.context_start
        for offset, content in enumerate(turn.question.context):
            if abs(start + offset - position) <= 1:
                found[start + offset] = {
                    "kind": "article",
                    "title": turn.question.title,
                    "url": turn.question.article_url,
                    "createdAt": None,
                    "paragraphIndex": start + offset,
                    "text": content,
                }
    return [found[key] for key in sorted(found)]


def describe(turn: Turn, source: dict[str, Any], is_new: bool) -> str:
    if not is_new:
        return f"[{source['id']}] (already shown above)"
    label = core.describe_source(source, turn.question)
    if source["kind"] == "thought" and source.get("createdAt"):
        label += f", written {str(source['createdAt'])[:10]}"
    return f"[{source['id']}] ({label})\n{source['text'][:MODEL_TEXT_LIMIT]}"


def similar(query: str, question: str) -> bool:
    """Close enough to reuse a lookup started speculatively for the question."""
    asked = set(core.keyword_query(question).split(" | ")) - {""}
    wanted = set(core.keyword_query(query).split(" | ")) - {""}
    if not asked or not wanted:
        return False
    return len(asked & wanted) / len(wanted) >= 0.6


async def run_tool(turn: Turn, name: str, raw: str) -> tuple[dict[str, Any], str, list[str]]:
    """Execute one tool call. Returns (step fields, text for the model, extra events)."""
    step: dict[str, Any] = {"tool": name}
    if name not in TOOL_NAMES:
        raise ToolError(f"Unknown tool {name!r}.")
    try:
        values = json.loads(raw or "{}")
        if not isinstance(values, dict):
            raise ValueError
        args = ARGS[name].model_validate(values)
    except (ValueError, ValidationError) as error:
        raise ToolError(f"Invalid arguments for {name}: {str(error)[:200]}") from error

    if isinstance(args, QueryArgs):
        step["query"] = args.query
        days = args.days if isinstance(args, NotesArgs) else None
        task = turn.prefetch.get(name)
        if task is not None and similar(args.query, turn.question.question):
            found = await task
            turn.prefetch_hits += 1
            if days:
                cutoff = datetime.now(UTC) - timedelta(days=days)
                found = [
                    s
                    for s in found
                    if s.get("createdAt") and datetime.fromisoformat(s["createdAt"]) >= cutoff
                ]
        else:
            found = await search_tool(turn, name, args.query, days)
        added = turn.ledger.add(found)
        step["results"] = len(added)
        if not added:
            return step, "No matches.", []
        return step, "\n\n".join(describe(turn, s, new) for s, new in added), []

    assert isinstance(args, PassageArgs | NoteArgs)
    passage_id = args.passage_id or turn.selected_id
    if passage_id is None:
        raise ToolError("Say which passage number the note belongs to.")
    source = turn.ledger.get(passage_id)
    if source is None:
        raise ToolError(f"There is no passage [{passage_id}] in this conversation.")
    step["paragraphIndex"] = source["paragraphIndex"]
    step["title"] = source["title"]

    if name == "get_passage":
        if source["kind"] != "article":
            raise ToolError(f"[{passage_id}] is the reader's note, not an article passage.")
        added = turn.ledger.add(await run_in_threadpool(neighbours, turn, source))
        step["results"] = len(added)
        if not added:
            return step, f"Nothing more is available around [{passage_id}].", []
        return step, "\n\n".join(describe(turn, s, True) for s, _ in added), []

    # propose_note: show a draft; never write.
    assert isinstance(args, NoteArgs)
    if source["url"] != turn.question.article_url:
        raise ToolError("Notes can only be proposed on passages of the current article.")
    if turn.proposal is not None:
        raise ToolError("Only one note can be proposed per answer.")
    turn.proposal = {
        "id": uuid4().hex[:12],
        "paragraphIndex": source["paragraphIndex"],
        "sourceId": source["id"],
        "text": args.text.strip(),
        "status": "pending",
    }
    step["results"] = 1
    return (
        step,
        f"Shown to the reader as a draft note on [{source['id']}]. It is pending: nothing is "
        "saved unless they approve it, so do not say it was saved.",
        [core.event("proposal", proposal=turn.proposal)],
    )


# --- speculation -------------------------------------------------------------------

SAVE_HINT = re.compile(
    r"^\s*(please\s+)?((can|could|would|will) you\s+)?(please\s+)?"
    r"((save|add|write|make|jot|draft|leave|keep)\b.{0,40}\bnotes?\b"
    r"|note (this|that|down|here)\b|remember (this|that)\b)",
    re.I,
)

NOTES_HINT = re.compile(
    r"\b(my|i|i've|i'd)\b[^.?!]{0,40}\b(thoughts?|notes?|noted|wrote|written|think|thinking"
    r"|said|highlight\w*)\b|\b(last|this) (week|month)\b|\byesterday\b",
    re.I,
)
LIBRARY_HINT = re.compile(
    r"\b(library|other articles?|elsewhere|across|everything i|i(?:'ve| have) read"
    r"|connect\w*|compare\w*)\b",
    re.I,
)


def start_prefetch(turn: Turn) -> None:
    """Start the lookups the question obviously needs, alongside the first model call."""
    question = turn.question.question
    wanted = []
    if NOTES_HINT.search(question):
        wanted.append("search_notes")
    if LIBRARY_HINT.search(question) and turn.question.scope != "library":
        wanted.append("search_library")
    for name in wanted:
        task = asyncio.create_task(search_tool(turn, name, question[:300]))
        task.add_done_callback(lambda done: done.cancelled() or done.exception())
        turn.prefetch[name] = task


# --- model -------------------------------------------------------------------------


async def model_stream(
    key: str, messages: list[dict[str, Any]], allow_tools: bool, force: str | None = None
) -> AsyncIterator[tuple[str, Any]]:
    payload: dict[str, Any] = {
        "model": AGENT_MODEL,
        "messages": messages,
        "stream": True,
        "temperature": 0.5,
        "max_tokens": 900,
        "tools": TOOLS,
        "tool_choice": "auto" if allow_tools else "none",
    }
    if allow_tools:
        payload["parallel_tool_calls"] = True
        if force:
            payload["tool_choice"] = {"type": "function", "function": {"name": force}}
    async with (await core.client()).stream(
        "POST",
        f"{core.BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json=payload,
    ) as response:
        if response.status_code >= 400:
            await response.aread()
            raise ModelError(response.status_code)
        async for line in response.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            data = json.loads(line[6:])
            choices = data.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            if delta.get("content"):
                yield "text", delta["content"]
            for call in delta.get("tool_calls") or []:
                yield "tool", call


def collect(calls: dict[int, dict[str, str]], chunk: dict[str, Any]) -> None:
    """Tool calls stream in pieces keyed by index; stitch them together."""
    slot = calls.setdefault(int(chunk.get("index", len(calls))), {"id": "", "name": "", "args": ""})
    if chunk.get("id"):
        slot["id"] = chunk["id"]
    function = chunk.get("function") or {}
    if function.get("name"):
        slot["name"] = function["name"]
    if function.get("arguments"):
        slot["args"] += function["arguments"]


# --- loop --------------------------------------------------------------------------


async def run_agent(
    question: core.Question,
    key: str,
    request: Request,
    started: float,
    timing: dict[str, object],
) -> AsyncIterator[str]:
    timing.update(agent=True, model=AGENT_MODEL)
    try:
        sources, library = await core.gather_sources(question, key, timing)
    except (httpx.HTTPError, KeyError, ValueError, HTTPException, SQLAlchemyError):
        yield core.event("error", detail=core.COMPANION_ERROR)
        return
    ledger = Ledger(sources)
    turn = Turn(question, key, ledger)
    yield core.event("sources", sources=ledger.sources)
    start_prefetch(turn)

    messages = core.base_messages(question, TOOL_GUIDE)
    timing["historyTurns"] = (len(messages) - 1) // 2
    prompt = core.build_prompt(question, sources, library)
    if turn.selected_id is not None:
        prompt += (
            f"\n\n(The reader is on passage [{turn.selected_id}]; 'here' or 'this passage' "
            "means that one.)"
        )
    messages.append({"role": "user", "content": prompt})
    # A plain request to save a note: make the first step draft it, instead of trusting
    # a small model to notice.
    force = "propose_note" if SAVE_HINT.search(question.question) else None
    deadline = started + TOTAL_TIMEOUT
    answer: core.AnswerStream | None = None
    step_id = 0
    tool_calls = 0
    steps = 0
    try:
        step = 0
        while step < MAX_STEPS:
            step += 1
            out_of_time = perf_counter() > deadline
            final = step == MAX_STEPS or out_of_time
            if final and step > 1:
                timing["stoppedEarly"] = "timeout" if out_of_time else "step cap"
                messages.append({"role": "user", "content": FINAL_NUDGE})
            steps = step
            calls: dict[int, dict[str, str]] = {}
            held = ""
            tick = perf_counter()
            try:
                async for kind, value in model_stream(
                    key, messages, allow_tools=not final, force=force if step == 1 else None
                ):
                    if await request.is_disconnected():
                        return
                    if kind == "tool":
                        collect(calls, value)
                        continue
                    if answer is None:
                        if calls:
                            continue  # text alongside tool calls is the model thinking aloud
                        held += value
                        if len(held) < HOLD_CHARS:
                            continue
                        answer = core.AnswerStream(ledger.ids)
                        timing["firstAnswerMs"] = round((perf_counter() - started) * 1000, 1)
                        value, held = held, ""
                    visible = answer.feed(value)
                    if visible:
                        yield core.event("delta", text=visible)
            except (ModelError, httpx.HTTPError, ValueError, KeyError) as error:
                if answer is not None:
                    raise
                if force and step == 1 and isinstance(error, ModelError):
                    # The provider may not support forcing a tool; retry this step on auto.
                    timing["forcedToolRejected"] = error.status
                    force = None
                    step = 0
                    continue
                # Tools unsupported or the call failed: answer the simple way, with
                # everything gathered so far.
                timing["fallback"] = str(error)[:120]
                async for line in core.single_pass(
                    question, key, request, started, timing, ledger.sources, library
                ):
                    yield line
                return
            timing[f"step{step}Ms"] = round((perf_counter() - tick) * 1000, 1)

            named = [calls[i] for i in sorted(calls) if calls[i]["name"]]
            if named and not final and answer is None:
                tool_calls += len(named)
                for i, call in enumerate(named):
                    call["id"] = call["id"] or f"call_{step}_{i}"
                messages.append(
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": call["id"],
                                "type": "function",
                                "function": {"name": call["name"], "arguments": call["args"]},
                            }
                            for call in named
                        ],
                    }
                )
                runnable = named[:MAX_CALLS_PER_STEP]
                ids = []
                for call in runnable:
                    step_id += 1
                    ids.append(step_id)
                    preview: dict[str, Any] = {"tool": call["name"]}
                    try:
                        parsed = json.loads(call["args"] or "{}")
                        if isinstance(parsed, dict) and isinstance(parsed.get("query"), str):
                            preview["query"] = parsed["query"][:300]
                    except ValueError:
                        pass
                    yield core.event("step", id=step_id, status="running", **preview)

                async def guarded(call: dict[str, str]):
                    try:
                        return await asyncio.wait_for(
                            run_tool(turn, call["name"], call["args"]), TOOL_TIMEOUT
                        )
                    except ToolError as error:
                        return error
                    except TimeoutError:
                        return ToolError("That lookup took too long.")
                    except (httpx.HTTPError, SQLAlchemyError, ValueError, KeyError):
                        return ToolError("That lookup failed.")

                before = len(ledger.sources)
                results = await asyncio.gather(*(guarded(call) for call in runnable))
                for call, sid, result in zip(runnable, ids, results, strict=True):
                    if isinstance(result, ToolError):
                        yield core.event(
                            "step", id=sid, tool=call["name"], status="error", detail=str(result)
                        )
                        content = f"Error: {result}"
                    else:
                        fields, content, extra = result
                        yield core.event("step", id=sid, status="done", **fields)
                        for line in extra:
                            yield line
                    messages.append(
                        {"role": "tool", "tool_call_id": call["id"], "content": content}
                    )
                for call in named[MAX_CALLS_PER_STEP:]:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "content": "Skipped: too many lookups at once.",
                        }
                    )
                if len(ledger.sources) > before:
                    yield core.event("sources", sources=ledger.sources)
                continue

            if answer is None and held:
                answer = core.AnswerStream(ledger.ids)
                timing["firstAnswerMs"] = round((perf_counter() - started) * 1000, 1)
                visible = answer.feed(held)
                if visible:
                    yield core.event("delta", text=visible)
            if answer is None:
                # No text: the model kept calling tools on its last step, or said nothing.
                timing["fallback"] = "no answer text"
                async for line in core.single_pass(
                    question, key, request, started, timing, ledger.sources, library
                ):
                    yield line
                return
            break

        assert answer is not None
        rest, follow_ups = answer.finish()
        if rest:
            yield core.event("delta", text=rest)
        if follow_ups:
            yield core.event("followups", questions=follow_ups)
        timing.update(
            steps=steps,
            toolCalls=tool_calls,
            prefetchHits=turn.prefetch_hits,
            strippedCitations=answer.guard.stripped,
            totalMs=round((perf_counter() - started) * 1000, 1),
        )
        yield core.event("done", metrics=timing)
    except (ModelError, httpx.HTTPError, KeyError, ValueError, SQLAlchemyError):
        yield core.event("error", detail=core.COMPANION_ERROR)
    finally:
        for task in turn.prefetch.values():
            task.cancel()
