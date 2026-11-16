"""Agent eval for Ask Margin: does the model pick the right tools, and what does it cost?

Run with PostgreSQL up and the API started in agent mode:
    MARGIN_AGENT=1 uv run fastapi dev main.py            # in one terminal
    NVIDIA_API_KEY=... uv run --directory api python benchmarks/agent_eval.py [--verbose]

It first sends one request straight to the NVIDIA API to check that the chat model
returns tool calls at all. Then it indexes a small synthetic library and some dated
notes under a fresh clientId (all deleted afterwards), asks 15 hand-written questions,
and reports:
  - tool-selection accuracy: every expected tool was used; for questions the passage
    already answers, no tool was used (extra tools on other questions are reported,
    not penalised)
  - average model steps and tool calls per answer
  - % answers whose citations all point at sources shown in that turn
  - fallbacks to the single-pass path
  - p50/p95 time to first answer text and to the full answer, measured by this client

Results are written to benchmarks/results/agent_eval.json. Fifteen questions is a smoke
test, not a benchmark: one changed answer moves accuracy by ~7 points.
"""

import asyncio
import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grounding_eval import LIBRARY  # noqa: E402

API = os.getenv("MARGIN_API_URL", "http://127.0.0.1:8000")
NVIDIA = os.getenv("MARGIN_NVIDIA_URL", "https://integrate.api.nvidia.com/v1")
MODEL = os.getenv(
    "MARGIN_AGENT_MODEL", os.getenv("MARGIN_CHAT_MODEL", "nvidia/nemotron-3.5-bolt-e4b")
)
OUTPUT = Path(__file__).resolve().parent / "results" / "agent_eval.json"

CURRENT_URL = "eval:current"
CURRENT_TITLE = "Notes on Deep Work"
CURRENT = [
    "Notes on Deep Work",
    "Deep work is focused effort on a demanding task without distraction. The author "
    "argues it is becoming rarer just as it becomes more valuable.",
    "Every switch to a quick message costs more than the seconds it takes, because the "
    "mind keeps part of its attention on the interruption.",
    "The proposed remedy is to schedule long blocks for one task and to batch shallow "
    "work into short, fixed windows.",
    "Later chapters return to email specifically: the author suggests checking it twice "
    "a day and treating the inbox as a queue, not a conversation.",
]
PARAGRAPH = 2  # the passage the reader is on

NOTES = [
    ("deadlines", "eval:deadlines", 1, 3, "Deadlines helped my team only when the goal was clear."),
    ("residue", "eval:residue", 0, 10, "Attention residue explains why my mornings feel foggy."),
    ("sleep", "eval:sleep", 0, 60, "I should protect sleep before exams; recall drops a lot."),
]


@dataclass
class Case:
    question: str
    expects: set[str] = field(default_factory=set)  # empty: the passage is enough
    scope: str = "passage"
    gap: bool = False


CASES = [
    Case("What does this passage mean?"),
    Case("Explain this in plain words."),
    Case("What is the strongest objection to this paragraph?"),
    Case("What did I think about deadlines last week?", {"search_notes"}),
    Case("Have I written any notes about sleep?", {"search_notes"}),
    Case("What were my thoughts on attention residue?", {"search_notes"}),
    Case("Do other articles I've read say anything about time pressure?", {"search_library"}),
    Case("What did Sophie Leroy find?", {"search_library"}),
    Case("How does this connect to what I've read about memory?", {"search_library"}),
    Case("Where else in this article does the author talk about email?", {"search_article"}),
    Case("Does the author come back to batching later in the article?", {"search_article"}),
    Case(
        "Read the passages around this one and tell me how the argument builds.",
        {"get_passage"},
    ),
    Case(
        "Save a note on this passage saying that switching costs are underrated.",
        {"propose_note"},
    ),
    Case(
        "Compare my notes on deadlines with what my library says about them.",
        {"search_notes", "search_library"},
    ),
    Case("What have I read about quantum computing?", {"search_library"}, gap=True),
]


def percentile(values: list[float], share: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, round(share * len(ordered) + 0.5) - 1))]


async def probe(client: httpx.AsyncClient, key: str) -> dict[str, Any]:
    """One direct request: does the model return a tool call when it clearly should?"""
    started = perf_counter()
    response = await client.post(
        f"{NVIDIA}/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": MODEL,
            "temperature": 0,
            "max_tokens": 200,
            "messages": [{"role": "user", "content": "Search my notes for 'deadlines'."}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "search_notes",
                        "description": "Search the user's notes.",
                        "parameters": {
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                            "required": ["query"],
                        },
                    },
                }
            ],
            "tool_choice": "auto",
        },
    )
    elapsed = round((perf_counter() - started) * 1000, 1)
    if response.status_code >= 400:
        return {"ok": False, "status": response.status_code, "ms": elapsed}
    message = response.json()["choices"][0]["message"]
    calls = message.get("tool_calls") or []
    return {
        "ok": bool(calls),
        "ms": elapsed,
        "call": calls[0]["function"] if calls else None,
        "text": (message.get("content") or "")[:200],
    }


async def ask(client: httpx.AsyncClient, key: str, owner: str, case: Case) -> dict[str, Any]:
    start = max(0, PARAGRAPH - 2)
    payload = {
        "clientId": owner,
        "question": case.question,
        "articleUrl": CURRENT_URL,
        "title": CURRENT_TITLE,
        "paragraphIndex": PARAGRAPH,
        "contextStart": start,
        "context": CURRENT[start : PARAGRAPH + 2],
        "scope": case.scope,
    }
    answer, sources, metrics, error = "", [], {}, None
    steps: dict[int, dict[str, Any]] = {}
    proposal = None
    first_ms = None
    started = perf_counter()
    async with client.stream(
        "POST", f"{API}/companion/ask", json=payload, headers={"Authorization": f"Bearer {key}"}
    ) as response:
        async for line in response.aiter_lines():
            if not line.strip():
                continue
            event = json.loads(line)
            kind = event["type"]
            if kind == "delta":
                if first_ms is None:
                    first_ms = round((perf_counter() - started) * 1000, 1)
                answer += event["text"]
            elif kind == "sources":
                sources = event["sources"]
            elif kind == "step":
                steps.setdefault(event["id"], {}).update(event)
            elif kind == "proposal":
                proposal = event["proposal"]
            elif kind == "done":
                metrics = event["metrics"]
            elif kind == "error":
                error = event["detail"]
    total_ms = round((perf_counter() - started) * 1000, 1)
    used = {step["tool"] for step in steps.values()}
    ids = {source["id"] for source in sources}
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
    correct = used == set() if not case.expects else case.expects <= used
    return {
        "question": case.question,
        "expected": sorted(case.expects),
        "used": sorted(used),
        "correct": correct,
        "extra": sorted(used - case.expects) if case.expects else [],
        "steps": metrics.get("steps"),
        "toolCalls": metrics.get("toolCalls", len(steps)),
        "fallback": metrics.get("fallback"),
        "agent": bool(metrics.get("agent")),
        "citationsValid": cited <= ids,
        "strippedCitations": metrics.get("strippedCitations", 0),
        "proposal": proposal,
        "firstAnswerMs": first_ms,
        "totalMs": total_ms,
        "serverFirstAnswerMs": metrics.get("firstAnswerMs"),
        "error": error,
        "answer": answer,
        "stepEvents": list(steps.values()),
    }


async def seed(client: httpx.AsyncClient, key: str, owner: str) -> None:
    headers = {"Authorization": f"Bearer {key}"}
    library = {**LIBRARY, CURRENT_URL: (CURRENT_TITLE, CURRENT)}
    for url, (title, paragraphs) in library.items():
        response = await client.post(
            f"{API}/companion/index",
            headers=headers,
            json={"clientId": owner, "url": url, "title": title, "paragraphs": paragraphs},
        )
        response.raise_for_status()
    for name, url, index, days_ago, text_value in NOTES:
        title, paragraphs = LIBRARY[url]
        source_key = f"eval-{name}-{owner[:8]}"
        response = await client.put(
            f"{API}/thoughts/{source_key}",
            headers=headers,
            json={
                "clientId": owner,
                "sourceKey": source_key,
                "note": text_value,
                "paragraphText": paragraphs[index],
                "paragraphIndex": index,
                "articleTitle": title,
                "articleUrl": url,
                "createdAt": (datetime.now(UTC) - timedelta(days=days_ago)).isoformat(),
            },
        )
        response.raise_for_status()


def cleanup(owner: str) -> None:
    from sqlalchemy import delete
    from sqlalchemy.orm import Session

    from companion import Passage
    from memory import ThoughtRecord, engine

    with Session(engine) as session:
        session.execute(delete(Passage).where(Passage.client_id == owner))
        session.execute(delete(ThoughtRecord).where(ThoughtRecord.client_id == owner))
        session.commit()


async def main() -> None:
    key = os.environ.get("NVIDIA_API_KEY")
    if not key:
        sys.exit("Set NVIDIA_API_KEY to run the eval.")
    verbose = "--verbose" in sys.argv
    owner = str(uuid4())
    results: list[dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=120) as client:
        tool_probe = await probe(client, key)
        print(f"model {MODEL}")
        print(
            f"probe: tool call {'returned' if tool_probe['ok'] else 'NOT returned'} "
            f"in {tool_probe['ms']} ms {tool_probe.get('call') or tool_probe.get('status', '')}\n"
        )
        try:
            await seed(client, key, owner)
            for case in CASES:
                result = await ask(client, key, owner, case)
                results.append(result)
                if not result["agent"] and not result["error"]:
                    sys.exit("The API answered without agent mode. Restart it with MARGIN_AGENT=1.")
                mark = "ok " if result["correct"] else "MISS"
                notes = []
                if result["extra"]:
                    notes.append(f"also {', '.join(result['extra'])}")
                if result["fallback"]:
                    notes.append(f"fallback: {result['fallback']}")
                if not result["citationsValid"] or result["strippedCitations"]:
                    notes.append(f"citations stripped: {result['strippedCitations']}")
                if result["error"]:
                    notes.append(f"error: {result['error']}")
                print(
                    f"{mark} {result['question'][:58]:<58} "
                    f"used={','.join(result['used']) or '-':<28} "
                    f"{result['firstAnswerMs'] or 0:>7.0f} ms  {'; '.join(notes)}"
                )
                if verbose:
                    print("    ", result["answer"].replace("\n", " ")[:300])
        finally:
            cleanup(owner)

    answered = [r for r in results if not r["error"]]
    first = [r["firstAnswerMs"] for r in answered if r["firstAnswerMs"] is not None]
    total = [r["totalMs"] for r in answered]
    no_tool_cases = [r for r in results if not r["expected"]]
    summary = {
        "model": MODEL,
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "cases": len(results),
        "probe": tool_probe,
        "toolSelectionAccuracy": sum(r["correct"] for r in results) / len(results),
        "noToolCasesAnsweredWithoutTools": sum(r["correct"] for r in no_tool_cases),
        "noToolCases": len(no_tool_cases),
        "averageSteps": sum(r["steps"] or 0 for r in answered) / max(1, len(answered)),
        "averageToolCalls": sum(r["toolCalls"] or 0 for r in answered) / max(1, len(answered)),
        "validCitationShare": sum(
            r["citationsValid"] and not r["strippedCitations"] for r in answered
        )
        / max(1, len(answered)),
        "fallbacks": sum(bool(r["fallback"]) for r in results),
        "errors": sum(bool(r["error"]) for r in results),
        "firstAnswerMs": {"p50": percentile(first, 0.5), "p95": percentile(first, 0.95)},
        "totalMs": {"p50": percentile(total, 0.5), "p95": percentile(total, 0.95)},
    }
    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text(json.dumps({"summary": summary, "results": results}, indent=2))
    print(
        f"\ntool selection   {sum(r['correct'] for r in results)}/{len(results)} "
        f"({summary['toolSelectionAccuracy']:.0%}); passage-only questions answered "
        f"without tools {summary['noToolCasesAnsweredWithoutTools']}/{summary['noToolCases']}"
    )
    print(
        f"steps            {summary['averageSteps']:.2f} model calls, "
        f"{summary['averageToolCalls']:.2f} tool calls per answer"
    )
    print(f"valid citations  {summary['validCitationShare']:.0%} of answers")
    print(f"fallbacks        {summary['fallbacks']}   errors {summary['errors']}")
    print(
        f"first answer     p50 {summary['firstAnswerMs']['p50']} ms, "
        f"p95 {summary['firstAnswerMs']['p95']} ms (n={len(first)}; p95 of so few is ~the max)"
    )
    print(
        f"full answer      p50 {summary['totalMs']['p50']} ms, p95 {summary['totalMs']['p95']} ms"
    )
    print(
        f"\nsaved {OUTPUT.relative_to(Path.cwd()) if OUTPUT.is_relative_to(Path.cwd()) else OUTPUT}"
    )


if __name__ == "__main__":
    asyncio.run(main())
