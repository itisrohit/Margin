import json
import os
import unittest
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import httpx
from fastapi import FastAPI

import agent
import companion

ARTICLE = "https://example.com/article"


def sse(*chunks: dict[str, Any]) -> str:
    return "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"


def text(content: str) -> dict[str, Any]:
    return {"choices": [{"delta": {"content": content}}]}


def tool_call(name: str, arguments: dict[str, Any] | str, index: int = 0) -> dict[str, Any]:
    raw = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {
                            "index": index,
                            "id": f"call_{name}_{index}",
                            "type": "function",
                            "function": {"name": name, "arguments": raw},
                        }
                    ]
                }
            }
        ]
    }


def note(text_value: str, url: str = "https://example.com/other", days_ago: int = 1):
    return {
        "kind": "thought",
        "title": "Other article",
        "url": url,
        "createdAt": (datetime.now(UTC) - timedelta(days=days_ago)).isoformat(),
        "paragraphIndex": 4,
        "text": text_value,
    }


def question(**overrides: Any) -> companion.Question:
    values: dict[str, Any] = {
        "clientId": str(uuid4()),
        "question": "Explain this claim.",
        "articleUrl": ARTICLE,
        "title": "Evidence",
        "paragraphIndex": 1,
        "contextStart": 0,
        "context": ["Intro.", "Pressure helps practiced tasks.", "It harms creative ones."],
    }
    values.update(overrides)
    return companion.Question.model_validate(values)


def make_turn(**overrides: Any) -> agent.Turn:
    q = question(**overrides)
    sources = companion.number_sources(
        [
            {
                "kind": "article",
                "title": q.title,
                "url": q.article_url,
                "paragraphIndex": q.context_start + i,
                "text": t,
            }
            for i, t in enumerate(q.context)
        ]
    )
    return agent.Turn(q, "test", agent.Ledger(sources))


class ToolTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.patches = [
            patch.object(companion, "query_vector", AsyncMock(return_value=([0.0], False))),
            patch.object(companion, "ensure_search_index"),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()

    async def test_search_article_is_scoped_to_reader_and_article(self):
        turn = make_turn()
        found = [
            {
                "kind": "article",
                "title": "Evidence",
                "url": ARTICLE,
                "paragraphIndex": 9,
                "text": "Later, the author returns to deadlines.",
            }
        ]
        with patch.object(companion, "search", return_value=found) as search:
            step, content, extra = await agent.run_tool(
                turn, "search_article", '{"query": "deadlines"}'
            )
        self.assertEqual(search.call_args.args[0], turn.question.client_id)
        self.assertEqual(search.call_args.kwargs["article_url"], ARTICLE)
        self.assertFalse(search.call_args.kwargs["thoughts"])
        self.assertEqual(step, {"tool": "search_article", "query": "deadlines", "results": 1})
        # New sources continue the turn's numbering, and the model sees that number.
        self.assertIn("[4]", content)
        added = turn.ledger.get(4)
        assert added is not None
        self.assertEqual(added["paragraphIndex"], 9)
        self.assertEqual(extra, [])

    async def test_search_library_spans_articles_and_notes_respect_days(self):
        turn = make_turn()
        with patch.object(companion, "search", return_value=[]) as search:
            await agent.run_tool(turn, "search_library", '{"query": "creativity"}')
            self.assertNotIn("article_url", search.call_args.kwargs)
            self.assertEqual(search.call_args.args[0], turn.question.client_id)
            await agent.run_tool(turn, "search_notes", '{"query": "creativity", "days": 7}')
        options = search.call_args.kwargs
        self.assertFalse(options["passages"])
        since = options["since"]
        self.assertAlmostEqual((datetime.now(UTC) - since).total_seconds(), 7 * 86400, delta=60)

    async def test_empty_results_and_known_sources_keep_ids(self):
        turn = make_turn()
        with patch.object(companion, "search", return_value=[]):
            step, content, _ = await agent.run_tool(turn, "search_notes", '{"query": "focus"}')
        self.assertEqual((step["results"], content), (0, "No matches."))
        existing = turn.ledger.get(2)
        assert existing is not None
        known = dict(existing)
        known.pop("id")
        with patch.object(companion, "search", return_value=[known]):
            _, content, _ = await agent.run_tool(turn, "search_article", '{"query": "pressure"}')
        self.assertEqual(content, "[2] (already shown above)")
        self.assertEqual(len(turn.ledger.sources), 3)

    async def test_invalid_arguments_and_unknown_ids_are_tool_errors(self):
        turn = make_turn()
        for name, raw in [
            ("search_notes", "{not json"),
            ("search_library", '{"query": ""}'),
            ("get_passage", '{"passage_id": 99}'),
            ("delete_everything", "{}"),
        ]:
            with self.subTest(name=name), self.assertRaises(agent.ToolError):
                await agent.run_tool(turn, name, raw)

    async def test_get_passage_returns_neighbours_from_the_window_when_not_indexed(self):
        turn = make_turn()
        session = MagicMock()
        session.__enter__.return_value.scalars.return_value.all.return_value = []
        with patch.object(agent, "Session", return_value=session):
            step, content, _ = await agent.run_tool(turn, "get_passage", '{"passage_id": 3}')
        # Passage [3] is paragraph 2; its neighbour 1 is already known, 3 is outside.
        self.assertEqual(step["paragraphIndex"], 2)
        self.assertIn("[2]", content)

    async def test_propose_note_only_returns_a_pending_proposal(self):
        turn = make_turn()
        with (
            patch.object(agent, "Session", side_effect=AssertionError("no DB access")),
            patch("memory.Session", side_effect=AssertionError("no DB access")),
        ):
            step, content, extra = await agent.run_tool(
                turn, "propose_note", '{"passage_id": 2, "text": "Pressure cuts both ways."}'
            )
        proposal = turn.proposal
        assert proposal is not None
        self.assertEqual(proposal["status"], "pending")
        self.assertEqual(proposal["paragraphIndex"], 1)
        self.assertIn("nothing is saved", content)
        self.assertEqual(json.loads(extra[0])["type"], "proposal")
        self.assertEqual(step["results"], 1)
        with self.assertRaises(agent.ToolError):
            await agent.run_tool(turn, "propose_note", '{"passage_id": 2, "text": "Again."}')

    async def test_propose_note_rejects_other_articles(self):
        turn = make_turn()
        turn.ledger.add([note("A thought elsewhere.")])
        with self.assertRaises(agent.ToolError):
            await agent.run_tool(turn, "propose_note", '{"passage_id": 4, "text": "Hmm."}')
        self.assertIsNone(turn.proposal)


class CitationGuardTests(unittest.TestCase):
    def test_unknown_citations_are_stripped_even_when_split(self):
        guard = companion.CitationGuard({1, 2})
        out = "".join(
            guard.feed(part) for part in ["Pressure helps [1", "] but not always [", "9]."]
        )
        out += guard.finish()
        self.assertEqual(out, "Pressure helps [1] but not always.")
        self.assertEqual(guard.stripped, 1)

    def test_brackets_that_are_not_citations_pass_through(self):
        guard = companion.CitationGuard({1})
        out = guard.feed("A list [see below] and [2a].") + guard.finish()
        self.assertEqual(out, "A list [see below] and [2a].")


class SpeculationTests(unittest.TestCase):
    def test_hints_and_similarity(self):
        self.assertTrue(agent.NOTES_HINT.search("What did I think about this last week?"))
        self.assertTrue(agent.LIBRARY_HINT.search("How does this connect to what I've read?"))
        self.assertFalse(agent.NOTES_HINT.search("What does the author mean by residue?"))
        self.assertTrue(agent.similar("pressure creativity", "Does pressure hurt creativity?"))
        self.assertFalse(agent.similar("sleep memory", "Does pressure hurt creativity?"))


class AgentLoopTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        app = FastAPI()
        app.include_router(companion.router)
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://margin.test"
        )
        self.body = question().model_dump(by_alias=True, mode="json")
        self.requests: list[dict[str, Any]] = []
        self.env = patch.dict(os.environ, {"MARGIN_AGENT": "1"})
        self.env.start()

    async def asyncTearDown(self):
        self.env.stop()
        await self.client.aclose()

    async def run_with(self, responses: list[httpx.Response | str], **patches: Any):
        script = iter(responses)

        def provider(request: httpx.Request) -> httpx.Response:
            self.requests.append(json.loads(request.content))
            reply = next(script)
            return reply if isinstance(reply, httpx.Response) else httpx.Response(200, text=reply)

        async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as upstream:
            with (
                patch.object(companion, "_client", upstream),
                patch.object(companion, "query_vector", AsyncMock(return_value=([0.0], False))),
                patch.object(companion, "ensure_search_index"),
                patch.object(companion, "search", **patches),
            ):
                response = await self.client.post(
                    "/companion/ask", json=self.body, headers={"Authorization": "Bearer test"}
                )
        return [json.loads(line) for line in response.text.splitlines()]

    async def test_model_calls_a_tool_then_answers_with_its_citation(self):
        self.body["question"] = "Have I written anything about creativity before?"
        events = await self.run_with(
            [
                sse(tool_call("search_notes", {"query": "creativity"})),
                sse(
                    text("You noted that deadlines can help when goals are clear [4]. "),
                    text("That fits this passage [2]."),
                ),
            ],
            return_value=[note("Deadlines help creative work if the goal is clear.")],
        )
        kinds = [e["type"] for e in events]
        self.assertEqual(kinds[0], "sources")
        steps = [e for e in events if e["type"] == "step"]
        self.assertEqual([s["status"] for s in steps], ["running", "done"])
        self.assertEqual(steps[1]["results"], 1)
        self.assertEqual(events[-1]["type"], "done")
        answer = "".join(e["text"] for e in events if e["type"] == "delta")
        self.assertIn("[4]", answer)
        latest = [e for e in events if e["type"] == "sources"][-1]["sources"]
        self.assertEqual(latest[3]["kind"], "thought")
        metrics = events[-1]["metrics"]
        self.assertEqual((metrics["steps"], metrics["toolCalls"]), (2, 1))
        self.assertTrue(metrics["agent"])
        # The tool result went back to the model under the call's id.
        tool_message = self.requests[1]["messages"][-1]
        self.assertEqual(tool_message["role"], "tool")
        self.assertEqual(tool_message["tool_call_id"], "call_search_notes_0")
        self.assertTrue(self.requests[0]["parallel_tool_calls"])

    async def test_simple_question_answers_in_one_call(self):
        events = await self.run_with(
            [sse(text("The author means pressure narrows attention onto routines [2]."))],
            return_value=[],
        )
        self.assertEqual(len(self.requests), 1)
        self.assertNotIn("step", [e["type"] for e in events])
        self.assertEqual(events[-1]["metrics"]["toolCalls"], 0)

    async def test_step_cap_forces_an_answer(self):
        looping = sse(tool_call("search_library", {"query": "pressure"}))
        events = await self.run_with(
            [looping, looping, looping, sse(text("Here is what I found so far, briefly [2]."))],
            return_value=[],
        )
        self.assertEqual(len(self.requests), agent.MAX_STEPS)
        final = self.requests[-1]
        self.assertEqual(final["tool_choice"], "none")
        self.assertIn("no more lookups", final["messages"][-1]["content"])
        metrics = events[-1]["metrics"]
        self.assertEqual(metrics["stoppedEarly"], "step cap")
        self.assertEqual(metrics["toolCalls"], agent.MAX_STEPS - 1)

    async def test_model_still_calling_tools_on_last_step_falls_back(self):
        looping = sse(tool_call("search_library", {"query": "pressure"}))
        events = await self.run_with(
            [
                looping,
                looping,
                looping,
                looping,
                sse(text("A plain answer from the evidence [1].")),
            ],
            return_value=[],
        )
        self.assertEqual(len(self.requests), agent.MAX_STEPS + 1)
        self.assertNotIn("tools", self.requests[-1])
        self.assertEqual(events[-1]["metrics"]["fallback"], "no answer text")

    async def test_tool_error_is_reported_to_the_model_not_the_reader(self):
        events = await self.run_with(
            [
                sse(tool_call("search_library", {"query": "pressure"})),
                sse(text("I couldn't search your library just now, but this passage says [2].")),
            ],
            side_effect=companion.SQLAlchemyError("database down"),
        )
        steps = [e for e in events if e["type"] == "step"]
        self.assertEqual(steps[-1]["status"], "error")
        self.assertIn("Error:", self.requests[1]["messages"][-1]["content"])
        self.assertEqual(events[-1]["type"], "done")

    async def test_unsupported_tool_calling_falls_back_to_single_pass(self):
        events = await self.run_with(
            [
                httpx.Response(400, json={"error": "tools not supported"}),
                sse(text("It depends on the task [1].")),
            ],
            return_value=[],
        )
        self.assertIn("tools", self.requests[0])
        self.assertNotIn("tools", self.requests[1])
        self.assertEqual(events[-1]["type"], "done")
        self.assertIn("400", events[-1]["metrics"]["fallback"])
        answer = "".join(e["text"] for e in events if e["type"] == "delta")
        self.assertEqual(answer, "It depends on the task [1].")

    async def test_citation_to_an_id_never_shown_is_stripped(self):
        events = await self.run_with(
            [sse(text("Pressure narrows attention onto routines [2], as studies show [17]."))],
            return_value=[],
        )
        answer = "".join(e["text"] for e in events if e["type"] == "delta")
        self.assertEqual(answer, "Pressure narrows attention onto routines [2], as studies show.")
        self.assertEqual(events[-1]["metrics"]["strippedCitations"], 1)

    async def test_propose_note_emits_a_proposal_and_never_writes(self):
        self.body["question"] = "Save a note that pressure cuts both ways."
        with patch("memory.Session", side_effect=AssertionError("no DB writes")):
            events = await self.run_with(
                [
                    sse(
                        tool_call(
                            "propose_note", {"passage_id": 2, "text": "Pressure cuts both ways."}
                        )
                    ),
                    sse(text("I've drafted that note for you to approve, on this passage [2].")),
                ],
                return_value=[],
            )
        proposal = next(e for e in events if e["type"] == "proposal")["proposal"]
        self.assertEqual(proposal["status"], "pending")
        self.assertEqual(proposal["paragraphIndex"], 1)
        self.assertEqual(events[-1]["type"], "done")

    async def test_save_request_forces_a_draft_on_the_current_passage(self):
        self.body["question"] = "Save a note here saying switching costs are underrated."
        events = await self.run_with(
            [
                sse(tool_call("propose_note", {"text": "Switching costs are underrated."})),
                sse(text("I've drafted that for you to approve; it builds on this passage [2].")),
            ],
            return_value=[],
        )
        first = self.requests[0]
        self.assertEqual(first["tool_choice"]["function"]["name"], "propose_note")
        self.assertIn("reader is on passage [2]", first["messages"][-1]["content"])
        self.assertEqual(self.requests[1]["tool_choice"], "auto")
        proposal = next(e for e in events if e["type"] == "proposal")["proposal"]
        self.assertEqual(proposal["paragraphIndex"], self.body["paragraphIndex"])
        self.assertEqual(proposal["text"], "Switching costs are underrated.")

    async def test_rejected_forced_tool_retries_on_auto(self):
        self.body["question"] = "Could you save a note that pressure cuts both ways?"
        events = await self.run_with(
            [
                httpx.Response(400, json={"error": "named tool_choice unsupported"}),
                sse(tool_call("propose_note", {"text": "Pressure cuts both ways."})),
                sse(text("Drafted below for you to approve, tied to this passage [2].")),
            ],
            return_value=[],
        )
        self.assertEqual(self.requests[1]["tool_choice"], "auto")
        self.assertIn("proposal", [e["type"] for e in events])
        self.assertEqual(events[-1]["metrics"]["forcedToolRejected"], 400)
        self.assertNotIn("fallback", events[-1]["metrics"])

    def test_save_hint(self):
        for asked in [
            "Save a note here saying switching costs are underrated.",
            "please note this down",
            "Can you add a note that this is weak?",
            "Remember that I disagree with this.",
        ]:
            self.assertTrue(agent.SAVE_HINT.search(asked), asked)
        for asked in ["What notes have I saved?", "Is this worth a note?"]:
            self.assertFalse(agent.SAVE_HINT.search(asked), asked)

    async def test_request_flag_overrides_the_server_default(self):
        with patch.dict(os.environ, {"MARGIN_AGENT": "0"}):
            self.body["agent"] = True
            await self.run_with([sse(text("Agent answer [1]."))], return_value=[])
        self.assertIn("tools", self.requests[-1])
        self.body["agent"] = False  # server default is on in this test class
        events = await self.run_with([sse(text("Plain answer [1]."))], return_value=[])
        self.assertNotIn("tools", self.requests[-1])
        self.assertNotIn("agent", events[-1]["metrics"])

    async def test_agent_disabled_uses_single_pass(self):
        with patch.dict(os.environ, {"MARGIN_AGENT": "0"}):
            events = await self.run_with([sse(text("Plain answer [1]."))], return_value=[])
        self.assertNotIn("tools", self.requests[0])
        self.assertNotIn("agent", events[-1]["metrics"])


@unittest.skipUnless(os.getenv("RUN_POSTGRES_INTEGRATION") == "1", "needs PostgreSQL")
class AgentSearchIntegrationTests(unittest.TestCase):
    def setUp(self):
        from sqlalchemy.orm import Session

        from memory import EMBEDDING_DIMENSIONS, ThoughtRecord, engine, ensure_schema

        ensure_schema()
        companion.ensure_search_index()
        self.owner, self.stranger = uuid4(), uuid4()
        self.engine = engine
        self.vector = [1.0] + [0.0] * (EMBEDDING_DIMENSIONS - 1)

        def passage(owner, url, position):
            return companion.Passage(
                client_id=owner,
                article_url=url,
                position=position,
                fragment=0,
                version="v",
                title=url,
                content=f"Focus passage {position}",
                embedding=self.vector,
            )

        def thought(owner, days_ago):
            return ThoughtRecord(
                client_id=owner,
                source_key=f"k{days_ago}{uuid4().hex[:6]}",
                note=f"Focus note from {days_ago} days ago",
                paragraph_text="p",
                paragraph_index=0,
                article_title="A",
                article_url="https://a",
                created_at=datetime.now(UTC) - timedelta(days=days_ago),
                embedding=self.vector,
            )

        with Session(engine) as session:
            session.add_all(
                [
                    *(passage(self.owner, "https://a", i) for i in range(4)),
                    passage(self.owner, "https://b", 0),
                    passage(self.stranger, "https://a", 9),
                    thought(self.owner, 2),
                    thought(self.owner, 40),
                    thought(self.stranger, 1),
                ]
            )
            session.commit()

    def tearDown(self):
        from sqlalchemy import delete
        from sqlalchemy.orm import Session

        from memory import ThoughtRecord

        with Session(self.engine) as session:
            for owner in (self.owner, self.stranger):
                session.execute(
                    delete(companion.Passage).where(companion.Passage.client_id == owner)
                )
                session.execute(delete(ThoughtRecord).where(ThoughtRecord.client_id == owner))
            session.commit()

    def test_article_scope_owner_scope_and_since(self):
        in_article = companion.search(
            self.owner,
            self.vector,
            "focus",
            article_url="https://a",
            thoughts=False,
            max_passages=6,
            per_article=6,
        )
        self.assertEqual({s["url"] for s in in_article}, {"https://a"})
        self.assertEqual(sorted(s["paragraphIndex"] for s in in_article), [0, 1, 2, 3])
        recent = companion.search(
            self.owner,
            self.vector,
            "focus",
            passages=False,
            since=datetime.now(UTC) - timedelta(days=7),
        )
        self.assertEqual([s["text"] for s in recent], ["Focus note from 2 days ago"])

    def test_get_passage_reads_neighbours_for_this_reader_only(self):
        turn = make_turn(clientId=str(self.owner), articleUrl="https://a")
        found = agent.neighbours(turn, {"kind": "article", "url": "https://a", "paragraphIndex": 2})
        self.assertEqual([s["paragraphIndex"] for s in found], [1, 2, 3])
