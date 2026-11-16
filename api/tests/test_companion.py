import asyncio
import json
import unittest
from typing import Any
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import httpx
from fastapi import FastAPI

import companion


class CompanionContractTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.app = FastAPI()
        self.app.include_router(companion.router)
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app), base_url="http://margin.test"
        )
        self.question: dict[str, Any] = {
            "clientId": str(uuid4()),
            "question": "Explain this claim.",
            "articleUrl": "https://example.com/article",
            "title": "Evidence",
            "paragraphIndex": 0,
            "contextStart": 0,
            "context": ["Pressure can help practiced tasks but can harm creative tasks."],
        }

    async def asyncTearDown(self):
        await self.client.aclose()
        companion._cache.clear()
        companion._inflight.clear()

    async def test_missing_key_rejected_before_streaming(self):
        response = await self.client.post("/companion/ask", json=self.question)
        self.assertEqual(response.status_code, 401)

    async def test_stream_contract_citations_and_timing(self):
        def provider(request):
            payload = json.loads(request.content)
            self.assertTrue(payload["stream"])
            self.assertIn("untrusted", payload["messages"][0]["content"])
            return httpx.Response(
                200,
                text=(
                    'data: {"choices":[{"delta":{"content":"It depends on the task [1]."}}]}\n\n'
                    "data: [DONE]\n\n"
                ),
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as upstream:
            with patch.object(companion, "_client", upstream):
                response = await self.client.post(
                    "/companion/ask", json=self.question, headers={"Authorization": "Bearer test"}
                )
        events = [json.loads(line) for line in response.text.splitlines()]
        self.assertEqual([e["type"] for e in events], ["sources", "delta", "done"])
        self.assertEqual(events[0]["sources"][0]["text"], self.question["context"][0])
        metrics = events[-1]["metrics"]
        self.assertGreaterEqual(metrics["totalMs"], metrics["firstAnswerMs"])
        self.assertNotIn("embeddingMs", metrics)

    async def test_upstream_failure_is_visible_without_done(self):
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(429, json={"error": "limited"}))
        ) as upstream:
            with patch.object(companion, "_client", upstream):
                response = await self.client.post(
                    "/companion/ask", json=self.question, headers={"Authorization": "Bearer test"}
                )
        events = [json.loads(line) for line in response.text.splitlines()]
        self.assertEqual(events[-1]["type"], "error")
        self.assertNotIn("done", [event["type"] for event in events])

    async def test_query_cache_is_scoped_and_concurrent_requests_deduplicated(self):
        count = 0

        async def provider(_):
            nonlocal count
            count += 1
            await asyncio.sleep(0.01)
            return httpx.Response(
                200,
                json={"data": [{"index": 0, "embedding": [0.0] * companion.EMBEDDING_DIMENSIONS}]},
            )

        owner = uuid4()
        async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as upstream:
            with patch.object(companion, "_client", upstream):
                await asyncio.gather(
                    *(companion.query_vector("same question", "key", owner) for _ in range(3))
                )
                self.assertEqual(count, 1)
                _, hit = await companion.query_vector("same question", "key", owner)
                self.assertTrue(hit)
                await companion.query_vector("same question", "other-key", owner)
                await companion.query_vector("same question", "key", uuid4())
                self.assertEqual(count, 3)

    async def test_library_scope_uses_owner_and_returns_retrieved_sources(self):
        owner = self.question["clientId"]

        def provider(_):
            return httpx.Response(
                200,
                text=(
                    'data: {"choices":[{"delta":{"content":"Compare the conditions [2]."}}]}\n\n'
                    "data: [DONE]\n\n"
                ),
            )

        evidence = {
            "kind": "article",
            "title": "Other article",
            "url": "https://example.com/other",
            "paragraphIndex": 3,
            "text": "Creativity responds differently to pressure.",
        }
        async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as upstream:
            with (
                patch.object(companion, "_client", upstream),
                patch.object(companion, "ensure_schema"),
                patch.object(companion, "query_vector", AsyncMock(return_value=([0.0], True))),
                patch.object(companion, "search", return_value=[evidence]) as search,
            ):
                response = await self.client.post(
                    "/companion/ask",
                    json={**self.question, "scope": "library"},
                    headers={"Authorization": "Bearer test"},
                )
        self.assertEqual(str(search.call_args.args[0]), owner)
        events = [json.loads(line) for line in response.text.splitlines()]
        self.assertEqual(events[0]["sources"][1]["url"], evidence["url"])
        self.assertEqual(events[-1]["type"], "done")
        self.assertTrue(events[-1]["metrics"]["queryCacheHit"])

    async def test_failed_embedding_batch_splits_without_losing_order(self):
        requests = []

        def provider(request):
            inputs = json.loads(request.content)["input"]
            requests.append(inputs)
            if len(inputs) > 1:
                return httpx.Response(500)
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "index": 0,
                            "embedding": [float(inputs[0])] * companion.EMBEDDING_DIMENSIONS,
                        }
                    ]
                },
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as upstream:
            with patch.object(companion, "_client", upstream):
                vectors = await companion.embeddings(["1", "2"], "test", "passage")
        self.assertEqual([v[0] for v in vectors], [1.0, 2.0])
        self.assertEqual(len(requests), 3)

    async def test_long_paragraph_chunks_preserve_tail_and_anchor(self):
        text = "a" * 10000 + "TAIL"
        parts = companion.chunks([text])
        self.assertTrue(parts[-1][2].endswith("TAIL"))
        self.assertTrue(all(p[0] == 0 and len(p[2]) <= 2400 for p in parts))
        self.assertEqual(len(set(p[1] for p in parts)), len(parts))

    async def test_invalid_scope_rejected(self):
        response = await self.client.post(
            "/companion/ask",
            json={**self.question, "scope": "web"},
            headers={"Authorization": "Bearer test"},
        )
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()


class CompanionPromptTests(unittest.TestCase):
    def setUp(self):
        self.question = companion.Question(
            clientId=uuid4(),
            question="What is your opinion on this?",
            articleUrl="https://example.com/a",
            title="Attention",
            paragraphIndex=3,
            contextStart=2,
            context=["Attention is selective focus.", "History"],
        )
        self.sources = [
            {
                "id": 1,
                "kind": "article",
                "title": "Attention",
                "url": "https://example.com/a",
                "paragraphIndex": 2,
                "text": "Attention is selective focus.",
            },
            {
                "id": 2,
                "kind": "article",
                "title": "Attention",
                "url": "https://example.com/a",
                "paragraphIndex": 3,
                "text": "History",
            },
            {
                "id": 3,
                "kind": "thought",
                "title": "Other",
                "url": "https://example.com/b",
                "paragraphIndex": 0,
                "text": "My note.",
            },
        ]

    def test_prompt_labels_sources_without_leaking_indexes(self):
        prompt = companion.build_prompt(self.question, self.sources)
        self.assertIn("[2] (the reader's selection; it is a short heading", prompt)
        self.assertIn("[1] (nearby in this article)", prompt)
        self.assertIn("[3] (the reader's own note", prompt)
        self.assertNotIn("paragraphIndex", prompt)
        self.assertNotIn("{", prompt)

    def test_system_prompt_invites_a_view(self):
        self.assertIn("give one", companion.SYSTEM_PROMPT)
        self.assertIn("Never mention paragraph indexes", companion.SYSTEM_PROMPT)


class CompanionMemoryTests(unittest.TestCase):
    def pair(self, n: int, size: int = 10) -> list[companion.Turn]:
        return [
            companion.Turn(role="user", content=f"question {n}"),
            companion.Turn(role="assistant", content="a" * size),
        ]

    def test_short_history_kept_verbatim(self):
        history = self.pair(1) + self.pair(2)
        kept, earlier = companion.fit_history(history, [])
        self.assertEqual(kept, history)
        self.assertEqual(earlier, [])

    def test_oldest_turns_become_recap_when_over_budget(self):
        history = [t for n in range(6) for t in self.pair(n, 6000)]
        kept, earlier = companion.fit_history(history, ["very first question"])
        self.assertEqual(kept[-2].content, "question 5")
        self.assertLessEqual(sum(len(t.content) for t in kept), companion.HISTORY_CHAR_BUDGET)
        self.assertEqual(kept[0].role, "user")
        self.assertEqual(earlier[0], "very first question")
        self.assertIn("question 0", earlier)
        self.assertIn("question 0", companion.recap(earlier))

    def test_recap_empty_without_earlier_questions(self):
        self.assertEqual(companion.recap([]), "")

    def test_request_accepts_long_conversation(self):
        question = companion.Question(
            clientId=uuid4(),
            question="And now?",
            articleUrl="https://example.com",
            title="T",
            paragraphIndex=0,
            contextStart=0,
            context=["text"],
            history=[t for n in range(16) for t in self.pair(n)],
            earlierQuestions=["q"] * 30,
        )
        self.assertEqual(len(question.history), 32)
