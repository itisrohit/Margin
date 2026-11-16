import os
import unittest
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

from fastapi import HTTPException
from sqlalchemy.dialects import postgresql

import memory


class FakeSession:
    def __init__(self, scalar_values=(), rows=()):
        self.scalar_values = iter(scalar_values)
        self.rows = rows
        self.executed = []
        self.committed = False

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def scalar(self, statement):
        self.executed.append(statement)
        return next(self.scalar_values, None)

    def execute(self, statement):
        self.executed.append(statement)
        return SimpleNamespace(all=lambda: self.rows)

    def commit(self):
        self.committed = True


class FakeEmbeddingResponse:
    def __init__(self, embedding):
        self.embedding = embedding

    def raise_for_status(self):
        return None

    def json(self):
        return {"data": [{"embedding": self.embedding}]}


class FakeEmbeddingClient:
    def __init__(self, response):
        self.response = response
        self.request = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    async def post(self, url, *, headers, json):
        self.request = {"url": url, "headers": headers, "json": json}
        return self.response


class MemoryEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def test_related_uses_current_thought_with_passage_context(self):
        query = memory.RelatedThoughtQuery(
            clientId=uuid4(),
            sourceKey="sample:5",
            paragraphText="A place for conversation between reading and memory.",
            note="I feel no pressure should be denied anywhere.",
        )
        embedding = AsyncMock(return_value=[0.1] * memory.EMBEDDING_DIMENSIONS)
        with (
            patch.object(memory, "ensure_schema"),
            patch.object(memory, "Session", return_value=FakeSession()),
            patch.object(memory, "embed_text", embedding),
        ):
            await memory.find_related_thought(query, "Bearer test-key")
        embedding.assert_awaited_once_with(
            f"Thought: {query.note}\n\nPassage: {query.paragraph_text}", "test-key", "passage"
        )

    async def test_saved_note_search_reuses_vector_without_provider_request(self):
        vector = [0.2] * memory.EMBEDDING_DIMENSIONS
        query = memory.RelatedThoughtQuery(
            clientId=uuid4(), sourceKey="sample:0", paragraphText="A passage", note="My idea"
        )
        embedding = AsyncMock()
        with (
            patch.object(memory, "ensure_schema"),
            patch.object(
                memory,
                "Session",
                side_effect=[FakeSession([SimpleNamespace(embedding=vector)]), FakeSession()],
            ),
            patch.object(memory, "embed_text", embedding),
        ):
            result = await memory.find_related_thought(query)
        self.assertIsNone(result["thought"])
        embedding.assert_not_awaited()

    async def test_embedding_uses_expected_model_and_input_mode(self):
        vector = [0.1] * memory.EMBEDDING_DIMENSIONS
        client = FakeEmbeddingClient(FakeEmbeddingResponse(vector))

        with patch.object(memory.httpx, "AsyncClient", return_value=client):
            result = await memory.embed_text("a passage", "test-key", "passage")

        self.assertEqual(result, vector)
        request = cast(dict[str, object], client.request)
        headers = cast(dict[str, str], request["headers"])
        body = cast(dict[str, object], request["json"])
        self.assertEqual(request["url"], memory.EMBEDDING_URL)
        self.assertEqual(headers["Authorization"], "Bearer test-key")
        self.assertEqual(body["model"], memory.EMBEDDING_MODEL)
        self.assertEqual(body["input_type"], "passage")

    async def test_embedding_rejects_wrong_vector_dimensions(self):
        client = FakeEmbeddingClient(FakeEmbeddingResponse([0.1, 0.2]))

        with (
            patch.object(memory.httpx, "AsyncClient", return_value=client),
            self.assertRaises(HTTPException) as raised,
        ):
            await memory.embed_text("query", "test-key", "query")

        self.assertEqual(raised.exception.status_code, 502)

    async def test_save_stores_note_and_passage_embedding(self):
        client_id = uuid4()
        created_at = datetime.now(UTC)
        thought = memory.ThoughtInput(
            clientId=client_id,
            sourceKey="sample:0",
            note="Attention is a kind of care.",
            paragraphText="The passage is about paying attention.",
            paragraphIndex=0,
            articleTitle="A sample essay",
            articleUrl="http://localhost:5173/",
            createdAt=created_at,
        )
        record_id = uuid4()
        lookup_session = FakeSession([None])
        save_session = FakeSession([record_id])
        embedding = AsyncMock(return_value=[0.25] * memory.EMBEDDING_DIMENSIONS)

        with (
            patch.object(memory, "ensure_schema"),
            patch.object(memory, "Session", side_effect=[lookup_session, save_session]),
            patch.object(memory, "embed_text", embedding),
        ):
            result = await memory.save_thought("sample:0", thought, "Bearer test-key")

        self.assertEqual(result, {"saved": True, "id": str(record_id)})
        embedding.assert_awaited_once_with(
            "Thought: Attention is a kind of care.\n\n"
            "Passage: The passage is about paying attention.",
            "test-key",
            "passage",
        )
        self.assertTrue(save_session.committed)

    async def test_related_returns_only_a_strong_match_and_uses_query_mode(self):
        client_id = uuid4()
        record = SimpleNamespace(
            id=uuid4(),
            note="Attention is a kind of care.",
            source_key="sample:0",
            paragraph_index=0,
            article_title="A sample essay",
            article_url="http://localhost:5173/",
            created_at=datetime.now(UTC),
        )
        session = FakeSession(rows=[(record, 0.12)])
        embedding = AsyncMock(return_value=[0.1] * memory.EMBEDDING_DIMENSIONS)
        query = memory.RelatedThoughtQuery(
            clientId=client_id,
            sourceKey="sample:1",
            paragraphText="Reading can make a line meet a memory.",
        )

        with (
            patch.object(memory, "ensure_schema"),
            patch.object(memory, "Session", return_value=session),
            patch.object(memory, "embed_text", embedding),
        ):
            result = await memory.find_related_thought(query, "Bearer test-key")

        match = cast(dict[str, object], result["thought"])
        self.assertEqual(match["id"], str(record.id))
        self.assertAlmostEqual(cast(float, match["score"]), 0.88)
        embedding.assert_awaited_once_with(query.paragraph_text, "test-key", "query")

        sql = str(session.executed[0].compile(dialect=postgresql.dialect()))
        self.assertIn("thought_dismissals", sql)
        self.assertIn("NOT (EXISTS", sql)

    async def test_related_hides_matches_below_threshold(self):
        record = SimpleNamespace(
            id=uuid4(),
            note="A distant note",
            source_key="sample:0",
            paragraph_index=0,
            article_title="A sample essay",
            article_url="http://localhost:5173/",
            created_at=datetime.now(UTC),
        )
        session = FakeSession(rows=[(record, 0.50)])
        query = memory.RelatedThoughtQuery(
            clientId=uuid4(),
            sourceKey="sample:1",
            paragraphText="A query passage.",
        )

        with (
            patch.object(memory, "ensure_schema"),
            patch.object(memory, "Session", return_value=session),
            patch.object(
                memory,
                "embed_text",
                AsyncMock(return_value=[0.1] * memory.EMBEDDING_DIMENSIONS),
            ),
        ):
            result = await memory.find_related_thought(query, "Bearer test-key")

        self.assertIsNone(result["thought"])

    def test_dismiss_persists_a_not_related_pair(self):
        session = FakeSession()
        dismissal = memory.DismissThoughtInput(
            clientId=uuid4(),
            sourceKey="sample:1",
            thoughtId=uuid4(),
        )

        with (
            patch.object(memory, "ensure_schema"),
            patch.object(memory, "Session", return_value=session),
        ):
            result = memory.dismiss_related_thought(dismissal)

        self.assertEqual(result, {"dismissed": True})
        self.assertTrue(session.committed)
        statement = session.executed[0]
        self.assertEqual(statement.compile().params["thought_id"], dismissal.thought_id)

    def test_api_key_is_required_for_embedding(self):
        with self.assertRaises(HTTPException) as raised:
            memory.require_api_key(None)

        self.assertEqual(raised.exception.status_code, 401)


@unittest.skipUnless(
    os.getenv("RUN_POSTGRES_INTEGRATION") == "1",
    "set RUN_POSTGRES_INTEGRATION=1 to run against local Postgres",
)
class PostgresMemoryIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_save_find_dismiss_and_remove_a_memory(self):
        client_id = uuid4()
        source_key = f"integration:{client_id}"
        target_source_key = f"{source_key}:next"
        vector = [0.0] * memory.EMBEDDING_DIMENSIONS
        vector[0] = 1.0

        memory.ensure_schema()
        thought = memory.ThoughtInput(
            clientId=client_id,
            sourceKey=source_key,
            note="Synthetic integration note about attention.",
            paragraphText="Synthetic passage used only for this integration check.",
            paragraphIndex=0,
            articleTitle="Synthetic integration article",
            articleUrl="http://localhost/integration-test",
        )
        query = memory.RelatedThoughtQuery(
            clientId=client_id,
            sourceKey=target_source_key,
            paragraphText="A related synthetic passage about attention.",
        )
        embed = AsyncMock(return_value=vector)

        try:
            with patch.object(memory, "embed_text", embed):
                saved = await memory.save_thought(source_key, thought, "Bearer synthetic-test-key")
                self.assertTrue(saved["saved"])
                saved_id = cast(str, saved["id"])

                match = await memory.find_related_thought(query, "Bearer synthetic-test-key")
                match_record = cast(dict[str, object], match["thought"])
                self.assertEqual(match_record["id"], saved_id)

                dismissal = memory.DismissThoughtInput(
                    clientId=client_id,
                    sourceKey=target_source_key,
                    thoughtId=UUID(saved_id),
                )
                memory.dismiss_related_thought(dismissal)
                dismissed = await memory.find_related_thought(query, "Bearer synthetic-test-key")
                self.assertIsNone(dismissed["thought"])
        finally:
            with memory.Session(memory.engine) as session:
                session.execute(
                    memory.delete(memory.ThoughtRecord).where(
                        memory.ThoughtRecord.client_id == client_id
                    )
                )
                session.commit()


if __name__ == "__main__":
    unittest.main()


class ThoughtRouteTests(unittest.TestCase):
    def test_save_route_accepts_source_keys_containing_urls(self):
        from starlette.routing import Match

        key = "https://en.wikipedia.org/wiki/Attention:1"
        scope = {
            "type": "http",
            "method": "PUT",
            "path": f"/thoughts/{key}",
            "root_path": "",
        }
        route = next(r for r in memory.router.routes if "PUT" in getattr(r, "methods", ()))
        match, child = route.matches(scope)
        self.assertEqual(match, Match.FULL)
        self.assertEqual(child["path_params"]["source_key"], key)
