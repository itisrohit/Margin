import os
import unittest
from uuid import uuid4

import httpx
from fastapi import FastAPI

import chats


class ChatRouteTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        app = FastAPI()
        app.include_router(chats.router)
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://margin.test"
        )

    async def asyncTearDown(self):
        await self.client.aclose()

    async def test_invalid_chat_id_rejected(self):
        response = await self.client.put("/companion/chats/not a valid id!", json={})
        self.assertEqual(response.status_code, 422)

    async def test_empty_chat_rejected(self):
        response = await self.client.put(
            "/companion/chats/abc",
            json={
                "clientId": str(uuid4()),
                "articleUrl": "https://example.com",
                "title": "Empty",
                "turns": [],
                "updatedAt": "2026-11-16T10:00:00Z",
            },
        )
        self.assertEqual(response.status_code, 422)


@unittest.skipUnless(
    os.getenv("RUN_POSTGRES_INTEGRATION") == "1",
    "set RUN_POSTGRES_INTEGRATION=1 to run against local Postgres",
)
class ChatPersistenceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        app = FastAPI()
        app.include_router(chats.router)
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://margin.test"
        )
        self.client_id = str(uuid4())
        self.url = "https://en.wikipedia.org/wiki/Attention"

    async def asyncTearDown(self):
        await self.client.aclose()

    def chat(self, updated_at: str, status: str = "done") -> dict:
        return {
            "clientId": self.client_id,
            "articleUrl": self.url,
            "title": "What is attention?",
            "turns": [
                {
                    "id": 1.5,
                    "question": "What is attention?",
                    "answer": "Selective focus [1].",
                    "sources": [
                        {
                            "id": 1,
                            "kind": "passage",
                            "title": "Attention",
                            "url": self.url,
                            "paragraphIndex": 2,
                            "text": "Attention is...",
                        }
                    ],
                    "status": status,
                }
            ],
            "updatedAt": updated_at,
        }

    async def list(self, client_id: str | None = None) -> list[dict]:
        response = await self.client.get(
            "/companion/chats",
            params={"clientId": client_id or self.client_id, "articleUrl": self.url},
        )
        self.assertEqual(response.status_code, 200)
        return response.json()["chats"]

    async def test_save_update_list_and_delete(self):
        saved = await self.client.put(
            "/companion/chats/chat-1", json=self.chat("2026-11-16T10:00:00Z", "streaming")
        )
        self.assertEqual(saved.status_code, 200)
        [stored] = await self.list()
        self.assertEqual(stored["turns"][0]["status"], "stopped")

        await self.client.put("/companion/chats/chat-1", json=self.chat("2026-11-16T11:00:00Z"))
        [updated] = await self.list()
        self.assertEqual(updated["turns"][0]["status"], "done")
        self.assertEqual(updated["turns"][0]["sources"][0]["paragraphIndex"], 2)

        self.assertEqual(await self.list(str(uuid4())), [])

        deleted = await self.client.delete(
            "/companion/chats/chat-1", params={"clientId": self.client_id}
        )
        self.assertEqual(deleted.status_code, 200)
        self.assertEqual(await self.list(), [])

    async def test_keeps_only_most_recent_chats(self):
        for n in range(chats.MAX_CHATS_PER_ARTICLE + 2):
            await self.client.put(
                f"/companion/chats/chat-{n}",
                json=self.chat(f"2026-11-16T10:{n:02d}:00Z"),
            )
        listed = await self.list()
        self.assertEqual(len(listed), chats.MAX_CHATS_PER_ARTICLE)
        self.assertEqual(listed[0]["id"], f"chat-{chats.MAX_CHATS_PER_ARTICLE + 1}")
        self.assertNotIn("chat-0", [c["id"] for c in listed])
