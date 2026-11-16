"""Live benchmark: NVIDIA_API_KEY=... uv run --directory api python benchmarks/companion.py.
Creates isolated test records; removes them when complete. Does not alter browser storage.
"""

import asyncio
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx

API = os.getenv("MARGIN_API_URL", "http://127.0.0.1:8000")
URLS = [
    "https://aeon.co/essays/weve-always-been-distracted-or-at-least-worried-that-we-are",
    "https://aeon.co/essays/what-is-the-self-if-not-that-which-pays-attention",
]


async def main():
    owner = str(uuid4())
    headers = {"Authorization": f"Bearer {os.environ['NVIDIA_API_KEY']}"}
    report: dict[str, Any] = {"clientId": owner, "articles": [], "runs": [], "concurrency": 2}
    async with httpx.AsyncClient(timeout=120) as client:
        try:
            for url in URLS:
                tick = time.perf_counter()
                response = await client.post(f"{API}/articles/extract", json={"url": url})
                response.raise_for_status()
                article = response.json()
                extracted = round((time.perf_counter() - tick) * 1000)
                tick = time.perf_counter()
                response = await client.post(
                    f"{API}/companion/index",
                    headers=headers,
                    json={
                        "clientId": owner,
                        "url": url,
                        "title": article["title"],
                        "paragraphs": article["paragraphs"],
                    },
                )
                response.raise_for_status()
                report["articles"].append(
                    {
                        **article,
                        "extractMs": extracted,
                        "indexMs": round((time.perf_counter() - tick) * 1000),
                        "index": response.json(),
                    }
                )
                print("indexed", article["title"], len(article["paragraphs"]), flush=True)
            article = report["articles"][0]
            semaphore = asyncio.Semaphore(2)

            async def run(number, scope):
                async with semaphore:
                    started = time.perf_counter()
                    answer = ""
                    sources = []
                    metrics = {}
                    error = None
                    first = None
                    question = (
                        "Explain the argument about distraction and what it does not prove."
                        if scope == "passage"
                        else "Compare these authors on attention. Preserve nuances."
                    )
                    async with client.stream(
                        "POST",
                        f"{API}/companion/ask",
                        headers=headers,
                        json={
                            "clientId": owner,
                            "question": question,
                            "articleUrl": article["url"],
                            "title": article["title"],
                            "paragraphIndex": 1,
                            "contextStart": 0,
                            "context": article["paragraphs"][:4],
                            "scope": scope,
                        },
                    ) as response:
                        response.raise_for_status()
                        async for line in response.aiter_lines():
                            if not line:
                                continue
                            event = json.loads(line)
                            if event["type"] == "delta":
                                if first is None:
                                    first = round((time.perf_counter() - started) * 1000, 1)
                                answer += event["text"]
                            elif event["type"] == "sources":
                                sources = event["sources"]
                            elif event["type"] == "done":
                                metrics = event["metrics"]
                            elif event["type"] == "error":
                                error = event["detail"]
                    cited = [int(i) for i in re.findall(r"\[(\d+)\]", answer)]
                    source_ids = {s["id"] for s in sources}
                    result = {
                        "run": number,
                        "scope": scope,
                        "firstAnswerMs": first,
                        "totalClientMs": round((time.perf_counter() - started) * 1000, 1),
                        "metrics": metrics,
                        "error": error,
                        "answer": answer,
                        "citationIdsValid": bool(cited) and all(i in source_ids for i in cited),
                        "retrievedArticleCount": len(
                            {s["url"] for s in sources if s["kind"] == "article"}
                        ),
                    }
                    report["runs"].append(result)
                    print("run", number, scope, first, "ms", "error", error, flush=True)

            # First run measures uncached query embedding; subsequent repeats expose caching.
            await run(0, "library")
            await asyncio.gather(*(run(i, "passage" if i % 2 else "library") for i in range(1, 13)))

            def percentile(values, fraction):
                values = sorted(values)
                return values[max(0, math.ceil(len(values) * fraction) - 1)] if values else None

            report["summary"] = {}
            for scope in ("passage", "library"):
                rows = [
                    r
                    for r in report["runs"]
                    if r["scope"] == scope and r["firstAnswerMs"] is not None
                ]
                values = [r["firstAnswerMs"] for r in rows]
                report["summary"][scope] = {
                    "count": len(rows),
                    "p50FirstAnswerMs": percentile(values, 0.5),
                    "p95FirstAnswerMs": percentile(values, 0.95),
                    "validCitationRuns": sum(r["citationIdsValid"] for r in rows),
                }
            for article in report["articles"]:
                article.pop("paragraphs", None)
                article.pop("blocks", None)
            output = Path(
                os.getenv("MARGIN_BENCHMARK_OUTPUT", "/tmp/margin-companion-benchmark.json")
            )
            output.write_text(json.dumps(report, indent=2))
            print(json.dumps(report["summary"], indent=2))
        finally:
            # This benchmark owns only this UUID's generated passage records.
            import sys

            sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
            from sqlalchemy import delete
            from sqlalchemy.orm import Session

            from companion import Passage
            from memory import engine

            with Session(engine) as session:
                session.execute(delete(Passage).where(Passage.client_id == owner))
                session.commit()


if __name__ == "__main__":
    asyncio.run(main())
