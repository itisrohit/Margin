"""Cross-library grounding eval: does Margin find the right articles, and do its
citations actually support what it says?

Run with the API and PostgreSQL up:
    NVIDIA_API_KEY=... uv run --directory api python benchmarks/grounding_eval.py [--verbose]

It indexes a small synthetic library under a fresh clientId (deleted afterwards),
asks library-scope questions, and reports:
  - retrieval recall: did the sources include every article the question needs?
  - citation validity: does every [n] point at a real source?
  - citation support: an LLM judge reads each cited sentence next to its source and
    labels it supported / partial / unsupported.
  - honesty on gaps: for a topic the library doesn't cover, does it say so?

The judge uses MARGIN_JUDGE_MODEL (defaults to the chat model). A judge is not ground
truth; read the unsupported examples it prints before trusting a number.
"""

import asyncio
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

API = os.getenv("MARGIN_API_URL", "http://127.0.0.1:8000")
NVIDIA = os.getenv("MARGIN_NVIDIA_URL", "https://integrate.api.nvidia.com/v1")
JUDGE_MODEL = os.getenv(
    "MARGIN_JUDGE_MODEL", os.getenv("MARGIN_CHAT_MODEL", "nvidia/nemotron-3.5-bolt-e4b")
)

LIBRARY = {
    "eval:pressure": (
        "Pressure and Performance",
        [
            "Studies of skilled performers suggest that moderate pressure sharpens well-practiced "
            "tasks, because attention narrows onto routines that are already automatic.",
            "The same narrowing harms creative work. In experiments where people solved insight "
            "problems under time pressure, they found fewer novel solutions than relaxed groups.",
        ],
    ),
    "eval:deadlines": (
        "The Case for Deadlines",
        [
            "Teams with clear, near deadlines shipped more creative prototypes than teams with "
            "open-ended timelines, according to a two-year study of design studios.",
            "The effect held only when the goal was well defined. With vague goals, deadlines "
            "produced rushed, conventional work.",
        ],
    ),
    "eval:residue": (
        "Focus at Work",
        [
            "Sophie Leroy coined the term attention residue: after switching tasks, part of our "
            "attention stays with the previous task, lowering performance on the next one.",
        ],
    ),
    "eval:sleep": (
        "Why We Sleep Less",
        [
            "Sleep consolidates memories formed during the day; deprived participants recalled "
            "fewer word pairs the next morning.",
        ],
    ),
}


@dataclass
class Case:
    name: str
    question: str
    needs: set[str] = field(default_factory=set)
    expects_gap: bool = False


CASES = [
    Case(
        "cross-article conflict",
        "Do my articles agree on whether time pressure helps creative work?",
        needs={"eval:pressure", "eval:deadlines"},
    ),
    Case("exact name", "What did Sophie Leroy find?", needs={"eval:residue"}),
    Case("topic", "What have I read about memory and sleep?", needs={"eval:sleep"}),
    Case(
        "not in library",
        "What have I read about quantum computing?",
        expects_gap=True,
    ),
]

GAP = re.compile(
    r"(doesn't|does not|don't|do not|no)\b.{0,40}\b"
    r"(cover|mention|discuss|include|find|found|anything|articles?|passages?)|"
    r"nothing (in|about)|not (covered|mentioned)|isn't covered",
    re.I,
)


def cited_claims(answer: str) -> list[tuple[str, list[int]]]:
    sentences = re.split(r"(?<=[.!?])\s+", answer)
    claims = []
    for sentence in sentences:
        ids = [int(n) for n in re.findall(r"\[(\d+)\]", sentence)]
        if ids:
            claims.append((re.sub(r"\s?\[\d+\]", "", sentence).strip(), ids))
    return claims


async def judge(client: httpx.AsyncClient, key: str, claim: str, source: str) -> str:
    response = await client.post(
        f"{NVIDIA}/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": JUDGE_MODEL,
            "temperature": 0,
            "max_tokens": 60,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You check whether a source supports a claim. Answer with exactly one "
                        "word first: SUPPORTED (the source states or directly implies it), "
                        "PARTIAL (the source supports part of it, or the claim drops an "
                        "important condition), or UNSUPPORTED. Then a short reason."
                    ),
                },
                {"role": "user", "content": f"Source:\n{source}\n\nClaim:\n{claim}"},
            ],
        },
    )
    response.raise_for_status()
    verdict = response.json()["choices"][0]["message"]["content"].strip().upper()
    for label in ("UNSUPPORTED", "PARTIAL", "SUPPORTED"):
        if verdict.startswith(label) or label in verdict.split()[:3]:
            return label
    return "UNSUPPORTED"


async def ask(client: httpx.AsyncClient, key: str, owner: str, question: str) -> dict[str, Any]:
    payload = {
        "clientId": owner,
        "question": question,
        "articleUrl": "eval:current",
        "title": "Scratch",
        "paragraphIndex": 0,
        "contextStart": 0,
        "context": ["(The reader is asking about their whole library.)"],
        "scope": "library",
    }
    answer, sources, metrics, error = "", [], {}, None
    async with client.stream(
        "POST", f"{API}/companion/ask", json=payload, headers={"Authorization": f"Bearer {key}"}
    ) as response:
        async for line in response.aiter_lines():
            if not line.strip():
                continue
            event = json.loads(line)
            if event["type"] == "delta":
                answer += event["text"]
            elif event["type"] == "sources":
                sources = event["sources"]
            elif event["type"] == "done":
                metrics = event["metrics"]
            elif event["type"] == "error":
                error = event["detail"]
    return {"answer": answer, "sources": sources, "metrics": metrics, "error": error}


async def main() -> None:
    key = os.environ.get("NVIDIA_API_KEY")
    if not key:
        sys.exit("Set NVIDIA_API_KEY to run the eval.")
    verbose = "--verbose" in sys.argv
    owner = str(uuid4())
    headers = {"Authorization": f"Bearer {key}"}
    totals = {"recall": 0, "recall_n": 0, "valid": 0, "claims": 0}
    verdicts = {"SUPPORTED": 0, "PARTIAL": 0, "UNSUPPORTED": 0}
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            for url, (title, paragraphs) in LIBRARY.items():
                response = await client.post(
                    f"{API}/companion/index",
                    headers=headers,
                    json={"clientId": owner, "url": url, "title": title, "paragraphs": paragraphs},
                )
                response.raise_for_status()
            print(f"indexed {len(LIBRARY)} articles\n")

            for case in CASES:
                result = await ask(client, key, owner, case.question)
                answer, sources = result["answer"], result["sources"]
                by_id = {s["id"]: s for s in sources}
                urls = {s["url"] for s in sources if s["kind"] == "article"}
                missing = case.needs - urls
                if case.needs:
                    totals["recall_n"] += 1
                    totals["recall"] += not missing

                claims = cited_claims(answer)
                valid = all(i in by_id for _, ids in claims for i in ids)
                totals["valid"] += valid
                case_verdicts = []
                for claim, ids in claims:
                    evidence = "\n\n".join(by_id[i]["text"] for i in ids if i in by_id)
                    verdict = (
                        await judge(client, key, claim, evidence) if evidence else "UNSUPPORTED"
                    )
                    verdicts[verdict] += 1
                    totals["claims"] += 1
                    case_verdicts.append((verdict, claim))

                flags = []
                if result["error"]:
                    flags.append(f"error: {result['error']}")
                if missing:
                    flags.append(f"missed {sorted(missing)}")
                if not valid:
                    flags.append("invalid citation id")
                if case.expects_gap and not GAP.search(answer):
                    flags.append("did not admit the gap")
                if not case.expects_gap and not claims:
                    flags.append("no citations")
                weak = [c for v, c in case_verdicts if v != "SUPPORTED"]
                status = "PASS" if not flags and not weak else "CHECK"
                print(
                    f"{case.name:<24} {status}  sources from {len(urls)} articles, "
                    f"{len(claims)} cited claims, "
                    f"{sum(v == 'SUPPORTED' for v, _ in case_verdicts)} supported"
                )
                for flag in flags:
                    print(f"    ! {flag}")
                for verdict, claim in case_verdicts:
                    if verdict != "SUPPORTED" or verbose:
                        print(f"    {verdict:<11} {claim[:160]}")
                if verbose:
                    print("    answer:", answer.replace("\n", " ")[:500])
                print()
    finally:
        from sqlalchemy import delete
        from sqlalchemy.orm import Session

        from companion import Passage
        from memory import engine

        with Session(engine) as session:
            session.execute(delete(Passage).where(Passage.client_id == owner))
            session.commit()

    claims = totals["claims"] or 1
    print(
        f"retrieval recall   {totals['recall']}/{totals['recall_n']} "
        "questions found every needed article"
    )
    print(f"citation ids valid {totals['valid']}/{len(CASES)} answers")
    print(
        f"citation support   {verdicts['SUPPORTED']}/{totals['claims']} supported "
        f"({verdicts['SUPPORTED'] / claims:.0%}), {verdicts['PARTIAL']} partial, "
        f"{verdicts['UNSUPPORTED']} unsupported"
    )


if __name__ == "__main__":
    asyncio.run(main())
