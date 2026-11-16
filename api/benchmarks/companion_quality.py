"""Answer-quality eval for Ask Margin.

Run with the API up:
    NVIDIA_API_KEY=... uv run --directory api python benchmarks/companion_quality.py

Each case sends a real question through /companion/ask and scores the answer with
cheap, deterministic checks. Use it before and after changing the prompt or model:
the pass rate is the number to watch. Passage-scope only, so nothing is written to
the database.
"""

import asyncio
import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

import httpx

API = os.getenv("MARGIN_API_URL", "http://127.0.0.1:8000")
TITLE = "The Economy of Attention"
URL = "margin:eval-attention"
ARTICLE = [
    "The Economy of Attention",
    "We like to say that attention is a spotlight, as if we simply point it where we wish. "
    "But a spotlight costs nothing to move. Attention does: every switch leaves a residue, "
    "a part of the mind still turned toward the last thing.",
    "History",
    "William James wrote in 1890 that everyone knows what attention is. A century later, "
    "researchers had split it into alerting, orienting and executive control, and found "
    "that each could fail on its own.",
    "The modern worry is not that we have less attention than our grandparents, which is "
    "hard to show, but that more of our environment is engineered to capture it.",
    "Perhaps the better metaphor is a budget. A budget can be spent well or badly, but it "
    "cannot be spent twice, and pretending otherwise is how debts begin.",
]

LEAK = re.compile(r"\b(paragraph ?index|index \d|json|paragraphIndex|selected paragraph:)\b", re.I)
VIEW = re.compile(
    r"\b(i think|my read|i find|i'd argue|in my view|i'm (not )?convinced|i suspect)\b", re.I
)
REFUSAL = re.compile(r"(do not|don't) have (personal )?opinions|as an ai", re.I)


@dataclass
class Case:
    name: str
    question: str
    paragraph: int
    wants_view: bool = False
    wants_citation: bool = True
    quote: str = ""
    history: list[dict[str, str]] = field(default_factory=list)
    must_mention: list[str] = field(default_factory=list)


CASES = [
    Case("factual", "What does the author mean by residue?", 1, must_mention=["switch"]),
    Case("opinion", "What is your opinion on the budget metaphor?", 5, wants_view=True),
    Case("heading-selection", "What is this section about?", 2, must_mention=["James"]),
    Case("steelman", "What is the strongest case against the author here?", 4, wants_view=True),
    Case(
        "highlight",
        "Why does this matter?",
        5,
        quote="it cannot be spent twice",
    ),
    Case(
        "memory",
        "Remind me, what was the first thing I asked you?",
        1,
        wants_citation=False,
        history=[
            {"role": "user", "content": "Is attention really like a spotlight?"},
            {
                "role": "assistant",
                "content": "The author argues it is not, because moving it has a cost.",
            },
        ],
        must_mention=["spotlight"],
    ),
]


async def run_case(client: httpx.AsyncClient, key: str, case: Case) -> dict[str, Any]:
    start = max(0, case.paragraph - 2)
    payload = {
        "clientId": str(uuid4()),
        "question": case.question,
        "articleUrl": URL,
        "title": TITLE,
        "paragraphIndex": case.paragraph,
        "contextStart": start,
        "context": ARTICLE[start : case.paragraph + 3],
        "scope": "passage",
        "history": case.history,
        "quote": case.quote,
        "outline": ["The Economy of Attention", "History"],
    }
    answer, sources, follow_ups, error = "", [], [], None
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
            elif event["type"] == "followups":
                follow_ups = event["questions"]
            elif event["type"] == "error":
                error = event["detail"]

    cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
    words = len(answer.split())
    checks = {
        "answered": not error and words > 0,
        "no_internal_leak": not LEAK.search(answer),
        "citations_valid": all(1 <= n <= len(sources) for n in cited),
        "cites_when_needed": bool(cited) or not case.wants_citation,
        "length_40_260": 40 <= words <= 260 or (not case.wants_citation and words > 0),
        "follow_ups": len(follow_ups) >= 1,
    }
    if case.wants_view:
        checks["gives_a_view"] = bool(VIEW.search(answer)) and not REFUSAL.search(answer)
    for term in case.must_mention:
        checks[f"mentions_{term.lower()}"] = term.lower() in answer.lower()
    return {
        "case": case.name,
        "checks": checks,
        "words": words,
        "answer": answer,
        "followUps": follow_ups,
        "error": error,
    }


async def main() -> None:
    key = os.environ.get("NVIDIA_API_KEY")
    if not key:
        sys.exit("Set NVIDIA_API_KEY to run the eval.")
    async with httpx.AsyncClient(timeout=120) as client:
        results = [await run_case(client, key, case) for case in CASES]

    total = passed = 0
    for result in results:
        failed = [name for name, ok in result["checks"].items() if not ok]
        total += len(result["checks"])
        passed += len(result["checks"]) - len(failed)
        status = "PASS" if not failed else "FAIL " + ", ".join(failed)
        print(f"{result['case']:<18} {result['words']:>4}w  {status}")
        if failed or "--verbose" in sys.argv:
            print("   ", result["answer"].replace("\n", " ")[:400])
            if result["followUps"]:
                print("    follow-ups:", " | ".join(result["followUps"]))
    print(f"\n{passed}/{total} checks passed ({passed / total:.0%})")
    if "--json" in sys.argv:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
