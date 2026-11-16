"""Tune the agent's semantic router on held-out questions.

    NVIDIA_API_KEY=... uv run --directory api python benchmarks/router_eval.py

Needs only the key (no API server or database). It embeds the router's examples and the
labelled questions below, which are worded differently from the examples, then sweeps
the similarity threshold and margin. For each setting it reports:
  - precision: of the questions the router acted on, how many it routed correctly.
    A wrong forced tool is worse than none, so this is the number to protect.
  - coverage: how many questions it acted on (the rest go to the model, as before).
It recommends the setting with the most coverage at precision >= TARGET, and prints the
mistakes at that setting. Put the result in .env as MARGIN_ROUTER_THRESHOLD and
MARGIN_ROUTER_MARGIN. With 54 questions, one mistake is ~2 points of precision.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import companion as core  # noqa: E402
import router  # noqa: E402

TARGET = 0.97
OUTPUT = Path(__file__).resolve().parent / "results" / "router_eval.json"

LABELLED: dict[str, list[str]] = {
    "propose_note": [
        "Save a note here saying switching costs are underrated.",
        "Add a note that this paragraph is the key one.",
        "Note this down: attention is a budget.",
        "Could you save a note that pressure cuts both ways?",
        "Leave a note on this passage that I disagree.",
        "Please write a margin note summarising this.",
        "Make this a note: check the original study.",
        "Remember that this contradicts what Leroy found.",
        "Draft a note for this paragraph about deadlines.",
    ],
    "search_notes": [
        "What did I think about deadlines last week?",
        "Have I written any notes about sleep?",
        "What were my thoughts on attention residue?",
        "Did I make any notes on this author before?",
        "What have I jotted down about creativity?",
        "Find what I wrote about multitasking.",
        "What did I note yesterday?",
        "Do my notes say anything about email?",
        "Remind me what I thought of the pressure article.",
    ],
    "search_library": [
        "Do other articles I've read say anything about time pressure?",
        "What did Sophie Leroy find?",
        "How does this connect to what I've read about memory?",
        "What have I read about quantum computing?",
        "Does anything else in my library disagree with this?",
        "Which of my saved pieces talk about deadlines?",
        "Have I saved anything by Kahneman?",
        "Is this idea in any other essay I've read?",
        "Compare this with the other articles on focus.",
    ],
    "search_article": [
        "Where else in this article does the author talk about email?",
        "Does the author come back to batching later in the article?",
        "Is there a part of this piece about remedies?",
        "Does this essay ever define deep work?",
        "Find where the author gives numbers in this article.",
        "What does the conclusion of this article say about habits?",
        "Is this point repeated anywhere else in the text?",
        "Where does the writer first mention attention?",
        "Search this piece for anything on meetings.",
    ],
    "get_passage": [
        "Read the passages around this one and tell me how the argument builds.",
        "What does the paragraph before this say?",
        "Show me what comes next.",
        "I need the surrounding context for this paragraph.",
        "How does this connect to the paragraph just above?",
        "Read on a little and tell me where this goes.",
        "What's the setup just before this point?",
        "Give me the neighbouring paragraphs.",
        "Continue from here: what is the next step of the argument?",
    ],
    router.ANSWER: [
        "What does this passage mean?",
        "Explain this in plain words.",
        "What is the strongest objection to this paragraph?",
        "Is the author right here?",
        "What's the takeaway?",
        "Put this into one sentence.",
        "Why might someone disagree with this?",
        "What does 'executive control' mean here?",
        "Is this a strong argument?",
    ],
}


async def main() -> None:
    key = os.environ.get("NVIDIA_API_KEY")
    if not key:
        sys.exit("Set NVIDIA_API_KEY to run the router eval.")
    exemplars = await router.exemplar_vectors(key)
    items = [(label, q) for label, questions in LABELLED.items() for q in questions]
    vectors: list[list[float]] = []
    for start in range(0, len(items), 16):
        vectors += await core.embeddings([q for _, q in items[start : start + 16]], key, "query")
    await core.close_client()

    rows: list[dict[str, Any]] = []
    for threshold in [x / 100 for x in range(30, 91, 5)]:
        for margin in (0.0, 0.02, 0.04, 0.06, 0.08):
            acted = correct = 0
            mistakes = []
            for (label, question), vector in zip(items, vectors, strict=True):
                result = router.classify(vector, exemplars, threshold, margin)
                if result.tool is None:
                    continue
                acted += 1
                if result.tool == label:
                    correct += 1
                else:
                    mistakes.append({"question": question, "expected": label, "got": result.tool})
            rows.append(
                {
                    "threshold": threshold,
                    "margin": margin,
                    "coverage": acted / len(items),
                    "precision": correct / acted if acted else None,
                    "mistakes": mistakes,
                }
            )

    eligible = [r for r in rows if r["precision"] is not None and r["precision"] >= TARGET]
    best = max(eligible, key=lambda r: (r["coverage"], -r["threshold"])) if eligible else None
    # Raw scores at the default setting, to see how close the calls are.
    detail = []
    for (label, question), vector in zip(items, vectors, strict=True):
        result = router.classify(vector, exemplars)
        detail.append(
            {
                "question": question,
                "expected": label,
                "best": max(result.scores, key=result.scores.__getitem__),
                "score": round(result.score, 3),
                "runnerUp": round(result.runner_up, 3),
                "routed": result.tool,
            }
        )

    print(f"{len(items)} labelled questions, {sum(map(len, router.ROUTES.values()))} examples\n")
    print("threshold margin  coverage precision")
    for r in rows:
        if r["margin"] in (0.0, 0.04) and r["precision"] is not None:
            print(
                f"  {r['threshold']:.2f}    {r['margin']:.2f}    {r['coverage']:>5.0%}"
                f"    {r['precision']:>5.0%}"
            )
    nearest = sum(d["best"] == d["expected"] for d in detail)
    print(f"\nnearest route correct (no threshold): {nearest}/{len(items)}")
    if best:
        print(
            f"\nrecommended: MARGIN_ROUTER_THRESHOLD={best['threshold']:.2f} "
            f"MARGIN_ROUTER_MARGIN={best['margin']:.2f}  -> routes {best['coverage']:.0%} "
            f"of questions at {best['precision']:.0%} precision"
        )
        for m in best["mistakes"]:
            print(f"   wrong: {m['question']!r} -> {m['got']} (expected {m['expected']})")
    else:
        print(f"\nno setting reached {TARGET:.0%} precision; add or reword route examples")
    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text(json.dumps({"best": best, "sweep": rows, "questions": detail}, indent=2))
    print(f"\nsaved {OUTPUT}")


if __name__ == "__main__":
    asyncio.run(main())
