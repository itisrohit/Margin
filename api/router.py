"""Semantic intent router for agent mode: which lookup does this question need?

A small model choosing tools on its own is right most of the time, not every time. The
usual fix is a cascade: a cheap, deterministic first stage for clear intents, and the
model for everything else. This is that first stage.

Each route has example questions. They are embedded once per API key (one batched call
to the embedding model Margin already uses), and each incoming question is compared with
them by cosine similarity. The router only acts when the best route clears a threshold
*and* beats the runner-up by a margin; otherwise it abstains and the model decides as
before. Thresholds are tuned with benchmarks/router_eval.py, not guessed: set
MARGIN_ROUTER_THRESHOLD / MARGIN_ROUTER_MARGIN from its output.

The examples below are deliberately different from the eval questions, so the eval
measures generalisation rather than memorisation.
"""

import asyncio
import hashlib
import math
import os
from dataclasses import dataclass, field
from uuid import UUID

import companion as core

THRESHOLD = float(os.getenv("MARGIN_ROUTER_THRESHOLD", "0.70"))
MARGIN = float(os.getenv("MARGIN_ROUTER_MARGIN", "0.05"))
ANSWER = "answer"  # the passage in front of the reader is enough; no lookup needed
SEARCHES = {"search_notes", "search_library", "search_article"}

ROUTES: dict[str, list[str]] = {
    "propose_note": [
        "Write this down as a note for me.",
        "Add a margin note here that the argument is circular.",
        "Make a note that I want to come back to this.",
        "Jot down that this contradicts chapter two.",
        "Please save this point to my notes.",
        "Keep a note: the evidence here is thin.",
        "Record a thought on this paragraph for me.",
        "Put a note on this passage saying I agree.",
        "Can you note that this example is useful?",
        "Remember this for me as a note.",
    ],
    "search_notes": [
        "What have I noted about this before?",
        "Did I write anything about habits?",
        "Find my thoughts on motivation.",
        "What was my take on this topic last month?",
        "Show me notes I left about productivity.",
        "Have I commented on this idea in my margins?",
        "What did I say about burnout earlier?",
        "Look through my notes for anything on memory.",
        "Which of my notes mention focus?",
        "What were my reactions to the essay on cities?",
    ],
    "search_library": [
        "What do my other articles say about this?",
        "Have I read anything else on this subject?",
        "Find related ideas in the rest of my reading.",
        "Is there another article I saved that disagrees?",
        "What does my library say about habits?",
        "Who else in my reading talks about attention?",
        "Search everything I've saved for this concept.",
        "Which articles I've read mention this researcher?",
        "Do any of my saved essays cover sleep?",
        "Where else have I read about this theory?",
    ],
    "search_article": [
        "Where else in this piece does the author mention this?",
        "Does the article discuss this again further on?",
        "Find the part of this essay about solutions.",
        "Is this idea defined somewhere earlier in the text?",
        "Where in this article are the statistics?",
        "Does the writer return to this example?",
        "Search this article for the word incentive.",
        "What does the author say about this in the conclusion?",
        "Show me the section of this piece on history.",
        "Is there a later paragraph that answers this?",
    ],
    "get_passage": [
        "What comes right before this paragraph?",
        "Show me the surrounding paragraphs for context.",
        "Read the next paragraph too.",
        "Give me more context around this passage.",
        "How does this follow from the previous paragraph?",
        "What leads up to this point?",
        "Read around this passage and explain the flow.",
        "What does the paragraph after this say?",
    ],
    ANSWER: [
        "What does this mean?",
        "Summarise this paragraph.",
        "Is this argument convincing?",
        "Explain this like I'm new to the topic.",
        "What's the main claim here?",
        "Give me a counterargument to this.",
        "Why does the author say this?",
        "What is your opinion of this passage?",
        "Define the key term in this sentence.",
        "Rephrase this more simply.",
        "What assumption is the author making?",
        "Give me an example of this idea.",
    ],
}


@dataclass
class Route:
    tool: str | None  # a tool name, ANSWER, or None when the router abstains
    score: float
    runner_up: float
    scores: dict[str, float] = field(default_factory=dict)

    @property
    def confident(self) -> bool:
        return self.tool is not None

    @property
    def also(self) -> list[str]:
        """Other searches that also clear the threshold: a compound question."""
        return [
            name
            for name, score in self.scores.items()
            if name in SEARCHES and name != self.tool and score >= THRESHOLD
        ]


_exemplars: dict[str, dict[str, list[list[float]]]] = {}
_locks: dict[str, asyncio.Lock] = {}


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


async def exemplar_vectors(key: str) -> dict[str, list[list[float]]]:
    """Embed every example once per key and keep them for the life of the process."""
    fingerprint = hashlib.sha256(f"{key}:{core.EMBEDDING_MODEL}".encode()).hexdigest()
    if fingerprint in _exemplars:
        return _exemplars[fingerprint]
    lock = _locks.setdefault(fingerprint, asyncio.Lock())
    async with lock:
        if fingerprint not in _exemplars:
            items = [(name, text) for name, texts in ROUTES.items() for text in texts]
            batches = [items[i : i + 16] for i in range(0, len(items), 16)]
            vectors = await asyncio.gather(
                *(core.embeddings([text for _, text in batch], key, "query") for batch in batches)
            )
            grouped: dict[str, list[list[float]]] = {name: [] for name in ROUTES}
            for batch, batch_vectors in zip(batches, vectors, strict=True):
                for (name, _), vector in zip(batch, batch_vectors, strict=True):
                    grouped[name].append(vector)
            _exemplars[fingerprint] = grouped
    return _exemplars[fingerprint]


def classify(
    vector: list[float],
    exemplars: dict[str, list[list[float]]],
    threshold: float = THRESHOLD,
    margin: float = MARGIN,
) -> Route:
    # A route's score is its closest example: robust to one-off phrasings.
    scores = {
        name: max(cosine(vector, example) for example in examples)
        for name, examples in exemplars.items()
        if examples
    }
    ranked = sorted(scores, key=scores.__getitem__, reverse=True)
    best = ranked[0]
    runner_up = scores[ranked[1]] if len(ranked) > 1 else 0.0
    confident = scores[best] >= threshold and scores[best] - runner_up >= margin
    return Route(best if confident else None, scores[best], runner_up, scores)


async def route(question: str, key: str, owner: UUID) -> Route:
    exemplars, (vector, _) = await asyncio.gather(
        exemplar_vectors(key), core.query_vector(question[:300], key, owner)
    )
    return classify(vector, exemplars)
