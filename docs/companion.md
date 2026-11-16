# Reading companion

Choose **Ask Margin** beside a passage. Hold the existing voice button or Space to ask;
questions are not saved as personal thoughts. Typed questions and follow-ups also work.
Choose **My library** to search indexed article passages and saved thought embeddings.
Citation buttons navigate to the source passage in saved articles.

Articles are indexed when the companion opens, separately from answering. The readiness
message reports progress and failures; library results can be incomplete while indexing.
Notes still use the existing memory endpoints. Conversations are held in memory for the
current page session. There is no synthesized answer playback in this version.

## API contract

All endpoints require `Authorization: Bearer <NVIDIA key>`. Keys are request-scoped and
never stored in passage records. Like the existing memory API, `clientId` identifies a
local browser installation; it is not account authentication.

- `POST /companion/index`: `{clientId, url, title, paragraphs}`. Returns
  `{indexed: true, cached: boolean, passages?: number}`. Content/model fingerprints skip
  unchanged articles. Publication is atomic after successful embedding of every batch.
- `POST /companion/ask`: `{clientId, question, articleUrl, title, paragraphIndex,
contextStart, context, scope: "passage" | "library", history?: [{role, content}]}`.
  Returns newline-delimited JSON: `sources`, zero or more `delta` events, then `done`
  with `metrics`. A provider failure emits `error` instead of `done`; invalid input or
  missing credentials returns HTTP 422/401 before streaming. Clients must treat missing
  `done` as an interrupted answer. Sources have stable IDs for that response only.

FastAPI's `/docs` contains complete request schemas and bounds.

## Performance decisions

Passage questions bypass embeddings and database retrieval. Library queries reuse a
bounded, model/key/client-scoped LRU of query vectors, with concurrent requests for the
same vector deduplicated. One pooled asynchronous HTTP client reuses provider connections.
Blocking PostgreSQL work runs off the event loop. Answers stream immediately; requests
can be cancelled in the UI. Context, history, output, embedding batch sizes and indexing
concurrency are bounded. Oversized paragraphs split with overlap and retain source anchors.
A failed provider batch is split into smaller batches; rejected keys are not retried.

Search currently uses exact cosine distance within one client's records. Approximate
HNSW indexing and an external reranker are deliberately not enabled for a 62-passage
benchmark: their speed/recall tradeoff needs a larger labeled corpus. There is no hosted
NVIDIA prefix-cache toggle or fabricated server-side GPU optimization.

## Reproduce the live benchmark

With the normal API and PostgreSQL running, set `NVIDIA_API_KEY` in your shell, then run:

```sh
uv run --directory api python benchmarks/companion.py
```

`MARGIN_API_URL` overrides the API address. `MARGIN_BENCHMARK_OUTPUT` overrides the
report path (default `/tmp/margin-companion-benchmark.json`). Two public Aeon articles
are imported into an isolated UUID's database records. The benchmark runs one cold library
question followed by twelve requests at concurrency two and removes its records afterward.
It does not change your browser library or notes. Extraction/indexing timings are separate
from answer timings. Questions repeat intentionally to expose query-cache behavior.

Percentiles use nearest rank. With only six passage and seven library samples, p95 is
essentially the slowest observation. These are a small demo benchmark, not a production
SLA or a large-library load test. Citation-ID validity checks source references exist;
it does not establish factual entailment, retrieval recall, or nuanced reasoning accuracy.

The UI's Response timings include server stages and client first-answer time. For voice
questions it also records release-to-transcript and release-to-first-answer. Those client
timers use one monotonic clock. Server/model timers start after request validation, so
client-visible latency is the primary demo number.

Automated contracts and failure cases:

```sh
RUN_POSTGRES_INTEGRATION=1 uv run --directory api python -m unittest discover -s tests -v
```

The HTTP contract tests simulate NVIDIA responses locally. The benchmark calls real NVIDIA
and PostgreSQL. A synthesized-speech check measures the live speech WebSocket plus answer
API; it does not replace testing microphone capture and browser rendering on real devices.

## Live voice benchmark

Provide a 16 kHz mono 16-bit PCM WAV file in `MARGIN_BENCHMARK_AUDIO` (default
`/tmp/margin-question.wav`), then run:

```sh
uv run --directory api python benchmarks/voice.py
```

This streams audio at real-time pace to the existing speech WebSocket, then sends its
transcript to the companion. It reports release-to-final-transcript and release-to-first
answer. The fixture used during development was synthesized English speech. The report
is written to `/tmp/margin-voice-benchmark.json`; no API key is included.
