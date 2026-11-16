# Margin companion: measured results

Live NVIDIA hosted API and local PostgreSQL. Two Aeon articles, 62 indexed passages. Thirteen text requests at concurrency two; repeated questions exercise query-vector caching. One separate synthesized English voice trial.

| Route | Samples | p50 first text | p95 first text | p50 full answer | p95 full answer |
|---|---:|---:|---:|---:|---:|
| passage | 6 | 247.4 ms | 263.4 ms | 1576.1 ms | 1906.6 ms |
| library | 7 | 300.1 ms | 913.1 ms | 3021.6 ms | 3620.2 ms |

Voice: release → final transcript **264.2 ms**; release → first answer **717.1 ms**. This is one synthesized-speech run, not a p95 or real microphone benchmark.

All 13 text requests completed with citation IDs that exist in their supplied evidence. That does not prove citation entailment or nuanced reasoning quality. Both articles appeared among the library evidence.

Initial article indexing took 17–23 seconds per article and runs separately from answering. Extraction and indexing details are in the JSON report.

With six passage and seven library samples, nearest-rank p95 is the maximum observed value. These numbers are suitable as a small demo result, not a production latency guarantee.

Checks: 18 automated tests passed, including PostgreSQL integration; backend lint/types and frontend lint/build passed. No browser UI or microphone verification was performed.

The reader exposes Ask Margin, passage/library scope, source links, cancellation, and Response timings. There is no spoken answer playback. Exact vector search remains in use; large-library indexes and reranking need a larger labeled benchmark.

## Agent mode (tool calling)

**Not yet measured live.** Agent mode (`MARGIN_AGENT=1`, see `docs/companion.md`) was
built and tested offline only: 22 new unit/integration tests with a mocked model cover
tool scoping by reader and article, the step cap, tool errors, fallback when tool calling
is rejected, citation stripping, and `propose_note` never writing. No NVIDIA key was
available while building it, so there are no tool-selection or latency numbers yet.

NVIDIA's model card lists function calling (including parallel calls) as supported for
`nvidia/nemotron-3.5-bolt-e4b`; whether the hosted model chooses tools reliably for these
questions is exactly what the eval checks.

To measure: start the API with `MARGIN_AGENT=1`, then run
`NVIDIA_API_KEY=... uv run --directory api python benchmarks/agent_eval.py`. It writes
`api/benchmarks/results/agent_eval.json` and reports tool-selection accuracy, average
steps, the share of answers whose citations are all valid, fallbacks, and client-measured
p50/p95 time to first text and full answer.

Tool choice starts with a semantic router whose threshold and margin are placeholders
until `benchmarks/router_eval.py` is run (key only; no server needed). It reports
precision (correct when it acts) and coverage (how often it acts) across settings, and
writes `api/benchmarks/results/router_eval.json`. Report the chosen setting with both
numbers, and note its 54 questions were written by hand.

Limitations to keep when reporting: 15 hand-written questions over a 5-article synthetic
library, one run, one model. One changed answer moves accuracy by about 7 points, and p95
of 15 samples is close to the maximum. Tool-selection "accuracy" checks that the expected
tool was used, not that the answer was good; pair it with `grounding_eval.py` for
citation support.
