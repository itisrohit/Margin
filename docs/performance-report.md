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
