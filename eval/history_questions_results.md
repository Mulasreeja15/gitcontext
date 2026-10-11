# Retrieval evaluation

Repo: psf/requests, 8 labelled questions, sources: history, ranks checked up to 10.

| Configuration | Recall@1 | Recall@3 | Recall@5 | MRR@10 | Median latency |
|---|---|---|---|---|---|
| Vector only (Phase 1 baseline) | 88% | 88% | 88% | 0.875 | 31 ms |
| Vector + test down-ranking | 88% | 88% | 88% | 0.875 | 33 ms |
| Hybrid (vector + BM25, RRF) | 88% | 88% | 88% | 0.889 | 54 ms |
| Hybrid + test down-ranking | 88% | 88% | 88% | 0.889 | 54 ms |
| Hybrid + test down-ranking + rerank | 100% | 100% | 100% | 1.000 | 979 ms |

## Rank of first correct result per question

| Question | Vector only (Phase 1 baseline) | Vector + test down-ranking | Hybrid (vector + BM25, RRF) | Hybrid + test down-ranking | Hybrid + test down-ranking + rerank |
|---|---|---|---|---|---|
| Why did Requests clarify the README's bad commit error as a bad commit timestamp? | - | - | 9 | 9 | 1 |
| Why was URLRequired removed from the documented Requests exceptions? | 1 | 1 | 1 | 1 | 1 |
| Why did Requests start considering TLS settings when choosing a connection pool? | 1 | 1 | 1 | 1 | 1 |
| Why was JSONDecodeError's __reduce__ method fixed for pickling? | 1 | 1 | 1 | 1 | 1 |
| Why can the total connection timeout exceed the timeout specified by the caller? | 1 | 1 | 1 | 1 | 1 |
| Why was an automatic retry example added to the advanced usage documentation? | 1 | 1 | 1 | 1 | 1 |
| Why was the Python version upper bound of less than 4 removed from python_requires? | 1 | 1 | 1 | 1 | 1 |
| Why did Requests replace chardet with charset_normalizer as its default dependency? | 1 | 1 | 1 | 1 | 1 |
