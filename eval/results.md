# Retrieval evaluation

Repo: psf/requests, 12 labelled questions, ranks checked up to 10.

| Configuration | Recall@1 | Recall@3 | Recall@5 | MRR@10 | Median latency |
|---|---|---|---|---|---|
| Vector only (Phase 1 baseline) | 50% | 58% | 67% | 0.591 | 28 ms |
| + BM25 hybrid (RRF) | 42% | 50% | 75% | 0.526 | 29 ms |
| + Test down-ranking | 50% | 67% | 100% | 0.662 | 21 ms |
| + Cross-encoder rerank | 42% | 75% | 75% | 0.574 | 1156 ms |
| + LLM query rewriting | 50% | 92% | 100% | 0.697 | 1631 ms |

## Rank of first correct result per question

| Question | Vector only (Phase 1 baseline) | + BM25 hybrid (RRF) | + Test down-ranking | + Cross-encoder rerank | + LLM query rewriting |
|---|---|---|---|---|---|
| How does requests decide whether to follow a redirect? | 9 | - | 4 | 2 | 3 |
| how are cookies stored | 2 | 4 | 2 | 3 | 2 |
| where is the timeout handled | 9 | 5 | 4 | 3 | 5 |
| HTTPAdapter | 1 | 1 | 1 | 1 | 1 |
| resolve_redirects | 1 | 1 | 1 | 1 | 2 |
| where is the Authorization header for basic auth built | 1 | 2 | 2 | 7 | 1 |
| how are query parameters added to the URL | - | - | 5 | 8 | 3 |
| how does requests guess the encoding of a response | 6 | 6 | 4 | 8 | 2 |
| which exception is raised when there are too many redirects | 1 | 1 | 1 | 1 | 1 |
| how are proxy settings taken from environment variables | 1 | 1 | 1 | 1 | 1 |
| how is a file uploaded as multipart form data | 5 | 5 | 1 | 1 | 1 |
| how are connection retries configured | 1 | 1 | 1 | 3 | 1 |

