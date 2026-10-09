Q: how are cookies stored
Expected: src/requests/cookies.py
Result: rank 2 (create_cookie). Rank 1 was a test again. RequestsCookieJar class summary not in top 6.
Q: where is the timeout handled
Expected: src/requests/adapters.py (HTTPAdapter.send)
Result: not found. Got exception classes (Timeout, ReadTimeout) that define errors, not handling code.
Q: HTTPAdapter
Expected: src/requests/adapters.py (class HTTPAdapter)
Result: rank 2 (class summary). All 6 results from the right class. Success.
Q: resolve_redirects
Expected: src/requests/sessions.py (resolve_redirects)
Result: rank 1. Success.

## Baseline (Phase 1, vector search only)
Hit rate in top 6: 3/5 (60%)
Exact names: 2/2 found. Natural questions: 1/3 found.
Main problems: tests outrank source code; real logic hidden in big methods.

## Reranker experiment
- ms-marco-MiniLM-L-6-v2: MRR 0.574, 1.6 s/query
- bge-reranker-base: MRR 0.503, 9.2 s/query
- Best without reranker: MRR 0.662, 31 ms/query
Conclusion: general-purpose rerankers (trained on web search) hurt code retrieval.
Bigger model = worse and 6x slower, so the problem is domain mismatch, not size.
Kept as optional --rerank flag, off by default.
