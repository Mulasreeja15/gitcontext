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
