# Retrieval evaluation

Repo: psf/requests, 45 labelled questions, ranks checked up to 10.

| Configuration | Recall@1 | Recall@3 | Recall@5 | MRR@10 | Median latency |
|---|---|---|---|---|---|
| Vector only (Phase 1 baseline) | 60% | 69% | 76% | 0.680 | 19 ms |
| Vector + test down-ranking | 69% | 76% | 91% | 0.750 | 22 ms |
| Hybrid (vector + BM25, RRF) | 42% | 62% | 73% | 0.561 | 20 ms |
| Hybrid + test down-ranking | 53% | 78% | 89% | 0.679 | 20 ms |
| Hybrid + test down-ranking + rerank | 44% | 69% | 82% | 0.592 | 1199 ms |

## Rank of first correct result per question

| Question | Vector only (Phase 1 baseline) | Vector + test down-ranking | Hybrid (vector + BM25, RRF) | Hybrid + test down-ranking | Hybrid + test down-ranking + rerank |
|---|---|---|---|---|---|
| How does requests decide whether to follow a redirect? | 9 | 4 | - | 4 | 2 |
| how are cookies stored | 2 | 1 | 4 | 2 | 3 |
| where is the timeout handled | 9 | 5 | 5 | 4 | 3 |
| HTTPAdapter | 1 | 1 | 1 | 1 | 1 |
| resolve_redirects | 1 | 1 | 1 | 1 | 1 |
| where is the Authorization header for basic auth built | 1 | 1 | 2 | 2 | 7 |
| how are query parameters added to the URL | - | 7 | - | 5 | 8 |
| how does requests guess the encoding of a response | 6 | 5 | 6 | 4 | 8 |
| which exception is raised when there are too many redirects | 1 | 1 | 1 | 1 | 1 |
| how are proxy settings taken from environment variables | 1 | 1 | 1 | 1 | 1 |
| how is a file uploaded as multipart form data | 5 | 1 | 5 | 1 | 1 |
| how are connection retries configured | 1 | 1 | 1 | 1 | 3 |
| How is a JSON response body parsed? | 1 | 1 | 1 | 1 | 1 |
| When does raise_for_status raise an error? | 1 | 1 | 1 | 1 | 1 |
| How is a response body streamed in chunks? | 1 | 1 | 1 | 1 | 2 |
| How are header names made case-insensitive? | 2 | 2 | 2 | 2 | 1 |
| How does requests verify SSL certificates? | 1 | 1 | 3 | 3 | 1 |
| Why is the Authorization header removed when redirecting to another host? | 1 | 1 | 1 | 1 | 2 |
| How does the HTTP method change after a 303 redirect? | 2 | 1 | 2 | 1 | 4 |
| How are session settings merged with per-request settings? | 1 | 1 | 1 | 1 | 1 |
| Where is the default User-Agent header set? | 1 | 1 | 1 | 1 | 1 |
| How are credentials read from a .netrc file? | 1 | 1 | 2 | 1 | 4 |
| How does digest authentication build its header? | 1 | 1 | 2 | 2 | 2 |
| How does a session choose the transport adapter for a URL? | 1 | 1 | 1 | 1 | 3 |
| How are response hooks called? | 6 | 5 | 8 | 5 | 4 |
| How is the length of a request body calculated? | 1 | 1 | 1 | 1 | 1 |
| How are cookies from a response saved into the cookie jar? | 1 | 1 | 1 | 1 | 1 |
| How does requests decide whether a host should bypass the proxy? | 1 | 1 | 1 | 1 | 1 |
| How are Link headers in a response parsed? | 1 | 1 | 1 | 1 | 1 |
| Which exception is raised when a URL has no scheme like http? | 1 | 1 | 1 | 1 | 3 |
| prepare_body | 1 | 1 | 1 | 1 | 1 |
| Session.send | 5 | 5 | 2 | 2 | 1 |
| Why do I lose my login credentials when the server sends me to a different website? | 1 | 1 | 9 | 2 | 2 |
| My request hangs forever, how does it give up after some seconds? | 7 | 3 | 4 | 2 | 6 |
| How does it reuse the same network connection across many requests? | 7 | 5 | 5 | 3 | 5 |
| Why is the text of my response showing weird characters? | 1 | 1 | 2 | 2 | 1 |
| How is my dictionary turned into form data sent to the server? | 5 | 5 | - | 9 | - |
| What stops it from bouncing between pages forever? | - | - | - | - | - |
| How does it pick a different proxy for secure and normal websites? | 1 | 1 | 2 | 1 | 2 |
| Where does it figure out how big an uploaded file is? | 1 | 1 | 1 | 1 | 5 |
| How can I run my own code every time a response comes back? | - | - | - | - | - |
| Why is a password being sent even though I never passed one? | - | - | - | 8 | - |
| How do I make it accept a server with a self-signed certificate? | 6 | 3 | 8 | 3 | 4 |
| How are spaces and special characters in a web address made safe? | 1 | 1 | 7 | 6 | 1 |
| Why does my POST become a GET after the server redirects me? | 2 | 1 | 8 | 1 | 1 |
