# Cached citation audit

The deterministic parser previously recognized only `[n]`. Groq also emitted `【n】` and `【n†line annotation】`. Source numbers map to the original cached hit order, including older six-hit answers. Line annotations are retained as evidence, not validated by this source-membership metric. No LLM calls, retrieval, label edits, or cache edits were made.

## where is the Authorization header for basic auth built

Configuration: vector + test down-ranking; cached sources: 4; correctness: 5; faithfulness: 1.0.

Expected labels: `src/requests/auth.py::_basic_auth_str`, `src/requests/auth.py::HTTPBasicAuth`.

| Citation in answer | Cached hit | Lines | Matches expected labels |
|---|---|---|---|
| 【4】 | [4] `src/requests/auth.py::_basic_auth_str` | 34-75 | True |
| 【1】 | [1] `src/requests/auth.py::HTTPBasicAuth` | 85-113 | True |

Correct-source citation: False before parsing fix → True after fix.

<details><summary>Original cached answer</summary>

The **Authorization** header for HTTP Basic authentication is created in the private helper `_basic_auth_str` and then attached to the request by the `HTTPBasicAuth` callable.

1. **Header value generation** – `_basic_auth_str` (in `src/requests/auth.py` lines 34‑75) builds the Base‑64 encoded “Basic …” string from the supplied username and password. This function is responsible for all of the encoding and compatibility handling that produces the final header value【4】.

2. **Header insertion** – `HTTPBasicAuth.__call__` (in `src/requests/auth.py` lines 85‑113) receives a `PreparedRequest`, calls `_basic_auth_str(self.username, self.password)`, and assigns the result to the request’s `Authorization` header:

```python
def __call__(self, r: PreparedRequest) -> PreparedRequest:
    r.headers["Authorization"] = _basic_auth_str(self.username, self.password)
    return r
```
【1】

Thus, the Authorization header for basic auth is built by `_basic_auth_str` and applied to the request in `HTTPBasicAuth.__call__`. (A similar process is used for proxy authentication in `HTTPProxyAuth.__call__`, which sets the `Proxy-Authorization` header.)

</details>

## how are cookies stored

Configuration: vector + test down-ranking; cached sources: 6; correctness: 5; faithfulness: 1.0.

Expected labels: `src/requests/cookies.py`.

| Citation in answer | Cached hit | Lines | Matches expected labels |
|---|---|---|---|
| 【1†L1-L22】 | [1] `src/requests/cookies.py::create_cookie` | 494-528 | True |
| 【2†L1-L4】 | [2] `src/requests/cookies.py::RequestsCookieJar.__contains__` | 352-356 | True |
| 【4†L1-L8】 | [4] `src/requests/cookies.py::RequestsCookieJar.items` | 293-300 | True |
| 【3†L1-L9】 | [3] `src/requests/cookies.py::_copy_cookie_jar` | 479-491 | True |
| 【5†L1-L13】 | [5] `src/requests/models.py::PreparedRequest.prepare_cookies` | 699-720 | False |

Correct-source citation: False before parsing fix → True after fix.

<details><summary>Original cached answer</summary>

In **Requests** the actual cookie data lives inside a **`CookieJar`** object (normally a `RequestsCookieJar`, which is a thin subclass of `http.cookiejar.CookieJar`).  

* When a user supplies a cookie (e.g. via `requests.cookies.create_cookie`), the helper builds a **`cookielib.Cookie`** instance with all of the standard cookie attributes (name, value, domain, path, expires, etc.) and returns it【1†L1-L22】.  

* The `RequestsCookieJar` stores those `Cookie` objects in the underlying `CookieJar` data structures. Because it subclasses `CookieJar`, it behaves like a mutable mapping: you can test for membership with `__contains__` (which falls back to `CookieConflictError` handling)【2†L1-L4】, iterate over items with `items()` (which simply returns a list of name‑value tuples from the jar)【4†L1-L8】, and copy the whole jar with `_copy_cookie_jar` (which either calls the jar’s own `copy()` method or makes a shallow copy of each `Cookie`)【3†L1-L9】.  

* When a `PreparedRequest` is built, the request object keeps a reference to the jar in `self._cookies`. The `prepare_cookies` method either uses the supplied `CookieJar` directly or converts a plain dict into a `RequestsCookieJar` via `cookiejar_from_dict`. It then generates the actual HTTP “Cookie” header from the jar’s contents【5†L1-L13】.  

Thus, **cookies are stored as `cookielib.Cookie` objects inside a `CookieJar` (usually a `RequestsCookieJar`).** The jar provides dict‑like access (`items`, `__contains__`, etc.) and can be copied or inspected, while the request preparation step extracts the stored cookies to form the outbound header.

</details>

## HTTPAdapter

Configuration: vector + test down-ranking; cached sources: 4; correctness: 5; faithfulness: 1.0.

Expected labels: `src/requests/adapters.py::HTTPAdapter`.

| Citation in answer | Cached hit | Lines | Matches expected labels |
|---|---|---|---|
| 【2†L1-L23】 | [2] `src/requests/adapters.py::HTTPAdapter` | 158-199 | True |
| 【2†L24-L33】 | [2] `src/requests/adapters.py::HTTPAdapter` | 158-199 | True |
| 【3†L1-L3】 | [3] `src/requests/adapters.py::HTTPAdapter.__getstate__` | 223-224 | True |
| 【4†L1-L22】 | [4] `src/requests/adapters.py::HTTPAdapter.get_connection` | 512-553 | True |
| 【4†L23-L33】 | [4] `src/requests/adapters.py::HTTPAdapter.get_connection` | 512-553 | True |
| 【1†L737-L748】 | [1] `src/requests/adapters.py::HTTPAdapter.send (part 2/2)` | 737-748 | True |
| 【4†L1-L12】 | [4] `src/requests/adapters.py::HTTPAdapter.get_connection` | 512-553 | True |

Correct-source citation: False before parsing fix → True after fix.

<details><summary>Original cached answer</summary>

**`HTTPAdapter`** is the built‑in transport adapter that `requests` uses to talk to HTTP and HTTPS URLs.  
It implements the *Transport Adapter* interface required by a `Session` and internally relies on **urllib3** connection pools.

### What it does
* Provides a generic way for a `Session` to obtain a connection pool, send a request, and build a `Response` object.  
* Handles pooling, retries, proxy configuration and TLS context handling.  
* Is the default adapter that `Session.mount()` attaches to the `"http://"` and `"https://"` prefixes.

### Key parts in the source

| Part | Description |
|------|-------------|
| **Class definition & docstring** | Shows the purpose, parameters and typical usage. It mentions the pool‑related arguments (`pool_connections`, `pool_maxsize`, `max_retries`, `pool_block`).【2†L1-L23】 |
| **Attributes (`__attrs__`)** | Lists the state that will be pickled/serialized (`max_retries`, `config`, `_pool_connections`, `_pool_maxsize`, `_pool_block`).【2†L24-L33】 |
| **`__getstate__`** | Returns a dict of those attributes so the adapter can be pickled.【3†L1-L3】 |
| **`get_connection`** | (Deprecated) Returns a urllib3 `HTTPConnectionPool` for a given URL, handling proxy selection and raising `InvalidProxyURL` if the proxy string is malformed. It also emits a deprecation warning pointing developers to the newer `get_connection_with_tls_context`.【4†L1-L22】【4†L23-L33】 |
| **`send` (second half)** | After the request is performed, the method catches urllib3‑specific exceptions (`_SSLError`, `_HTTPError`, `ReadTimeoutError`, `_InvalidHeader`) and re‑raises the appropriate `requests` exceptions (`SSLError`, `ReadTimeout`, `InvalidHeader`). Finally it builds the `Response` via `self.build_response(request, resp)`.【1†L737-L748】 |

### Typical usage

```python
import requests
s = requests.Session()
adapter = requests.adapters.HTTPAdapter(max_retries=3)
s.mount('http://', adapter)   # now all http:// requests go through this adapter
```

The adapter will:

1. **Create / reuse** a urllib3 connection pool (`get_connection` / `get_connection_with_tls_context`).  
2. **Send** the prepared request, handling low‑level errors as shown in `send`.  
3. **Build** a `requests.Response` object from the urllib3 response.

### When to subclass / customize

* Subclassing is supported for advanced use‑cases (e.g., custom TLS contexts).  
* The old `get_connection` method is deprecated; new subclasses should implement `get_connection_with_tls_context` as indicated in the deprecation warning.【4†L1-L12】

---

**In short:** `HTTPAdapter` is the core component that bridges a `requests.Session` to urllib3’s connection pools, managing pooling, retries, proxies, and error translation. Its implementation lives in `src/requests/adapters.py`.

</details>

## Remaining perfect-score mismatches

These cite different sources from the unchanged gold labels:

- how does requests guess the encoding of a response
- How can I run my own code every time a response comes back?
- Why is a password being sent even though I never passed one?
