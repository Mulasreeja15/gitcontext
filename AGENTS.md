# GitContext: guidance for future coding tasks

## Project summary

GitContext is a retrieval-augmented generation (RAG) system for Python codebases. It parses code with tree-sitter into function, class, and method chunks with context headers containing file paths and symbol names. Small classes stay intact; large classes produce a summary plus method chunks. Oversized chunks are split into line windows. Module-level code and Markdown/RST/text documentation are also indexed.

Chunks are embedded locally with `BAAI/bge-small-en-v1.5` and stored in local Qdrant at `data/qdrant`, with one collection per repository (`repo__<owner>__<name>`). State lives under `data/index_state/<owner>__<name>.json`. The default repository is `psf/requests`. The default retrieval pipeline uses vector search and test-file down-ranking. Optional BM25 hybrid retrieval (`--bm25`) uses a code-aware tokenizer and Reciprocal Rank Fusion (RRF); optional cross-encoder reranking (`--rerank`) uses `cross-encoder/ms-marco-MiniLM-L-6-v2`. Optional LLM query rewriting (`--rewrite`) retrieves with both the original and rewritten query and fuses their rankings. Payload/BM25 caches belong to each Retriever, and rewrite/judge cache keys include repository identity.

Groq serves `openai/gpt-oss-120b` for query rewriting and grounded answers. Answers use numbered source excerpts with file paths and line ranges, cite claims inline with `[n]`, and must acknowledge when the supplied sources do not contain the answer.

## Module map

| File | Responsibility |
|---|---|
| `gitcontext/__init__.py` | Package initializer. |
| `gitcontext/config.py` | Central settings: dotenv loading, models, local storage paths, retrieval tuning, chunk/file limits, supported extensions, and skipped directories. |
| `gitcontext/chunker.py` | `Chunk` metadata, deterministic UUIDs, embedding context headers, tree-sitter Python parsing, decorated definitions, large-class summaries and methods, line-window splitting, and document sections. |
| `gitcontext/ingest.py` | Resolve local repositories or clone Git URLs into `data/repos`; walk supported files, skip hidden/generated/oversized content, and collect chunks with repository-relative POSIX paths. |
| `gitcontext/indexing.py` | Full/incremental indexing, atomic per-repo state, disk/chunk limits, Git diffs/managed-clone pulls, freshness checks, interrupted-update recovery, and summaries. |
| `gitcontext/repositories.py` | Validate repository identities, derive collection/state names, infer URL/remote identities, list indexes, and remove a selected collection/state. |
| `gitcontext/store.py` | `Embedder` loads sentence-transformers and produces normalized embeddings; `VectorStore` resets/upserts the Qdrant collection, searches by cosine similarity, and scrolls payloads for BM25. |
| `gitcontext/retriever.py` | Retrieval options/results, identifier tokenization, test-file detection, lazy BM25/reranker setup, vector and keyword rankings, RRF, test penalties, query rewriting, reranking, and latency measurements. |
| `gitcontext/answer.py` | Groq `_chat()` client with 429 retries, source formatting, cited-answer prompts, and cached query rewriting. |
| `gitcontext/judge.py` | Answer-quality judge, JSON validation/retry, gold context, deterministic citation checks, resumable JSONL cache, and Markdown reports. |
| `gitcontext/api.py` | FastAPI lifespan owns one QdrantClient/embedding model, per-repo Retriever caches, repo routing/listing/deletion, one background indexing job with status polling, and source-first SSE answers. Locks serialize shared model/storage work. |
| `gitcontext/service_indexing.py` | Strict HTTPS GitHub URL validation, shallow clones, background full-index pipeline using injected client/model, batch embedding progress, and filename-based language notices. |
| `gitcontext/history.py` | Batched commit message/numstat parsing, shallow detection/unshallow option, paginated optional GitHub PR discussions with rate-limit handling and per-repo JSON cache, history-only replacement preserving code and unavailable PR data. |
| `gitcontext/static/index.html` | Self-contained UI served at `/`: vanilla JavaScript, DOM-only Markdown rendering, SSE answers, search-only mode, source cards, and citation navigation. |
| `judge.py` | Answer-quality evaluation CLI: default retrieval, optional `--rewrite`, `--limit N`, and `--no-cache`; writes `eval/judge_results.md`. |
| `main.py` | CLI entry point for `index`, `update`, `status`, `repos`, `remove`, `search`, `ask`, and `serve --host 127.0.0.1 --port 8000`; displays ranked sources and timings. Indexing replaces only the selected collection. |
| `evaluate.py` | Ablation harness: evaluates first relevant ranks up to 10, Recall@1/3/5, MRR@10, and median latency; writes summary and per-question ranks to `eval/results.md`. |
| `check_labels.py` | Checks that every expected evaluation target matches a chunk in the current index. |
| `eval/questions.jsonl` | 45 labelled questions for `psf/requests`, with expected file or `path::Symbol` targets. |
| `README.md` | Architecture, usage, published evaluation results, limitations, and roadmap. |
| `notes.md` | Historical manual baseline observations and reranker experiments; their smaller-set figures are distinct from the README evaluation below. |

## Running on Windows

Indexing batches Git history to attach `last_commit_sha`, `last_author` (name only),
and `last_modified` (ISO commit time) to each file's chunks. These fields stay out
of embedding text. Search/ask accept `--since YYYY-MM-DD` (inclusive UTC date) and
`--author NAME` (exact case-sensitive last-author match). API bodies accept `since`
and `author`; vector and hybrid candidates respect the same Qdrant payload filters.

Run commands from the repository root and use the existing virtual environment. PowerShell activation:

```powershell
.\.venv\Scripts\Activate.ps1
```

The CLI commands are `python main.py index/search/ask`, `python evaluate.py --with-rewrite --details`, and `python check_labels.py`. For coding tasks, invoke the existing environment's interpreter explicitly:

```powershell
.\.venv\Scripts\python.exe main.py index https://github.com/psf/requests
.\.venv\Scripts\python.exe main.py search "where is the timeout handled"
.\.venv\Scripts\python.exe main.py ask "How does requests decide whether to follow a redirect?"
.\.venv\Scripts\python.exe main.py search "HTTPAdapter" --bm25
.\.venv\Scripts\python.exe main.py search "HTTPAdapter" --rerank
.\.venv\Scripts\python.exe main.py search "HTTPAdapter" --include-tests
.\.venv\Scripts\python.exe evaluate.py
.\.venv\Scripts\python.exe evaluate.py --with-rewrite --details
.\.venv\Scripts\python.exe check_labels.py
.\.venv\Scripts\python.exe -m pytest
```

`index` accepts a local directory or Git URL and resets only its repository collection. URL/remotes infer `owner/name`; local-only directories use `local/<directory>`. New clones use owner-qualified names; matching older clones are reused. `search`, `ask`, `status`, and `update` accept `--repo owner/name` (default `psf/requests`). `update --repo pallets/click` uses the saved checkout; an optional positional source is also supported. `repos` lists counts/commits/staleness; `remove owner/name` removes only index/state, preserving the checkout. Reindexing requests removes the legacy `gitcontext` collection. Evaluation, label checks, and judge evaluation still default to requests. API search/ask/stream accept optional `repo`; health accepts a repo query parameter. Qdrant locks the shared data folder, so stop the server before any CLI storage operations.

The service owns exactly one QdrantClient. Inject it and the loaded embedder into repo-local retrievers; never create another local client from an API route or background job. Optional reranker models are shared too. `GET /repos` lists indexes; `DELETE /repos/{owner}/{name}` removes selected state/collection/caches without closing the shared client. `POST /index` validates public HTTPS GitHub URLs, starts one background thread, and clones new checkouts with `--depth 1`. `GET /index/status` reports idle/cloning/chunking/embedding/done/error with repo, chunk progress, and safe error text. Busy jobs return 409. Job status has its own lock so progress polling stays responsive; shared embedding batches and database publication use the retrieval lock. Shutdown joins an active index job before closing storage. The web picker selects repo-scoped search/answers; Add repository polls status and auto-selects on completion. Language warnings use filename counts (less than 20% Python by default); no source contents are read for this detection. Tests mock the pipeline/client/models/LLM and must not trigger clones or Groq calls.

Limits in config are 200 MB on disk (including Git history) and 20,000 chunks. Check limits before embedding/database mutation, including the projected incremental chunk count. `search` retrieves chunks; `ask` additionally calls Groq. `-k` defaults to 6; `--include-tests` disables test down-ranking. During coding commands set `$env:PYTHON_DOTENV_DISABLED='1'`; never inspect the credential file or print keys.

## Published evaluation results

History sources are opt-in: `main.py index-history --repo owner/name` examines the
latest `HISTORY_MAX_COMMITS=500` commits and imports up to `PR_MAX=200` merged PR
discussions when `os.getenv(config.GITHUB_KEY_NAME)` finds a token. Never open the
credential file or print tokens/headers. Missing credentials skip PRs gracefully.
`--commits-only` forbids API calls; shallow clones require explicit `--unshallow`
to fetch full history. Cache response JSON under per-repo `data/github_cache`,
never credentials. Respect GitHub Retry-After and rate/reset headers; long waits
are reported so users can retry. Commits touching only excluded/unsupported files
are omitted; no full diffs or author-email metadata are indexed.

Search/ask accept `--sources code|history|all` (code by default); API bodies and
the web selector use the same options. Code filters exclude commit/PR payloads,
including in BM25; BM25 corpora are cached separately per source selection.
History metadata includes SHA/PR number, title, author, date, link, and changed
files; API sources expose it under `history`. Keep citations numbered and render
links only for validated GitHub commit/PR URLs. Code rebuilds preserve history
when the embedding model stays the same. History reindexing preserves old PRs
when import is skipped; all sources share the collection chunk limit.

`eval/history_questions.jsonl` contains eight separately labelled why questions;
evidence is in `eval/history_labels.md`. Never conflate these with the 45 code
questions. `is_match` supports `commit::<SHA prefix>` and `pr::<number>`;
`evaluate.py --questions <file>` infers the source selection and writes a separate
report for alternate question files. Run both code and history evaluations after
changing source filtering. Tests use mocked Git/API responses and memory Qdrant.

Copied from `README.md`: `psf/requests`, 45 labelled questions spanning exact identifiers, technical questions, and plain-English paraphrases. These are published results, not a fresh run.

| Configuration | Recall@1 | Recall@3 | Recall@5 | MRR@10 | Median latency |
|---|---|---|---|---|---|
| Vector only (baseline) | 60% | 69% | 76% | 0.680 | 17 ms |
| **Vector + test down-ranking** (default) | **69%** | 76% | **91%** | **0.750** | **16 ms** |
| Hybrid (vector + BM25, RRF) | 42% | 62% | 73% | 0.561 | 17 ms |
| Hybrid + test down-ranking | 53% | 78% | 89% | 0.679 | 18 ms |
| Hybrid + test down-ranking + cross-encoder rerank | 44% | 69% | 82% | 0.592 | 1070 ms |
| **Vector + test down-ranking + LLM rewrite** (`--rewrite`) | 67% | 84% | 93% | **0.768** | ~1.3 s |
| Hybrid + test down-ranking + LLM rewrite | 56% | **87%** | **96%** | 0.725 | ~1.3 s |

Key findings:

- Test files often outrank implementation because their names share query vocabulary. Down-ranking tests improved Recall@5 from 76% to 91% and MRR from 0.680 to 0.750 without added latency, motivating the default.
- BM25 hurts natural-language retrieval without rewriting: MRR falls from 0.750 to 0.679 with test down-ranking. An early 12-question evaluation favored hybrid retrieval, but the larger, harder set reversed that conclusion.
- LLM rewriting bridges some paraphrases to code identifiers and gives the best MRR (0.768), at roughly one LLM call and 1.3 seconds per query. Rewrites are cached per process.
- Combining rewriting with BM25 gives the best Recall@5 (96%, versus 93% without BM25), while lowering top-1 precision and MRR.
- General-purpose web-search rerankers hurt code retrieval: MiniLM lowers MRR to 0.592 and adds about one second; the larger `bge-reranker-base` was worse and roughly six times slower in historical experiments. Domain mismatch is the likely cause.
- Vocabulary gaps and implicit behavior remain limitations: paraphrased response-hook questions fail in every configuration, and automatic `.netrc` credential lookup is rarely retrieved when the question omits that mechanism.

## Rules for all future changes

1. Never open, read, print, modify, or commit `.env` or API keys. Keys come from `.env` via python-dotenv in `gitcontext/config.py`; use environment lookup in application code without exposing values. Avoid commands that could display secrets.
2. Do not change labels in `eval/questions.jsonl` unless explicitly asked. Do not adjust ground truth to make retrieval results look better.
3. Keep all settings in `gitcontext/config.py`, including new model choices, paths, limits, and tuning parameters.
4. Keep the code Windows-compatible. Use PowerShell-compatible commands and `pathlib` for paths; preserve repository-relative POSIX paths in chunk metadata and evaluation targets.
5. The Groq free tier is rate-limited. All LLM calls must go through `gitcontext/answer.py`'s `_chat()`, which retries HTTP 429 using `retry-after` or exponential backoff. Do not add a separate direct LLM request path.
6. After any retrieval change, run `python evaluate.py` with `.venv\Scripts\python.exe` and report the full results table. Include `--with-rewrite --details` when assessing rewriting, and use `check_labels.py` to diagnose missing targets without editing labels. Evaluation writes `eval/results.md`.
7. Prefer small, readable functions with docstrings. This is a portfolio project, so code clarity matters.
8. Add or update pytest tests for new logic, and run relevant tests using `.venv\Scripts\python.exe -m pytest`.
9. Use the existing virtual environment. Run Python as `.venv\Scripts\python.exe`; do not create a replacement environment or rely on a global interpreter.

## Complete command reference

Use `.venv\Scripts\python.exe` for every Python command below. Set
`$env:PYTHON_DOTENV_DISABLED = '1'` for coding/verification commands; never
open, print, or modify `.env`. Normal user startup may load credentials via dotenv.

```powershell
.\.venv\Scripts\python.exe main.py index <path-or-url>
.\.venv\Scripts\python.exe main.py update [source] --repo owner/name
.\.venv\Scripts\python.exe main.py status --repo owner/name
.\.venv\Scripts\python.exe main.py repos
.\.venv\Scripts\python.exe main.py remove owner/name
.\.venv\Scripts\python.exe main.py search "query" --repo owner/name -k 6
.\.venv\Scripts\python.exe main.py ask "question" --repo owner/name
.\.venv\Scripts\python.exe main.py index-history --repo owner/name --commits-only
.\.venv\Scripts\python.exe main.py index-history --repo owner/name --unshallow
.\.venv\Scripts\python.exe main.py serve --host 127.0.0.1 --port 8000
.\.venv\Scripts\python.exe evaluate.py --with-rewrite --details
.\.venv\Scripts\python.exe evaluate.py --questions eval/history_questions.jsonl --details
.\.venv\Scripts\python.exe check_labels.py
.\.venv\Scripts\python.exe judge.py --rewrite --limit 3
.\.venv\Scripts\python.exe judge.py --no-cache
.\.venv\Scripts\python.exe -m pytest -q
docker compose up --build
```

Search/ask optional flags: `--bm25`, `--rerank`, `--include-tests`, `--rewrite`,
`--since YYYY-MM-DD`, `--author NAME`, `--sources code|history|all`. The default
repo is psf/requests. `serve --host` defaults to config.API_HOST. Use one server
worker; stop it before any CLI command that opens local Qdrant. Answers, rewrites
and judge runs call Groq; avoid them during uncredentialed verification.

Docker installs CPU torch and caches both configured models at build time outside
the data mount. Compose injects the user's environment file without copying it
into the image. CI runs offline unit tests with dotenv disabled on Python 3.12.
The header Add repository button opens a native dialog; preserve indexing IDs,
source routing, polling and safe DOM-only rendering when editing the UI.
`eval/judge_results.md` is currently absent: never fabricate answer-quality scores.
The README screenshot at docs/screenshot.png is explicitly a placeholder.

## Resilience and doctor

All owned Qdrant connections must use gitcontext.database.open_client(); injected
clients stay shared. Local locks become DatabaseLocked with stop-server guidance;
evaluation scripts use run_cli to suppress only this known failure. QDRANT_URL
selects a server; unset it for default local storage. Docker's optional qdrant
profile provides a separate persistent server volume; reindex to migrate data.

`.venv\Scripts\python.exe main.py doctor --offline` checks packages, Python,
key presence (never its value), database and repo freshness without Groq. Doctor
without --offline makes a tiny Groq request and requires authorization during
coding. All Groq calls reuse answer.SESSION and share bounded retries for 429,
transport failures and temporary HTTP server errors. Streams never replay tokens
already emitted. Save judge answers and judgments immediately on completion.
Use requirements-lock.txt to reproduce pinned installed versions; Python 3.12 is
the tested CI version. Never run freeze output directly into public logs if it
contains editable/local references; filter them from the lock file.
