# GitContext: guidance for future coding tasks

## Project summary

GitContext is a retrieval-augmented generation (RAG) system for Python codebases. It parses code with tree-sitter into function, class, and method chunks with context headers containing file paths and symbol names. Small classes stay intact; large classes produce a summary plus method chunks. Oversized chunks are split into line windows. Module-level code and Markdown/RST/text documentation are also indexed.

Chunks are embedded locally with `BAAI/bge-small-en-v1.5` and stored in local Qdrant at `data/qdrant`, in the `gitcontext` collection. The default retrieval pipeline uses vector search and test-file down-ranking. Optional BM25 hybrid retrieval (`--bm25`) uses a code-aware tokenizer and Reciprocal Rank Fusion (RRF); optional cross-encoder reranking (`--rerank`) uses `cross-encoder/ms-marco-MiniLM-L-6-v2`. Optional LLM query rewriting (`--rewrite`) retrieves with both the original and rewritten query and fuses their rankings.

Groq serves `openai/gpt-oss-120b` for query rewriting and grounded answers. Answers use numbered source excerpts with file paths and line ranges, cite claims inline with `[n]`, and must acknowledge when the supplied sources do not contain the answer.

## Module map

| File | Responsibility |
|---|---|
| `gitcontext/__init__.py` | Package initializer. |
| `gitcontext/config.py` | Central settings: dotenv loading, models, local storage paths, retrieval tuning, chunk/file limits, supported extensions, and skipped directories. |
| `gitcontext/chunker.py` | `Chunk` metadata, deterministic UUIDs, embedding context headers, tree-sitter Python parsing, decorated definitions, large-class summaries and methods, line-window splitting, and document sections. |
| `gitcontext/ingest.py` | Resolve local repositories or clone Git URLs into `data/repos`; walk supported files, skip hidden/generated/oversized content, and collect chunks with repository-relative POSIX paths. |
| `gitcontext/store.py` | `Embedder` loads sentence-transformers and produces normalized embeddings; `VectorStore` resets/upserts the Qdrant collection, searches by cosine similarity, and scrolls payloads for BM25. |
| `gitcontext/retriever.py` | Retrieval options/results, identifier tokenization, test-file detection, lazy BM25/reranker setup, vector and keyword rankings, RRF, test penalties, query rewriting, reranking, and latency measurements. |
| `gitcontext/answer.py` | Groq `_chat()` client with 429 retries, source formatting, cited-answer prompts, and cached query rewriting. |
| `gitcontext/judge.py` | Answer-quality judge, JSON validation/retry, gold context, deterministic citation checks, resumable JSONL cache, and Markdown reports. |
| `gitcontext/api.py` | FastAPI lifespan holds one Retriever; validated health/search/ask routes and source-first SSE streaming. Retrieval is serialized to protect lazy initialization and local Qdrant. |
| `gitcontext/static/index.html` | Self-contained UI served at `/`: vanilla JavaScript, DOM-only Markdown rendering, SSE answers, search-only mode, source cards, and citation navigation. |
| `judge.py` | Answer-quality evaluation CLI: default retrieval, optional `--rewrite`, `--limit N`, and `--no-cache`; writes `eval/judge_results.md`. |
| `main.py` | CLI entry point for `index`, `search`, `ask`, and `serve --port 8000`; displays ranked sources and timings. Indexing replaces the existing collection. |
| `evaluate.py` | Ablation harness: evaluates first relevant ranks up to 10, Recall@1/3/5, MRR@10, and median latency; writes summary and per-question ranks to `eval/results.md`. |
| `check_labels.py` | Checks that every expected evaluation target matches a chunk in the current index. |
| `eval/questions.jsonl` | 45 labelled questions for `psf/requests`, with expected file or `path::Symbol` targets. |
| `README.md` | Architecture, usage, published evaluation results, limitations, and roadmap. |
| `notes.md` | Historical manual baseline observations and reranker experiments; their smaller-set figures are distinct from the README evaluation below. |

## Running on Windows

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

`index` also accepts a local repository directory and resets the collection, so index `psf/requests` before running its evaluation or label checks. `search` retrieves chunks; `ask` additionally calls Groq. `-k` controls the number of returned chunks (default 6). `--include-tests` disables test down-ranking. Rewriting and answering require credentials loaded by python-dotenv; never inspect the credential file or print keys.

## Published evaluation results

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
