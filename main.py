"""GitContext CLI.

  python main.py index <repo_path_or_git_url>
  python main.py search "query" [options]
  python main.py ask "question" [options]

Retrieval options:
  -k              number of results (default 6)
  --bm25          add BM25 keyword search (hybrid)
  --rerank        use the cross-encoder reranker
  --include-tests  don't down-rank test files
  --rewrite        let the LLM rewrite the query first
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter

try:
    from gitcontext.answer import generate_answer
    from gitcontext.ingest import collect_chunks, resolve_repo
    from gitcontext.retriever import RetrievalOptions, Retriever, is_test
    from gitcontext.store import Embedder, VectorStore
except ImportError as e:
    sys.exit(
        f"Import failed: {e}\n"
        "Fix: make sure (.venv) is active, then run: pip install -r requirements.txt"
    )


def cmd_index(args):
    root = resolve_repo(args.repo)
    print(f"Scanning {root} ...")
    chunks = collect_chunks(root)
    if not chunks:
        sys.exit("No supported files found.")
    print(f"{len(chunks)} chunks: {dict(Counter(c.kind for c in chunks))}")

    embedder = Embedder()
    t0 = time.time()
    vectors = embedder.encode([c.embedding_text() for c in chunks])
    store = VectorStore()
    store.reset(embedder.dim)
    store.upsert(chunks, vectors)
    print(f"Done. Embedded + stored in {time.time() - t0:.1f}s")


def options_from(args) -> RetrievalOptions:
    return RetrievalOptions(
        bm25=args.bm25,
        rerank=args.rerank,
        penalize_tests=not args.include_tests,
        rewrite=args.rewrite,
    )


def run_retrieval(query: str, args):
    opts = options_from(args)
    retriever = Retriever()
    retriever.prepare(opts)
    return retriever.retrieve(query, args.k, opts), opts


def print_result(result, opts):
    print(f"\nMode: {opts.describe()}")
    if result.rewritten_query:
        print(f"Rewritten query: {result.rewritten_query}")
    for i, (score, p) in enumerate(result.hits, 1):
        tag = "  [test]" if is_test(p) else ""
        print(
            f"[{i}] {score:8.4f}  {p['file_path']}:{p['start_line']}-{p['end_line']}  "
            f"({p['kind']}: {p['name']}){tag}"
        )
    parts = [f"{name.replace('_ms', '')} {ms:.0f} ms" for name, ms in result.timings.items()]
    print("\nLatency: " + ", ".join(parts))


def cmd_search(args):
    result, opts = run_retrieval(args.query, args)
    print_result(result, opts)


def cmd_ask(args):
    result, opts = run_retrieval(args.question, args)
    print("\nThinking ...\n")
    print(generate_answer(args.question, result.hits))
    print("\nSources:")
    print_result(result, opts)


def add_retrieval_flags(p):
    """Register the shared search and ask retrieval options."""
    p.add_argument("-k", type=int, default=6, help="number of results")
    p.add_argument("--bm25", action="store_true", help="add BM25 keyword search (hybrid)")
    p.add_argument("--include-tests", action="store_true", help="don't down-rank tests")
    p.add_argument("--rerank", action="store_true", help="use the cross-encoder reranker")
    p.add_argument("--rewrite", action="store_true", help="let the LLM rewrite the query first")


def main():
    parser = argparse.ArgumentParser(description="GitContext: ask questions about a codebase")
    sub = parser.add_subparsers(required=True)

    p = sub.add_parser("index", help="index a repo (local path or Git URL)")
    p.add_argument("repo")
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("search", help="show retrieved chunks only")
    p.add_argument("query")
    add_retrieval_flags(p)
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("ask", help="answer a question with citations")
    p.add_argument("question")
    add_retrieval_flags(p)
    p.set_defaults(func=cmd_ask)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
