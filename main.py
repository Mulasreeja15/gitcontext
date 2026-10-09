"""GitContext CLI (Phase 1).

Usage:
  python main.py index <repo_path_or_git_url>
  python main.py search "query"
  python main.py ask "question"
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter

try:
    from gitcontext.answer import generate_answer
    from gitcontext.ingest import collect_chunks, resolve_repo
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


def retrieve(query: str, k: int):
    embedder = Embedder()
    store = VectorStore()
    t0 = time.perf_counter()  # measure retrieval only, not model loading
    hits = store.search(embedder.encode([query])[0], k)
    return hits, (time.perf_counter() - t0) * 1000


def print_hits(hits):
    for i, (score, p) in enumerate(hits, 1):
        print(
            f"[{i}] {score:.3f}  {p['file_path']}:{p['start_line']}-{p['end_line']}  "
            f"({p['kind']}: {p['name']})"
        )


def cmd_search(args):
    hits, ms = retrieve(args.query, args.k)
    print_hits(hits)
    print(f"\nRetrieval latency: {ms:.0f} ms")


def cmd_ask(args):
    hits, ms = retrieve(args.question, args.k)
    print("\nThinking ...\n")
    print(generate_answer(args.question, hits))
    print("\nSources:")
    print_hits(hits)
    print(f"\nRetrieval latency: {ms:.0f} ms")


def main():
    parser = argparse.ArgumentParser(description="GitContext: ask questions about a codebase")
    sub = parser.add_subparsers(required=True)

    p = sub.add_parser("index", help="index a repo (local path or Git URL)")
    p.add_argument("repo")
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("search", help="show retrieved chunks only")
    p.add_argument("query")
    p.add_argument("-k", type=int, default=6)
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("ask", help="answer a question with citations")
    p.add_argument("question")
    p.add_argument("-k", type=int, default=6)
    p.set_defaults(func=cmd_ask)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()