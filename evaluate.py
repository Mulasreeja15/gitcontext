"""Retrieval evaluation: an ablation study over a labelled question set.

  python evaluate.py                  # compare all configurations
  python evaluate.py --with-rewrite   # also test LLM query rewriting (uses Groq)
  python evaluate.py --details        # print the rank of the first correct hit per question

Results are also written to eval/results.md.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from dataclasses import replace

from gitcontext.retriever import RetrievalOptions, Retriever

QUESTIONS = Path("eval/questions.jsonl")
RESULTS = Path("eval/results.md")
DEPTH = 10  # ranks checked per question (for MRR)

CONFIGS = [
    ("Vector only (Phase 1 baseline)", RetrievalOptions(bm25=False, rerank=False, penalize_tests=False)),
    ("Vector + test down-ranking", RetrievalOptions(bm25=False, rerank=False, penalize_tests=True)),
    ("Hybrid (vector + BM25, RRF)", RetrievalOptions(bm25=True, rerank=False, penalize_tests=False)),
    ("Hybrid + test down-ranking", RetrievalOptions(bm25=True, rerank=False, penalize_tests=True)),
    ("Hybrid + test down-ranking + rerank", RetrievalOptions(bm25=True, rerank=True, penalize_tests=True)),
]
REWRITE_CONFIGS = [
    ("Vector + test down-ranking + LLM rewrite", RetrievalOptions(bm25=False, rerank=False, penalize_tests=True, rewrite=True)),
    ("Hybrid + test down-ranking + LLM rewrite", RetrievalOptions(bm25=True, rerank=False, penalize_tests=True, rewrite=True)),
]


def load_questions(path: Path | None = None) -> list[dict]:
    rows = []
    for line in (path or QUESTIONS).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            rows.append(json.loads(line))
    return rows


def is_match(payload: dict, target: str) -> bool:
    """target is 'path' or 'path::Symbol'. A chunk matches if it's in that file
    and its name contains the symbol, or it contains the symbol's definition."""
    path, _, symbol = target.partition("::")
    if path == 'commit':
        return payload.get('kind') == 'commit' and bool(symbol) and (payload.get('sha') or '').startswith(symbol)
    if path == 'pr':
        return payload.get('kind') == 'pr' and str(payload.get('number')) == symbol
    if payload["file_path"] != path:
        return False
    if not symbol:
        return True
    if symbol in payload["name"]:
        return True
    return f"def {symbol.split('.')[-1]}(" in payload["text"]


def first_hit_rank(hits, targets) -> int | None:
    for rank, (_, p) in enumerate(hits, 1):
        if any(is_match(p, t) for t in targets):
            return rank
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-rewrite", action="store_true")
    parser.add_argument("--details", action="store_true")
    parser.add_argument('--questions', type=Path, default=QUESTIONS)
    parser.add_argument('--sources', choices=['code', 'history', 'all'])
    args = parser.parse_args()

    questions = load_questions(args.questions)
    if not questions:
        parser.error('Question file is empty.')
    sources = args.sources or ('history' if all(t.startswith(('commit::', 'pr::'))
                        for q in questions for t in q['expected']) else 'code')
    configs = CONFIGS + (REWRITE_CONFIGS if args.with_rewrite else [])
    configs = [(name, replace(opts, sources=sources)) for name, opts in configs]
    n = len(questions)

    retriever = Retriever()
    retriever.prepare(RetrievalOptions(rerank=True, sources=sources))
    retriever.retrieve("warm up", 5, RetrievalOptions(rerank=True, sources=sources))

    summary, ranks_by_config = [], {}
    for name, opts in configs:
        print(f"Running: {name} ...")
        ranks, latencies = [], []
        for q in questions:
            result = retriever.retrieve(q["question"], DEPTH, opts)
            ranks.append(first_hit_rank(result.hits, q["expected"]))
            latencies.append(result.timings["total_ms"])

        def recall(k):
            return sum(1 for r in ranks if r and r <= k) / n

        mrr = sum(1 / r for r in ranks if r) / n
        summary.append((name, recall(1), recall(3), recall(5), mrr, statistics.median(latencies)))
        ranks_by_config[name] = ranks

    table = [
        "| Configuration | Recall@1 | Recall@3 | Recall@5 | MRR@10 | Median latency |",
        "|---|---|---|---|---|---|",
    ]
    for name, r1, r3, r5, mrr, lat in summary:
        table.append(f"| {name} | {r1:.0%} | {r3:.0%} | {r5:.0%} | {mrr:.3f} | {lat:.0f} ms |")

    detail = [
        "| Question | " + " | ".join(name for name, _ in configs) + " |",
        "|---|" + "---|" * len(configs),
    ]
    for i, q in enumerate(questions):
        cells = [str(ranks_by_config[name][i] or "-") for name, _ in configs]
        detail.append(f"| {q['question']} | " + " | ".join(cells) + " |")

    print("\n" + "\n".join(table))
    if args.details:
        print("\nRank of first correct result ('-' = not in top 10):\n")
        print("\n".join(detail))

    output = RESULTS if args.questions == QUESTIONS else args.questions.with_name(args.questions.stem + '_results.md')
    output.write_text(
        "# Retrieval evaluation\n\n"
        f"Repo: psf/requests, {n} labelled questions, sources: {sources}, ranks checked up to {DEPTH}.\n\n"
        + "\n".join(table)
        + "\n\n## Rank of first correct result per question\n\n"
        + "\n".join(detail) + "\n",
        encoding="utf-8",
    )
    print(f"\nSaved to {output}")


if __name__ == "__main__":
    from gitcontext.database import run_cli
    run_cli(main)
