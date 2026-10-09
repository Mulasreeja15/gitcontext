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

from gitcontext.retriever import RetrievalOptions, Retriever

QUESTIONS = Path("eval/questions.jsonl")
RESULTS = Path("eval/results.md")
DEPTH = 10  # ranks checked per question (for MRR)

CONFIGS = [
    ("Vector only (Phase 1 baseline)", RetrievalOptions(bm25=False, rerank=False, penalize_tests=False)),
    ("+ BM25 hybrid (RRF)", RetrievalOptions(bm25=True, rerank=False, penalize_tests=False)),
    ("+ Test down-ranking", RetrievalOptions(bm25=True, rerank=False, penalize_tests=True)),
    ("+ Cross-encoder rerank", RetrievalOptions(bm25=True, rerank=True, penalize_tests=True)),
]
REWRITE_CONFIG = ("+ LLM query rewriting", RetrievalOptions(rewrite=True))


def load_questions() -> list[dict]:
    rows = []
    for line in QUESTIONS.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            rows.append(json.loads(line))
    return rows


def is_match(payload: dict, target: str) -> bool:
    """target is 'path' or 'path::Symbol'. A chunk matches if it's in that file
    and its name contains the symbol, or it contains the symbol's definition."""
    path, _, symbol = target.partition("::")
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
    args = parser.parse_args()

    questions = load_questions()
    configs = CONFIGS + ([REWRITE_CONFIG] if args.with_rewrite else [])
    n = len(questions)

    retriever = Retriever()
    retriever.prepare(RetrievalOptions(rerank=True))    # load BM25 + reranker up front
    retriever.retrieve("warm up", 5, RetrievalOptions(rerank=True))

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

    RESULTS.write_text(
        "# Retrieval evaluation\n\n"
        f"Repo: psf/requests, {n} labelled questions, ranks checked up to {DEPTH}.\n\n"
        + "\n".join(table)
        + "\n\n## Rank of first correct result per question\n\n"
        + "\n".join(detail) + "\n",
        encoding="utf-8",
    )
    print(f"\nSaved to {RESULTS}")


if __name__ == "__main__":
    main()