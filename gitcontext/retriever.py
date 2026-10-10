from __future__ import annotations

import re
import time
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from rank_bm25 import BM25Okapi

from . import config
from .store import Embedder, VectorStore

_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CAMEL = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")
_STOPWORDS = {
    "the", "a", "an", "is", "are", "how", "does", "do", "what", "where", "which",
    "when", "why", "in", "of", "to", "for", "and", "or", "it", "this", "that",
    "be", "with", "on", "by", "self",
}


def tokenize(text: str) -> list[str]:
    """Code-aware tokenizer. Keeps whole identifiers AND splits snake_case /
    camelCase, so 'resolve_redirects' also matches 'redirects' and
    'HTTPAdapter' also matches 'adapter'."""
    tokens = []
    for word in _WORD.findall(text):
        low = word.lower()
        if low in _STOPWORDS:
            continue
        tokens.append(low)
        parts = [p.lower() for piece in word.split("_") if piece for p in _CAMEL.findall(piece)]
        if len(parts) > 1:
            tokens.extend(p for p in parts if p not in _STOPWORDS)
    return tokens


def is_test(payload: dict) -> bool:
    parts = payload["file_path"].lower().split("/")
    name = parts[-1]
    return (
        any(p in {"test", "tests", "testing"} for p in parts[:-1])
        or name.startswith("test_")
        or name.endswith("_test.py")
        or name == "conftest.py"
    )


def _passage(p: dict) -> str:
    """What the reranker reads for each candidate."""
    return f"{p['file_path']} | {p['kind']} {p['name']}\n{p['text'][:2000]}"


@dataclass
class RetrievalOptions:
    bm25: bool = False
    rerank: bool = False
    penalize_tests: bool = True
    rewrite: bool = False

    def describe(self) -> str:
        parts = ["vector"]
        if self.bm25:
            parts.append("BM25 (RRF fusion)")
        if self.penalize_tests:
            parts.append("test down-ranking")
        if self.rerank:
            parts.append("cross-encoder rerank")
        if self.rewrite:
            parts.append("query rewriting")
        return " + ".join(parts)


@dataclass
class RetrievalResult:
    hits: list[tuple[float, dict]]
    timings: dict[str, float] = field(default_factory=dict)
    rewritten_query: str | None = None


class Retriever:
    def __init__(self):
        self.embedder = Embedder()
        self.store = VectorStore()
        self._payloads: dict[str, dict] = {}
        self._bm25 = None
        self._bm25_ids: list[str] = []
        self._reranker = None

    def invalidate_index(self) -> None:
        """Discard payload/BM25 caches after reindexing while retaining ML models."""
        self._payloads.clear()
        self._bm25 = None
        self._bm25_ids.clear()

    # ---------- lazy setup ----------

    def prepare(self, opts: RetrievalOptions) -> None:
        """Load what these options need, so it isn't counted as query latency."""
        if opts.bm25:
            self._ensure_bm25()
        if opts.rerank:
            self._ensure_reranker()

    def _ensure_bm25(self) -> None:
        if self._bm25 is not None:
            return
        docs = self.store.all_payloads()
        self._bm25_ids = [pid for pid, _ in docs]
        corpus = []
        for pid, p in docs:
            self._payloads[pid] = p
            corpus.append(tokenize(f"{p['file_path']} {p['name']} {p.get('parent', '')} {p['text']}"))
        self._bm25 = BM25Okapi(corpus)

    def _ensure_reranker(self) -> None:
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            print(f"Loading reranker {config.RERANK_MODEL} ...")
            self._reranker = CrossEncoder(config.RERANK_MODEL, max_length=config.RERANK_MAX_TOKENS)

    # ---------- individual retrievers ----------

    def _vector_ranking(self, query: str) -> list[str]:
        hits = self.store.search(self.embedder.encode([query])[0], config.CANDIDATES)
        for pid, _, p in hits:
            self._payloads[pid] = p
        return [pid for pid, _, _ in hits]

    def _bm25_ranking(self, query: str) -> list[str]:
        self._ensure_bm25()
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        top = np.argsort(scores)[::-1][:config.CANDIDATES]
        return [self._bm25_ids[i] for i in top if scores[i] > 0]

    # ---------- full pipeline ----------

    def retrieve(self, query: str, k: int = 6, opts: RetrievalOptions | None = None) -> RetrievalResult:
        opts = opts or RetrievalOptions()
        timings: dict[str, float] = {}
        start = time.perf_counter()

        queries, rewritten = [query], None
        if opts.rewrite:
            from .answer import rewrite_query
            t = time.perf_counter()
            rewritten = rewrite_query(query)
            timings["rewrite_ms"] = (time.perf_counter() - t) * 1000
            if rewritten and rewritten.lower() != query.lower():
                queries.append(rewritten)

        # 1) gather rankings from every retriever for every query version
        t = time.perf_counter()
        rankings = []
        for q in queries:
            rankings.append(self._vector_ranking(q))
            if opts.bm25:
                rankings.append(self._bm25_ranking(q))

        # 2) Reciprocal Rank Fusion: score = sum of 1 / (RRF_K + rank)
        fused: dict[str, float] = defaultdict(float)
        for ranking in rankings:
            for rank, pid in enumerate(ranking):
                fused[pid] += 1.0 / (config.RRF_K + rank + 1)

        # 3) down-rank test files
        if opts.penalize_tests:
            for pid in fused:
                if is_test(self._payloads[pid]):
                    fused[pid] *= config.TEST_RRF_WEIGHT

        candidates = sorted(fused, key=fused.get, reverse=True)[:config.CANDIDATES]
        timings["search_ms"] = (time.perf_counter() - t) * 1000

        # 4) cross-encoder reranking
        scores: dict[str, float] = dict(fused)
        ordered = candidates
        if opts.rerank and candidates:
            t = time.perf_counter()
            self._ensure_reranker()
            pairs = [(query, _passage(self._payloads[pid])) for pid in candidates]
            raw = self._reranker.predict(pairs, batch_size=16, show_progress_bar=False)
            scores = {}
            for pid, s in zip(candidates, raw):
                s = float(s)
                if opts.penalize_tests and is_test(self._payloads[pid]):
                    s -= config.TEST_RERANK_PENALTY
                scores[pid] = s
            ordered = sorted(scores, key=scores.get, reverse=True)
            timings["rerank_ms"] = (time.perf_counter() - t) * 1000

        timings["total_ms"] = (time.perf_counter() - start) * 1000
        hits = [(scores[pid], self._payloads[pid]) for pid in ordered[:k]]
        return RetrievalResult(hits, timings, rewritten)
