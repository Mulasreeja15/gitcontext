"""Local embeddings (sentence-transformers) + local Qdrant vector store."""
from __future__ import annotations

from pathlib import Path
import warnings

from qdrant_client import QdrantClient
from qdrant_client.models import (Distance, PointStruct, VectorParams, Filter,
                                 FieldCondition, MatchValue, FilterSelector, PayloadSchemaType, DatetimeRange)

from . import config
from .chunker import Chunk


class Embedder:
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        print(f"Loading embedding model {config.EMBED_MODEL} ...")
        self.model = SentenceTransformer(config.EMBED_MODEL)
        self.model.max_seq_length = config.EMBED_MAX_TOKENS
        get_dim = (getattr(self.model, "get_embedding_dimension", None)
                   or self.model.get_sentence_embedding_dimension)
        self.dim = get_dim()

    def encode(self, texts: list[str], batch_size: int = 16) -> list[list[float]]:
        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 32,
        )
        return vectors.tolist()


class VectorStore:
    def __init__(self):
        Path(config.QDRANT_PATH).mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=config.QDRANT_PATH)

    def reset(self, dim: int) -> None:
        if self.client.collection_exists(config.COLLECTION):
            self.client.delete_collection(config.COLLECTION)
        self.client.create_collection(
            config.COLLECTION,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        self.ensure_file_index()

    def ensure_file_index(self) -> None:
        """Request a keyword index where supported; local Qdrant scans filters."""
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='Payload indexes have no effect.*')
                for field, schema in [('file_path', PayloadSchemaType.KEYWORD),
                                      ('last_author', PayloadSchemaType.KEYWORD),
                                      ('last_modified', PayloadSchemaType.DATETIME)]:
                    self.client.create_payload_index(config.COLLECTION, field, schema)
        except NotImplementedError:
            pass

    def delete_by_file(self, file_path: str) -> int:
        """Delete all chunks for an exact repository-relative path; return count."""
        if not self.client.collection_exists(config.COLLECTION):
            return 0
        selector = Filter(must=[FieldCondition(key='file_path', match=MatchValue(value=file_path))])
        removed = self.client.count(config.COLLECTION, count_filter=selector, exact=True).count
        self.client.delete(config.COLLECTION, points_selector=FilterSelector(filter=selector))
        return removed

    def count(self) -> int:
        """Return the indexed chunk count, including zero before first indexing."""
        if not self.client.collection_exists(config.COLLECTION):
            return 0
        return self.client.count(config.COLLECTION, exact=True).count

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]], batch: int = 128) -> None:
        for i in range(0, len(chunks), batch):
            points = [
                PointStruct(id=c.id, vector=v, payload=c.payload())
                for c, v in zip(chunks[i:i + batch], vectors[i:i + batch])
            ]
            self.client.upsert(config.COLLECTION, points=points)

    @staticmethod
    def metadata_filter(since: str | None = None, author: str | None = None):
        """Match exact author names and an inclusive UTC date lower bound."""
        from datetime import date
        conditions = []
        if since:
            parsed = date.fromisoformat(since)
            if parsed.isoformat() != since:
                raise ValueError('since must use YYYY-MM-DD')
            conditions.append(FieldCondition(key='last_modified', range=DatetimeRange(gte=since + 'T00:00:00Z')))
        if author:
            conditions.append(FieldCondition(key='last_author', match=MatchValue(value=author)))
        return Filter(must=conditions) if conditions else None

    def search(self, vector: list[float], k: int = 6, *, since=None, author=None) -> list[tuple[str, float, dict]]:
        """Returns (id, cosine score, payload) for the k nearest chunks."""
        result = self.client.query_points(
            config.COLLECTION, query=vector, limit=k, with_payload=True,
            query_filter=self.metadata_filter(since, author)
        )
        return [(str(p.id), p.score, p.payload) for p in result.points]

    def all_payloads(self, *, since=None, author=None) -> list[tuple[str, dict]]:
        """Every stored chunk. Used to build the BM25 keyword index."""
        out, offset = [], None
        while True:
            points, offset = self.client.scroll(
                config.COLLECTION, limit=512, offset=offset,
                with_payload=True, with_vectors=False,
                scroll_filter=self.metadata_filter(since, author),
            )
            out.extend((str(p.id), p.payload) for p in points)
            if offset is None:
                return out
