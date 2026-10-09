"""Local embeddings (sentence-transformers) + local Qdrant vector store."""
from __future__ import annotations

from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from . import config
from .chunker import Chunk


class Embedder:
    def __init__(self):
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

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]], batch: int = 128) -> None:
        for i in range(0, len(chunks), batch):
            points = [
                PointStruct(id=c.id, vector=v, payload=c.payload())
                for c, v in zip(chunks[i:i + batch], vectors[i:i + batch])
            ]
            self.client.upsert(config.COLLECTION, points=points)

    def search(self, vector: list[float], k: int = 6) -> list[tuple[str, float, dict]]:
        """Returns (id, cosine score, payload) for the k nearest chunks."""
        result = self.client.query_points(
            config.COLLECTION, query=vector, limit=k, with_payload=True
        )
        return [(str(p.id), p.score, p.payload) for p in result.points]

    def all_payloads(self) -> list[tuple[str, dict]]:
        """Every stored chunk. Used to build the BM25 keyword index."""
        out, offset = [], None
        while True:
            points, offset = self.client.scroll(
                config.COLLECTION, limit=512, offset=offset,
                with_payload=True, with_vectors=False,
            )
            out.extend((str(p.id), p.payload) for p in points)
            if offset is None:
                return out