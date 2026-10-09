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
        self.model = SentenceTransformer(config.EMBED_MODEL, trust_remote_code=True)
        self.model.max_seq_length = config.EMBED_MAX_TOKENS
        self.dim = self.model.get_embedding_dimension()

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
        """Phase 1 does a full rebuild each time. Phase 4 makes this incremental."""
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

    def search(self, vector: list[float], k: int = 6) -> list[tuple[float, dict]]:
        result = self.client.query_points(
            config.COLLECTION, query=vector, limit=k, with_payload=True
        )
        return [(p.score, p.payload) for p in result.points]