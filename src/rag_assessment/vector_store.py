from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
import faiss
import numpy as np

from .embeddings import l2_normalize


class VectorStore(Protocol):
    def add(self, vectors: np.ndarray) -> None:
        """Add [N, D] vectors to the index."""

    def search(self, query_vector: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return scores and indices."""


@dataclass
class NumpyIPVectorStore:
    """Inner-product retrieval over normalized vectors (cosine equivalent)."""

    vectors: np.ndarray | None = None

    def add(self, vectors: np.ndarray) -> None:
        normalized = l2_normalize(vectors.astype(np.float32))
        self.vectors = normalized

    def search(self, query_vector: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        if self.vectors is None:
            raise RuntimeError("Vector store is empty")
        q = l2_normalize(query_vector.astype(np.float32))
        scores = np.dot(self.vectors, q[0])
        indices = np.argsort(-scores)[:top_k]
        return scores[indices], indices


class FaissIPVectorStore:
    """FAISS IndexFlatIP with explicit L2 normalization for cosine similarity."""

    def __init__(self, dimension: int) -> None:
        self._faiss = faiss
        self._index = faiss.IndexFlatIP(dimension)

    def add(self, vectors: np.ndarray) -> None:
        normalized = l2_normalize(vectors.astype(np.float32))
        self._index.add(normalized)

    def search(self, query_vector: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        q = l2_normalize(query_vector.astype(np.float32))
        scores, indices = self._index.search(q, top_k)
        return scores[0], indices[0]
