from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol
from sentence_transformers import SentenceTransformer
import numpy as np


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return vectors / norms


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return [N, D] float32 vectors."""


@dataclass
class DeterministicHashEmbedder:
    """Dependency free local embedder for deterministic tests and demos."""

    dimension: int = 384

    def embed(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in text.lower().split():
                digest = hashlib.md5(token.encode("utf-8")).hexdigest()
                idx = int(digest[:8], 16) % self.dimension
                sign = -1.0 if int(digest[-1], 16) % 2 else 1.0
                matrix[i, idx] += sign
        return l2_normalize(matrix)


class SentenceTransformerEmbedder:
    """Optional sentence-transformers adapter for local semantic embeddings."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(texts, convert_to_numpy=True)
        return l2_normalize(vectors.astype(np.float32))


@dataclass
class MockEmbedding:
    values: list[float]


class MockTextEmbeddingModel:

    def __init__(self, dimension: int = 384) -> None:
        self.embedder = DeterministicHashEmbedder(dimension=dimension)

    def get_embeddings(self, texts: list[str]) -> list[MockEmbedding]:
        vectors = self.embedder.embed(texts)
        return [MockEmbedding(values=v.tolist()) for v in vectors]


class VertexLikeEmbedder:
    """Adapter around a Vertex-like model with get_embeddings()."""

    def __init__(self, model: MockTextEmbeddingModel) -> None:
        self.model = model

    def embed(self, texts: list[str]) -> np.ndarray:
        response = self.model.get_embeddings(texts)
        matrix = np.array([item.values for item in response], dtype=np.float32)
        return l2_normalize(matrix)
