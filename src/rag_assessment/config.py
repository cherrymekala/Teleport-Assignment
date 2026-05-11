from dataclasses import dataclass
from typing import Literal


SimilarityMetric = Literal["cosine", "euclidean"]


@dataclass
class RetrievalConfig:
    """Central retrieval configuration used across ingestion and search."""

    top_k: int = 3
    chunk_size: int = 70
    chunk_overlap: int = 20
    similarity_metric: SimilarityMetric = "cosine"
