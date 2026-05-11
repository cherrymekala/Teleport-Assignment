from __future__ import annotations

from dataclasses import dataclass

from .models import RetrievalResult


@dataclass
class QueryMetrics:
    recall_at_k: float
    mrr: float
    avg_similarity: float


def recall_at_k(results: list[RetrievalResult], relevant_chunk_ids: set[str], k: int) -> float:
    if not relevant_chunk_ids:
        return 0.0
    top_results = results[:k]
    hits = sum(1 for item in top_results if item.chunk_id in relevant_chunk_ids)
    return hits / len(relevant_chunk_ids)


def mrr(results: list[RetrievalResult], relevant_chunk_ids: set[str]) -> float:
    for item in results:
        if item.chunk_id in relevant_chunk_ids:
            return 1.0 / item.rank
    return 0.0


def average_similarity(results: list[RetrievalResult]) -> float:
    if not results:
        return 0.0
    return sum(item.score for item in results) / len(results)
