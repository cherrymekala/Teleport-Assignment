from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .chunking import split_into_chunks
from .config import RetrievalConfig
from .embeddings import Embedder
from .models import RetrievalResult, StrategyOutput, TextChunk
from .query_expansion import QueryExpander
from .vector_store import FaissIPVectorStore, NumpyIPVectorStore, VectorStore


SearchStrategy = Literal["A", "B"]


@dataclass
class LocalRAGEngine:
    config: RetrievalConfig
    embedder: Embedder
    query_expander: QueryExpander
    prefer_faiss: bool = True

    def __post_init__(self) -> None:
        self.chunks: list[TextChunk] = []
        self.vectors: np.ndarray | None = None
        self.store: VectorStore | None = None

    def ingest(self, paragraphs: list[str]) -> None:
        self.chunks = split_into_chunks(
            paragraphs,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        texts = [chunk.text for chunk in self.chunks]
        self.vectors = self.embedder.embed(texts)
        self.store = self._build_store(self.vectors)
        self.store.add(self.vectors)

    def run_strategy(self, query: str, strategy: SearchStrategy) -> StrategyOutput:
        if self.store is None or self.vectors is None:
            raise RuntimeError("Call ingest() before retrieval")

        rewritten = None
        effective_query = query
        if strategy == "B":
            rewritten = self.query_expander.rewrite(query)
            effective_query = rewritten

        query_vector = self.embedder.embed([effective_query])
        scores, indices = self.store.search(query_vector, self.config.top_k)
        results = self._to_results(
            query=effective_query,
            scores=scores,
            indices=indices,
        )
        return StrategyOutput(
            original_query=query,
            rewritten_query=rewritten,
            results=results,
        )

    def _build_store(self, vectors: np.ndarray) -> VectorStore:
        if self.config.similarity_metric != "cosine":
            return NumpyIPVectorStore()

        if self.prefer_faiss:
            try:
                return FaissIPVectorStore(dimension=vectors.shape[1])
            except ImportError:
                return NumpyIPVectorStore()
        return NumpyIPVectorStore()

    def _to_results(
        self,
        query: str,
        scores: np.ndarray,
        indices: np.ndarray,
    ) -> list[RetrievalResult]:
        query_terms = self._tokenize(query)
        output: list[RetrievalResult] = []
        for rank, (score, idx) in enumerate(zip(scores, indices), start=1):
            chunk = self.chunks[int(idx)]
            reason = self._explain_retrieval(query_terms, chunk.text)
            output.append(
                RetrievalResult(
                    chunk_id=chunk.chunk_id,
                    source_id=chunk.source_id,
                    text=chunk.text,
                    score=float(score),
                    rank=rank,
                    why_retrieved=reason,
                )
            )
        return output

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", text.lower()))

    @classmethod
    def _explain_retrieval(cls, query_terms: set[str], chunk_text: str) -> str:
        chunk_terms = cls._tokenize(chunk_text)
        overlap = sorted(query_terms.intersection(chunk_terms))
        if overlap:
            preview = ", ".join(overlap[:5])
            return f"Term overlap with query: {preview}."
        return "Semantic similarity match from embedding space."
