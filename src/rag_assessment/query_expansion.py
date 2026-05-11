from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol


class QueryExpander(Protocol):
    def rewrite(self, query: str) -> str:
        """Rewrite the query to improve retrieval quality."""


@dataclass
class MockGenerationResponse:
    text: str


class MockGenerativeModel:

    def generate_content(self, prompt: str) -> MockGenerationResponse:
        return MockGenerationResponse(text=prompt)


class IntentAwareQueryExpander:
    """Rule-based, intention-oriented query expansion for benchmarking."""

    _intent_terms: dict[str, list[str]] = {
        "peak_load": [
            "autoscaling",
            "concurrency",
            "load balancing",
            "throughput",
            "failover",
            "retries",
        ],
        "reliability": [
            "high availability",
            "retry policy",
            "fault tolerance",
            "circuit breaker",
            "graceful degradation",
        ],
        "latency": [
            "p95 latency",
            "tail latency",
            "caching",
            "cold starts",
            "connection pooling",
            "response time",
        ],
    }
    intent_priority = ["latency", "reliability", "peak_load"]

    def rewrite(self, query: str) -> str:
        lower_q = query.lower()
        intent = self._detect_intent(lower_q)
        concepts = self._intent_terms[intent]
        concept_str = ", ".join(concepts)
        return (
            f"{query}. Context focus: {intent.replace('_', ' ')}. "
            f"Related concepts: {concept_str}."
        )

    @staticmethod
    def _detect_intent(query: str) -> str:
        tokens = set(re.findall(r"[a-z0-9]+", query.lower()))

        scores = {
            "latency": 0,
            "reliability": 0,
            "peak_load": 0,
        }

        for term in ["latency", "slow", "response", "time", "p95", "p99"]:
            if term in tokens:
                scores["latency"] += 1
        for term in ["reliability", "resilience", "failure", "failures", "fault"]:
            if term in tokens:
                scores["reliability"] += 1
        for term in ["peak", "load", "scale", "scaling", "traffic", "throughput", "concurrency"]:
            if term in tokens:
                scores["peak_load"] += 1

        best_score = max(scores.values())
        if best_score == 0:
            return "peak_load"

        for intent in IntentAwareQueryExpander.intent_priority:
            if scores[intent] == best_score:
                return intent
        return "peak_load"


class VertexLikeQueryExpander:
    """Adapter around a Vertex-like GenerativeModel interface."""

    def __init__(self, model: MockGenerativeModel) -> None:
        self.model = model

    def rewrite(self, query: str) -> str:
        prompt = (
            "Rewrite this query for semantic retrieval while preserving intent. "
            "Add relevant systems concepts where useful. Query: "
            f"{query}"
        )
        response = self.model.generate_content(prompt)
        return response.text
