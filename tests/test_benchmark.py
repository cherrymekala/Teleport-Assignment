from src.rag_assessment.benchmark import BenchmarkQuery, BenchmarkRunner
from src.rag_assessment.config import RetrievalConfig
from src.rag_assessment.embeddings import DeterministicHashEmbedder
from src.rag_assessment.query_expansion import IntentAwareQueryExpander
from src.rag_assessment.retriever import LocalRAGEngine


def _build_engine() -> LocalRAGEngine:
    config = RetrievalConfig(top_k=3, chunk_size=25, chunk_overlap=5, similarity_metric="cosine")
    engine = LocalRAGEngine(
        config=config,
        embedder=DeterministicHashEmbedder(dimension=96),
        query_expander=IntentAwareQueryExpander(),
        prefer_faiss=False,
    )
    engine.ingest(
        [
            "Autoscaling and load balancing handle traffic spikes.",
            "Retries, circuit breakers and failover improve resilience.",
            "Caching and pooling reduce API latency.",
        ]
    )
    return engine


def test_benchmark_contains_expected_fields() -> None:
    engine = _build_engine()
    runner = BenchmarkRunner(engine)
    report = runner.run(
        [
            BenchmarkQuery(
                query="How does the system handle peak load?",
                relevant_chunk_ids={"p0-c0"},
            )
        ]
    )

    row = report["comparison"][0]
    assert "rewritten_query" in row
    assert "similarity_scores" in row
    assert "relevance_observations" in row
    assert "retrieval_reasoning" in row
    assert "failure_mode_flags" in row
    assert "metrics" in row
