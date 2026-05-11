from src.rag_assessment.config import RetrievalConfig
from src.rag_assessment.embeddings import DeterministicHashEmbedder
from src.rag_assessment.query_expansion import IntentAwareQueryExpander
from src.rag_assessment.retriever import LocalRAGEngine


def test_retrieval_ranking_returns_top_k_and_reasons() -> None:
    config = RetrievalConfig(top_k=2, chunk_size=20, chunk_overlap=5, similarity_metric="cosine")
    engine = LocalRAGEngine(
        config=config,
        embedder=DeterministicHashEmbedder(dimension=64),
        query_expander=IntentAwareQueryExpander(),
        prefer_faiss=False,
    )
    paragraphs = [
        "Autoscaling and load balancing improve throughput under peak traffic.",
        "Retries and failover policies improve reliability.",
    ]
    engine.ingest(paragraphs)
    output = engine.run_strategy("How to handle peak traffic?", strategy="A")

    assert len(output.results) == 2
    assert output.results[0].rank == 1
    assert output.results[0].why_retrieved
