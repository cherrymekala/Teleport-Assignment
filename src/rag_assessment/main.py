from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.rag_assessment.benchmark import BenchmarkQuery, BenchmarkRunner
from src.rag_assessment.config import RetrievalConfig
from src.rag_assessment.embeddings import MockTextEmbeddingModel, VertexLikeEmbedder
from src.rag_assessment.query_expansion import IntentAwareQueryExpander
from src.rag_assessment.retriever import LocalRAGEngine


def load_paragraphs(path: Path) -> list[str]:
    raw = path.read_text(encoding="utf-8")
    return [part.strip() for part in raw.split("\n\n") if part.strip()]


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    config = RetrievalConfig(
        top_k=3,
        chunk_size=65,
        chunk_overlap=20,
        similarity_metric="cosine",
    )
    model = MockTextEmbeddingModel(dimension=384)
    embedder = VertexLikeEmbedder(model)
    expander = IntentAwareQueryExpander()
    engine = LocalRAGEngine(config=config, embedder=embedder, query_expander=expander)

    paragraphs = load_paragraphs(root / "data" / "technical_paragraphs.txt")
    engine.ingest(paragraphs)

    # Relevant chunk ids are deterministic for the bundled sample dataset.
    benchmark_queries = [
        BenchmarkQuery(
            query="How does the system handle peak load?",
            relevant_chunk_ids={"p0-c0", "p4-c0"},
        ),
        BenchmarkQuery(
            query="What mechanisms improve reliability during failures?",
            relevant_chunk_ids={"p1-c0", "p3-c0"},
        ),
        BenchmarkQuery(
            query="How is API latency reduced at high traffic?",
            relevant_chunk_ids={"p2-c0", "p4-c0"},
        ),
    ]

    runner = BenchmarkRunner(engine)
    report = runner.run(benchmark_queries)
    BenchmarkRunner.write_outputs(
        report,
        json_path=root / "benchmark_output.json",
        markdown_path=root / "retrieval_benchmark.md",
    )
    print("Generated benchmark_output.json and retrieval_benchmark.md")


if __name__ == "__main__":
    main()
