from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .evaluation import average_similarity, mrr, recall_at_k
from .retriever import LocalRAGEngine


@dataclass
class BenchmarkQuery:
    query: str
    relevant_chunk_ids: set[str]


class BenchmarkRunner:
    def __init__(self, engine: LocalRAGEngine) -> None:
        self.engine = engine

    def run(self, queries: list[BenchmarkQuery]) -> dict:
        comparison_rows = []
        for case in queries:
            a_out = self.engine.run_strategy(case.query, strategy="A")
            b_out = self.engine.run_strategy(case.query, strategy="B")

            metrics_a = {
                "recall_at_k": recall_at_k(
                    a_out.results, case.relevant_chunk_ids, self.engine.config.top_k
                ),
                "mrr": mrr(a_out.results, case.relevant_chunk_ids),
                "avg_similarity": average_similarity(a_out.results),
            }
            metrics_b = {
                "recall_at_k": recall_at_k(
                    b_out.results, case.relevant_chunk_ids, self.engine.config.top_k
                ),
                "mrr": mrr(b_out.results, case.relevant_chunk_ids),
                "avg_similarity": average_similarity(b_out.results),
            }

            comparison_rows.append(
                {
                    "original_query": case.query,
                    "rewritten_query": b_out.rewritten_query,
                    "strategy_a": self._serialize_results(a_out.results),
                    "strategy_b": self._serialize_results(b_out.results),
                    "similarity_scores": {
                        "a": [round(item.score, 4) for item in a_out.results],
                        "b": [round(item.score, 4) for item in b_out.results],
                    },
                    "metrics": {
                        "a": metrics_a,
                        "b": metrics_b,
                        "delta_b_minus_a": {
                            "recall_at_k": metrics_b["recall_at_k"] - metrics_a["recall_at_k"],
                            "mrr": metrics_b["mrr"] - metrics_a["mrr"],
                            "avg_similarity": metrics_b["avg_similarity"]
                            - metrics_a["avg_similarity"],
                        },
                    },
                    "relevance_observations": self._observe(metrics_a, metrics_b),
                    "retrieval_reasoning": {
                        "a": [item.why_retrieved for item in a_out.results],
                        "b": [item.why_retrieved for item in b_out.results],
                    },
                    "failure_mode_flags": self._failure_modes(case.query, b_out.rewritten_query or ""),
                }
            )

        return {"comparison": comparison_rows}

    @staticmethod
    def _serialize_results(results: list) -> list[dict]:
        serialized = []
        for item in results:
            serialized.append(
                {
                    "rank": item.rank,
                    "chunk_id": item.chunk_id,
                    "source_id": item.source_id,
                    "score": round(item.score, 4),
                    "text": item.text,
                    "why_retrieved": item.why_retrieved,
                }
            )
        return serialized

    @staticmethod
    def _observe(metrics_a: dict, metrics_b: dict) -> str:
        if metrics_b["mrr"] > metrics_a["mrr"]:
            return "Strategy B ranked a relevant chunk earlier."
        if metrics_b["recall_at_k"] > metrics_a["recall_at_k"]:
            return "Strategy B improved top-k recall."
        if metrics_b["avg_similarity"] > metrics_a["avg_similarity"]:
            return "Strategy B increased average semantic similarity."
        return "No measurable gain from query expansion in this query."

    @staticmethod
    def _failure_modes(original_query: str, rewritten_query: str) -> dict[str, bool]:
        extra_terms = set(rewritten_query.lower().split()) - set(original_query.lower().split())
        return {
            "semantic_drift": len(extra_terms) > 25,
            "over_expansion": len(extra_terms) > 15,
            "retrieval_dilution": any(word in rewritten_query.lower() for word in ["everything", "all aspects"]),
            "false_positives": False,
        }

    @staticmethod
    def write_outputs(report: dict, json_path: Path, markdown_path: Path) -> None:
        json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        markdown_path.write_text(_to_markdown(report), encoding="utf-8")


def _to_markdown(report: dict) -> str:
    lines = [
        "# Retrieval Benchmark",
        "",
        "Comparison of Strategy A (raw vector search) vs Strategy B (query expansion + vector search).",
        "",
        "## Similarity Metric Choice",
        "Cosine similarity was selected because embedding direction better represents semantic meaning than magnitude in typical transformer embeddings.",
        "With FAISS this is implemented as IndexFlatIP over normalized vectors and normalized query vectors.",
        "",
        "## Production Migration Path",
        "- sentence-transformers -> Vertex AI Embeddings",
        "- FAISS -> Vertex AI Matching Engine",
        "- Mock query expansion -> Gemini/Vertex AI GenerativeModel",
        "",
        "## Query Comparison",
        "",
    ]
    for i, row in enumerate(report["comparison"], start=1):
        lines.extend(
            [
                f"### Query {i}",
                f"- Original Query: {row['original_query']}",
                f"- Rewritten Query: {row['rewritten_query']}",
                f"- Observation: {row['relevance_observations']}",
                f"- Failure Modes: {row['failure_mode_flags']}",
                "",
                "#### Strategy A Top Results",
            ]
        )
        for item in row["strategy_a"]:
            lines.append(
                f"- Rank {item['rank']} | {item['chunk_id']} | score={item['score']} | why={item['why_retrieved']}"
            )
        lines.append("")
        lines.append("#### Strategy B Top Results")
        for item in row["strategy_b"]:
            lines.append(
                f"- Rank {item['rank']} | {item['chunk_id']} | score={item['score']} | why={item['why_retrieved']}"
            )
        lines.append("")
        lines.append("#### Metrics")
        lines.append(f"- A: {row['metrics']['a']}")
        lines.append(f"- B: {row['metrics']['b']}")
        lines.append(f"- Delta (B-A): {row['metrics']['delta_b_minus_a']}")
        lines.append("")
    lines.extend(
        [
            "## Query Expansion Tradeoffs",
            "- Semantic drift: rewrite can alter user intent.",
            "- Over-expansion: too many concepts can broaden retrieval.",
            "- Retrieval dilution: expanded terms can reduce precision.",
            "- False positives: broad terms can retrieve tangential chunks.",
            "",
            "## Fairness Rule",
            "Both strategies use identical chunking, embeddings, index, scoring, and top-k. Only query transformation differs.",
            "",
        ]
    )
    return "\n".join(lines)
