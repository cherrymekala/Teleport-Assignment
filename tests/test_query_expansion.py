from src.rag_assessment.query_expansion import IntentAwareQueryExpander


def test_query_expansion_adds_semantic_concepts() -> None:
    expander = IntentAwareQueryExpander()
    rewritten = expander.rewrite("How does the system handle peak load?")
    assert "autoscaling" in rewritten.lower()
    assert "concurrency" in rewritten.lower()
    assert "load balancing" in rewritten.lower()
