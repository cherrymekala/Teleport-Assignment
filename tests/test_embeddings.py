import numpy as np

from src.rag_assessment.embeddings import DeterministicHashEmbedder, MockTextEmbeddingModel, VertexLikeEmbedder


def test_deterministic_embedder_shape_and_norm() -> None:
    embedder = DeterministicHashEmbedder(dimension=16)
    vectors = embedder.embed(["autoscaling load balancing", "retry failover"])
    assert vectors.shape == (2, 16)
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0)


def test_vertex_like_embedder_mock_model() -> None:
    model = MockTextEmbeddingModel(dimension=8)
    embedder = VertexLikeEmbedder(model)
    vectors = embedder.embed(["hello world"])
    assert vectors.shape == (1, 8)
