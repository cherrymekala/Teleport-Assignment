# Retrieval Benchmark

Comparison of Strategy A (raw vector search) vs Strategy B (query expansion + vector search).

## Similarity Metric Choice
Cosine similarity was selected because embedding direction better represents semantic meaning than magnitude in typical transformer embeddings.
With FAISS this is implemented as IndexFlatIP over normalized vectors and normalized query vectors.

## Production Migration Path
- sentence-transformers -> Vertex AI Embeddings
- FAISS -> Vertex AI Matching Engine
- Mock query expansion -> Gemini/Vertex AI GenerativeModel

## Query Comparison

### Query 1
- Original Query: How does the system handle peak load?
- Rewritten Query: How does the system handle peak load?. Context focus: peak load. Related concepts: autoscaling, concurrency, load balancing, throughput, failover, retries.
- Observation: Strategy B increased average semantic similarity.
- Failure Modes: {'semantic_drift': False, 'over_expansion': False, 'retrieval_dilution': False, 'false_positives': False}

#### Strategy A Top Results
- Rank 1 | p4-c0 | score=0.126 | why=Term overlap with query: load, system, the.
- Rank 2 | p0-c0 | score=0.1226 | why=Term overlap with query: load, peak, the.
- Rank 3 | p2-c0 | score=0.0 | why=Semantic similarity match from embedding space.

#### Strategy B Top Results
- Rank 1 | p4-c0 | score=0.1667 | why=Term overlap with query: autoscaling, balancing, concurrency, failover, load.
- Rank 2 | p0-c0 | score=0.1622 | why=Term overlap with query: autoscaling, load, peak, the.
- Rank 3 | p2-c0 | score=0.0373 | why=Term overlap with query: throughput.

#### Metrics
- A: {'recall_at_k': 1.0, 'mrr': 1.0, 'avg_similarity': 0.08287201325098674}
- B: {'recall_at_k': 1.0, 'mrr': 1.0, 'avg_similarity': 0.1220519629617532}
- Delta (B-A): {'recall_at_k': 0.0, 'mrr': 0.0, 'avg_similarity': 0.039179949710766465}

### Query 2
- Original Query: What mechanisms improve reliability during failures?
- Rewritten Query: What mechanisms improve reliability during failures?. Context focus: reliability. Related concepts: high availability, retry policy, fault tolerance, circuit breaker, graceful degradation.
- Observation: Strategy B ranked a relevant chunk earlier.
- Failure Modes: {'semantic_drift': False, 'over_expansion': True, 'retrieval_dilution': False, 'false_positives': False}

#### Strategy A Top Results
- Rank 1 | p4-c0 | score=0.1361 | why=Term overlap with query: during.
- Rank 2 | p0-c0 | score=0.1325 | why=Semantic similarity match from embedding space.
- Rank 3 | p1-c0 | score=0.069 | why=Term overlap with query: reliability.

#### Strategy B Top Results
- Rank 1 | p1-c0 | score=0.0738 | why=Term overlap with query: circuit, reliability.
- Rank 2 | p4-c0 | score=0.0364 | why=Term overlap with query: during, retry.
- Rank 3 | p0-c0 | score=0.0354 | why=Semantic similarity match from embedding space.

#### Metrics
- A: {'recall_at_k': 0.5, 'mrr': 0.3333333333333333, 'avg_similarity': 0.11251418789227803}
- B: {'recall_at_k': 0.5, 'mrr': 1.0, 'avg_similarity': 0.04851346214612325}
- Delta (B-A): {'recall_at_k': 0.0, 'mrr': 0.6666666666666667, 'avg_similarity': -0.06400072574615479}

### Query 3
- Original Query: How is API latency reduced at high traffic?
- Rewritten Query: How is API latency reduced at high traffic?. Context focus: latency. Related concepts: p95 latency, tail latency, caching, cold starts, connection pooling, response time.
- Observation: Strategy B increased average semantic similarity.
- Failure Modes: {'semantic_drift': False, 'over_expansion': True, 'retrieval_dilution': False, 'false_positives': False}

#### Strategy A Top Results
- Rank 1 | p2-c0 | score=0.2635 | why=Term overlap with query: api, is, latency.
- Rank 2 | p1-c0 | score=0.0598 | why=Term overlap with query: is, traffic.
- Rank 3 | p4-c0 | score=0.0589 | why=Semantic similarity match from embedding space.

#### Strategy B Top Results
- Rank 1 | p2-c0 | score=0.2631 | why=Term overlap with query: api, caching, cold, connection, is.
- Rank 2 | p1-c0 | score=0.0663 | why=Term overlap with query: is, traffic.
- Rank 3 | p3-c0 | score=0.062 | why=Term overlap with query: is, traffic.

#### Metrics
- A: {'recall_at_k': 1.0, 'mrr': 1.0, 'avg_similarity': 0.12740337600310644}
- B: {'recall_at_k': 0.5, 'mrr': 1.0, 'avg_similarity': 0.1304780418674151}
- Delta (B-A): {'recall_at_k': -0.5, 'mrr': 0.0, 'avg_similarity': 0.0030746658643086566}

## Query Expansion Tradeoffs
- Semantic drift: rewrite can alter user intent.
- Over-expansion: too many concepts can broaden retrieval.
- Retrieval dilution: expanded terms can reduce precision.
- False positives: broad terms can retrieve tangential chunks.

## Fairness Rule
Both strategies use identical chunking, embeddings, index, scoring, and top-k. Only query transformation differs.
