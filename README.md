# GenAI RAG Vector Search Assessment

This repository contains a local Retrieval-Augmented Generation benchmark that compares:

- Strategy A: raw vector search
- Strategy B: intention-based query expansion before vector search

## Features

- Centralized `RetrievalConfig` for retrieval settings.
- `RetrievalResult` model with retrieval reasoning.
- Cosine similarity with explicit L2 normalization.
- FAISS `IndexFlatIP` support with a NumPy fallback.
- Evaluation metrics: Recall@K, MRR, average similarity.
- Structured benchmark output including rewrite details and failure-mode flags.
- Mock-friendly adapters for Vertex AI embedding and generation interfaces.

## Run

```bash
PYTHONPATH=src /opt/homebrew/bin/python3 -m rag_assessment.main

or 

python3 main.py
```

This generates:

- `benchmark_output.json`
- `retrieval_benchmark.md`

## Test

```bash
PYTHONPATH=src /opt/homebrew/bin/python3 -m pytest -q
```
