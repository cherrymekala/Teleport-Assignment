from dataclasses import dataclass


@dataclass
class TextChunk:
    chunk_id: str
    source_id: str
    text: str
    position: int


@dataclass
class RetrievalResult:
    chunk_id: str
    source_id: str
    text: str
    score: float
    rank: int
    why_retrieved: str


@dataclass
class StrategyOutput:
    original_query: str
    rewritten_query: str | None
    results: list[RetrievalResult]
