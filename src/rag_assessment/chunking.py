from __future__ import annotations

from .models import TextChunk


def split_into_chunks(
    paragraphs: list[str],
    chunk_size: int,
    chunk_overlap: int,
) -> list[TextChunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and < chunk_size")

    chunks: list[TextChunk] = []
    step = chunk_size - chunk_overlap
    for p_idx, paragraph in enumerate(paragraphs):
        words = paragraph.split()
        if not words:
            continue
        chunk_pos = 0
        for start in range(0, len(words), step):
            window = words[start : start + chunk_size]
            if not window:
                continue
            chunk_id = f"p{p_idx}-c{chunk_pos}"
            chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    source_id=f"paragraph-{p_idx}",
                    text=" ".join(window),
                    position=chunk_pos,
                )
            )
            chunk_pos += 1
            if start + chunk_size >= len(words):
                break
    return chunks
