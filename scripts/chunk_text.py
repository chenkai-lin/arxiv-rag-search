"""Chunk extracted paper text into <=512-token pieces for embedding.

Sentence-aware: chunks are built by accumulating whole sentences up to
max_tokens, so breaks fall at sentence boundaries instead of mid-sentence
(better retrieval precision than a pure sliding window). A short word-level
overlap is carried into the next chunk so context isn't lost across a
boundary. Only the rare oversized "sentence" (e.g. a garbled PDF table)
falls back to a fixed-size word window.

Token count here means whitespace-split word count, matching the
assignment's own definition (text.split()) -- no extra tokenizer dependency.

Usage:
    python scripts/chunk_text.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEXT_DIR = ROOT / "data" / "texts"
CHUNKS_PATH = ROOT / "data" / "chunks" / "chunks.jsonl"

MAX_TOKENS = 512
OVERLAP_TOKENS = 50

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    return [s.strip() for s in _SENTENCE_BOUNDARY.split(text) if s.strip()]


def chunk_paper(text: str, max_tokens: int = MAX_TOKENS, overlap_tokens: int = OVERLAP_TOKENS) -> list[str]:
    sentences = split_sentences(text)
    chunks: list[str] = []
    current: list[str] = []

    def flush():
        if current:
            chunks.append(" ".join(current))

    for sent in sentences:
        sent_words = sent.split()

        if len(sent_words) > max_tokens:
            # Oversized single "sentence" (e.g. no punctuation was found in a
            # run of text) -- slice it with a fixed sliding window.
            flush()
            current.clear()
            step = max_tokens - overlap_tokens
            for i in range(0, len(sent_words), step):
                chunks.append(" ".join(sent_words[i:i + max_tokens]))
            continue

        if len(current) + len(sent_words) > max_tokens:
            flush()
            carry = current[-overlap_tokens:] if len(current) > overlap_tokens else current[:]
            current = carry + sent_words
            if len(current) > max_tokens:
                # the carried-over overlap plus this (large) sentence alone
                # would already exceed the cap -- drop the overlap here
                current = list(sent_words)
        else:
            current.extend(sent_words)

    flush()
    return chunks


def main() -> None:
    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    text_paths = sorted(p for p in TEXT_DIR.glob("*.txt"))
    print(f"Found {len(text_paths)} text files in {TEXT_DIR}")

    records = []
    for text_path in text_paths:
        paper_id = text_path.stem
        text = text_path.read_text(encoding="utf-8")
        paper_chunks = chunk_paper(text)
        for i, chunk_text_ in enumerate(paper_chunks):
            records.append(
                {
                    "chunk_id": f"{paper_id}_{i}",
                    "paper_id": paper_id,
                    "text": chunk_text_,
                    "token_count": len(chunk_text_.split()),
                }
            )
        print(f"  {paper_id}: {len(paper_chunks)} chunks")

    with CHUNKS_PATH.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Sanity checks
    token_counts = [r["token_count"] for r in records]
    assert all(tc <= MAX_TOKENS for tc in token_counts), "found a chunk over the token cap"
    assert all(tc > 0 for tc in token_counts), "found an empty chunk"

    print(f"\nTotal chunks: {len(records)}")
    print(f"Token count: min={min(token_counts)} max={max(token_counts)} avg={sum(token_counts)/len(token_counts):.1f}")
    print(f"Saved -> {CHUNKS_PATH}")


if __name__ == "__main__":
    main()
