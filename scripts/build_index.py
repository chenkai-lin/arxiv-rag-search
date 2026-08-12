"""Embed chunks and build the FAISS index.

Usage:
    python scripts/build_index.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = ROOT / "data" / "chunks" / "chunks.jsonl"
INDEX_DIR = ROOT / "index"
EMBEDDINGS_PATH = INDEX_DIR / "embeddings.npy"
FAISS_PATH = INDEX_DIR / "faiss.index"
META_PATH = INDEX_DIR / "chunks_meta.json"

MODEL_NAME = "all-MiniLM-L6-v2"


def load_chunks() -> list[dict]:
    with CHUNKS_PATH.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main(model_name: str, batch_size: int) -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    chunks = load_chunks()
    texts = [c["text"] for c in chunks]
    print(f"Loaded {len(chunks)} chunks from {CHUNKS_PATH}")

    print(f"Encoding with {model_name} (this takes a few minutes on CPU)...")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype("float32")

    assert embeddings.shape[0] == len(chunks), "embedding count does not match chunk count"
    assert np.isfinite(embeddings).all(), "embeddings contain NaN or Inf"
    print(f"Embeddings: shape={embeddings.shape} dtype={embeddings.dtype}")

    np.save(EMBEDDINGS_PATH, embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, str(FAISS_PATH))
    print(f"FAISS index: ntotal={index.ntotal} dim={dim}")

    # Keep the index-position -> chunk mapping alongside the vectors so the
    # search service can turn FAISS row ids back into readable passages.
    meta = [
        {
            "chunk_id": c["chunk_id"],
            "paper_id": c["paper_id"],
            "text": c["text"],
            "token_count": c["token_count"],
        }
        for c in chunks
    ]
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    # Sanity check: reload from disk and confirm a chunk retrieves itself.
    reloaded = faiss.read_index(str(FAISS_PATH))
    assert reloaded.ntotal == len(chunks), "reloaded index has wrong size"
    distances, indices = reloaded.search(embeddings[:1], 1)
    assert indices[0][0] == 0, "self-search did not return the query chunk"
    print(f"Self-search check: top-1 index={indices[0][0]} distance={distances[0][0]:.6f}")

    print(f"\nSaved:\n  {EMBEDDINGS_PATH}\n  {FAISS_PATH}\n  {META_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    main(args.model, args.batch_size)
