"""Shared semantic search helper used by the demo notebook, the retrieval
report generator, and the FastAPI service, so all three query the index the
same way instead of re-implementing encode+search separately.

Usage:
    from search_lib import SearchEngine
    engine = SearchEngine()
    engine.search("How does chain-of-thought reasoning work?", k=3)
"""
from __future__ import annotations

import json
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
FAISS_PATH = ROOT / "index" / "faiss.index"
META_PATH = ROOT / "index" / "chunks_meta.json"

MODEL_NAME = "all-MiniLM-L6-v2"


class SearchEngine:
    def __init__(self, model_name: str = MODEL_NAME):
        self.model = SentenceTransformer(model_name)
        self.index = faiss.read_index(str(FAISS_PATH))
        self.meta = json.loads(META_PATH.read_text(encoding="utf-8"))

    def search(self, query: str, k: int = 3) -> list[dict]:
        embedding = self.model.encode([query], convert_to_numpy=True).astype("float32")
        distances, indices = self.index.search(embedding, k)
        results = []
        for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), start=1):
            chunk = self.meta[idx]
            results.append(
                {
                    "rank": rank,
                    "distance": float(dist),
                    "chunk_id": chunk["chunk_id"],
                    "paper_id": chunk["paper_id"],
                    "text": chunk["text"],
                }
            )
        return results
