"""FastAPI service exposing semantic search over the arXiv cs.CL index.

Usage:
    uvicorn app.main:app --reload
    curl "http://localhost:8000/search?q=transformer attention mechanism&k=3"
"""
from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Query

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from search_lib import SearchEngine  # noqa: E402

engine: SearchEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    engine = SearchEngine()  # loads the model + FAISS index once, at startup
    yield


app = FastAPI(title="arxiv-rag-search", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "chunks": engine.index.ntotal}


@app.get("/search")
def search(
    q: str = Query(..., min_length=1, description="Search query"),
    k: int = Query(3, ge=1, le=20, description="Number of results to return"),
) -> dict:
    return {"query": q, "results": engine.search(q, k=k)}
