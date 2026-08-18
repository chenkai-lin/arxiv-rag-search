"""FastAPI service over the arXiv cs.CL index.

Endpoints:
    /search         Week 4 semantic search, response shape unchanged
    /hybrid_search  Week 5 hybrid search (FAISS + BM25, fused)

Both are served by one HybridSearchEngine, so the sentence-transformer model
and the FAISS index are loaded once at startup rather than once per endpoint.

Usage:
    uvicorn app.main:app --reload
    curl "http://localhost:8000/search?q=transformer attention mechanism&k=3"
    curl "http://localhost:8000/hybrid_search?query=MoE load balancing&k=3"
"""
from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Query

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from hybrid_lib import DEFAULT_POOL, HybridSearchEngine  # noqa: E402

engine: HybridSearchEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    # Loads the model, the FAISS index and the SQLite connection once.
    engine = HybridSearchEngine()
    yield
    engine.close()


app = FastAPI(title="arxiv-rag-search", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", **engine.stats()}


@app.get("/search")
def search(
    q: str = Query(..., min_length=1, description="Search query"),
    k: int = Query(3, ge=1, le=20, description="Number of results to return"),
) -> dict:
    """Dense-vector search. Kept on the Week 4 response shape.

    Callers built against week 4 expect exactly {rank, distance, chunk_id,
    paper_id, text}, so the richer hybrid fields are projected away here
    rather than leaking into an established contract.
    """
    results = [
        {
            "rank": r["rank"],
            "distance": r["distance"],
            "chunk_id": r["chunk_id"],
            "paper_id": r["paper_id"],
            "text": r["text"],
        }
        for r in engine.search_vector_only(q, k=k)
    ]
    return {"query": q, "results": results}


@app.get("/hybrid_search")
def hybrid_search(
    query: str = Query(..., min_length=1, description="Search query"),
    k: int = Query(3, ge=1, le=20, description="Number of results to return"),
    method: str = Query("weighted", pattern="^(weighted|rrf)$", description="Fusion strategy"),
    alpha: float = Query(
        0.5, ge=0.0, le=1.0,
        description="Vector weight for 'weighted' fusion: 1.0 is pure vector, 0.0 pure keyword. "
                    "Ignored when method=rrf.",
    ),
    kw_backend: str = Query(
        "fts5", pattern="^(fts5|bm25)$", description="Keyword backend: SQLite FTS5 or rank_bm25"
    ),
    pool: int = Query(
        DEFAULT_POOL, ge=1, le=500,
        description="Candidates each leg contributes before fusion. Must exceed k for fusion "
                    "to be able to reorder anything.",
    ),
) -> dict:
    """Hybrid search: dense vectors and BM25 keywords, fused and re-ranked.

    Each result carries its component scores (`vec_score`, `kw_score`) next to
    the `fused_score`, so a caller can see which leg is responsible for a hit
    instead of having to trust an opaque ranking.
    """
    results = engine.search(
        query, k=k, method=method, alpha=alpha, kw_backend=kw_backend, pool=pool
    )
    return {
        "query": query,
        "method": method,
        # Reporting alpha under RRF would imply it did something; it does not.
        "alpha": alpha if method == "weighted" else None,
        "kw_backend": kw_backend,
        "pool": pool,
        "results": results,
    }
