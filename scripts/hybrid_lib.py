"""Hybrid retrieval: FAISS dense vectors + BM25 sparse keywords, fused (Week 5).

Week 4's SearchEngine does semantic search only. That is strong on paraphrase
("how do models make things up" finds papers about hallucination) but weak on
precise strings: a 384-dim MiniLM embedding of "LongCat-Flash" lands in a
generic neighbourhood of MoE terminology and the chunk that literally names the
model can rank below chunks that merely discuss the topic. Keyword search has
the mirror-image failure. Running both and fusing their rankings recovers each
one's blind spot.

Two keyword backends are implemented so the evaluation can separate "BM25
helps" from "this particular BM25 implementation helps":

    fts5    SQLite's built-in FTS5 with bm25() -- no extra dependency, porter
            stemming, and it queries the same database the metadata lives in
    bm25    rank_bm25.BM25Okapi -- scores all 914 chunks in Python, no stemming

Two fusion strategies:

    weighted  min-max normalize both score lists, then alpha*vec + (1-alpha)*kw
    rrf       reciprocal rank fusion, sum of 1/(K + rank); ignores raw scores,
              so it needs no normalization and is robust to outlier scores

Usage:
    from hybrid_lib import HybridSearchEngine
    engine = HybridSearchEngine()
    engine.search("MoE routing and load balancing", k=3)
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from text_utils import to_fts_query, to_terms

ROOT = Path(__file__).resolve().parent.parent
FAISS_PATH = ROOT / "index" / "faiss.index"
DB_PATH = ROOT / "index" / "hybrid.db"

MODEL_NAME = "all-MiniLM-L6-v2"

# How many candidates each leg contributes before fusion. This must be much
# larger than the k we finally return: if both legs only offered their top 3,
# the "fusion" would just be concatenating two short lists and could never
# promote a chunk that ranked, say, 12th on vectors and 2nd on keywords --
# which is precisely the case hybrid retrieval exists to catch.
DEFAULT_POOL = 50

# RRF's rank damping constant. 60 is the value from the original Cormack et al.
# paper and the de-facto default; it flattens the difference between ranks 1
# and 2 enough that one leg's overconfidence cannot dominate the fusion.
RRF_K = 60


def _minmax(scores: dict[int, float]) -> dict[int, float]:
    """Scale a leg's scores into [0, 1] within this query's candidate pool.

    Normalizing per query (not globally) matters: FAISS L2 distances and BM25
    scores live on incomparable scales, and BM25's scale further depends on how
    many query terms matched. Only the *relative* ordering inside one result
    list is meaningful, so that is what we preserve.
    """
    if not scores:
        return {}
    values = list(scores.values())
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        # Single candidate, or an exact tie across the pool -- any affine
        # rescaling is arbitrary, so treat them all as equally good.
        return {key: 1.0 for key in scores}
    return {key: (v - lo) / (hi - lo) for key, v in scores.items()}


class HybridSearchEngine:
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        db_path: Path = DB_PATH,
        faiss_path: Path = FAISS_PATH,
    ):
        self.model = SentenceTransformer(model_name)
        self.index = faiss.read_index(str(faiss_path))
        # check_same_thread=False: FastAPI runs sync endpoints on a threadpool,
        # so the connection opened at startup gets used from worker threads.
        # Safe here because every query is read-only.
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

        self.n_chunks = self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        self.n_documents = self.conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        assert self.n_chunks == self.index.ntotal, (
            f"SQLite has {self.n_chunks} chunks but FAISS has {self.index.ntotal} vectors -- "
            "rerun scripts/build_sqlite.py"
        )

        # Built on first use: tokenizing 914 chunks costs a second or two, and
        # a caller who only ever uses the fts5 backend should not pay for it.
        self._bm25 = None
        self._bm25_faiss_ids: list[int] = []

    # ------------------------------------------------------------------
    # Recall legs -- each returns {faiss_id: score}, higher score = better
    # ------------------------------------------------------------------

    def _vector_recall(self, query: str, pool: int) -> tuple[dict[int, float], dict[int, float]]:
        """Dense retrieval over the Week 4 FAISS index.

        IndexFlatL2 returns squared L2 distance (lower = closer), so we map it
        to a similarity with 1/(1+d): monotonic, bounded in (0, 1], and defined
        at d=0. The subsequent min-max normalization makes the exact shape of
        this mapping mostly irrelevant -- what matters is that it is decreasing.

        Returns both the similarities used for fusion and the raw distances,
        which the API surfaces so that /search keeps the Week 4 response shape
        without needing a second SearchEngine (and a second copy of the model)
        loaded alongside this one.
        """
        embedding = self.model.encode([query], convert_to_numpy=True).astype("float32")
        distances, indices = self.index.search(embedding, min(pool, self.index.ntotal))
        sims: dict[int, float] = {}
        dists: dict[int, float] = {}
        for idx, dist in zip(indices[0], distances[0]):
            if idx < 0:  # FAISS pads with -1 when fewer than `pool` results exist
                continue
            sims[int(idx)] = 1.0 / (1.0 + float(dist))
            dists[int(idx)] = float(dist)
        return sims, dists

    def _fts5_recall(self, query: str, pool: int) -> dict[int, float]:
        """Sparse retrieval via SQLite FTS5's built-in bm25() ranking."""
        match_expr = to_fts_query(query)
        if match_expr is None:
            return {}

        rows = self.conn.execute(
            """
            SELECT c.faiss_id AS faiss_id, bm25(chunks_fts) AS score
            FROM chunks_fts
            JOIN chunks c ON c.chunk_pk = chunks_fts.rowid
            WHERE chunks_fts MATCH ?
            ORDER BY score          -- bm25() is negative; more negative = better
            LIMIT ?
            """,
            (match_expr, pool),
        ).fetchall()
        # Flip the sign so that, like every other leg, bigger means better.
        return {row["faiss_id"]: -float(row["score"]) for row in rows}

    def _bm25okapi_recall(self, query: str, pool: int) -> dict[int, float]:
        """Sparse retrieval via rank_bm25, as an independent BM25 reference."""
        if self._bm25 is None:
            self._build_bm25()

        terms = to_terms(query)
        if not terms:
            return {}

        scores = self._bm25.get_scores(terms)
        # BM25Okapi scores every document; keep the top `pool`, and drop zeros
        # (no query term present) so they do not pad the pool with non-matches.
        top = np.argsort(scores)[::-1][:pool]
        return {
            self._bm25_faiss_ids[i]: float(scores[i])
            for i in top
            if scores[i] > 0
        }

    def _build_bm25(self) -> None:
        from rank_bm25 import BM25Okapi  # imported lazily, see __init__

        rows = self.conn.execute(
            "SELECT faiss_id, text FROM chunks ORDER BY faiss_id"
        ).fetchall()
        self._bm25_faiss_ids = [row["faiss_id"] for row in rows]
        # Same tokenizer as the FTS5 leg (text_utils.to_terms), so the two
        # backends differ only in their ranking implementation -- notably FTS5
        # applies porter stemming here and rank_bm25 does not.
        self._bm25 = BM25Okapi([to_terms(row["text"]) for row in rows])

    def _keyword_recall(self, query: str, pool: int, backend: str) -> dict[int, float]:
        if backend == "fts5":
            return self._fts5_recall(query, pool)
        if backend == "bm25":
            return self._bm25okapi_recall(query, pool)
        raise ValueError(f"unknown keyword backend: {backend!r} (expected 'fts5' or 'bm25')")

    # ------------------------------------------------------------------
    # Fusion
    # ------------------------------------------------------------------

    @staticmethod
    def _fuse(
        vec: dict[int, float],
        kw: dict[int, float],
        method: str,
        alpha: float,
    ) -> list[tuple[int, float, float, float]]:
        """Merge two ranked legs into one list of (faiss_id, fused, vec, kw).

        The candidate set is the *union* of both legs. A chunk found by only
        one leg keeps that leg's score and takes 0 from the other, so it can
        still win if that single signal is strong enough -- the union is what
        lets keyword search rescue documents the embedding missed entirely.
        """
        # Ordering the union with the vector leg first (in its own rank order)
        # makes ties deterministic under Python's stable sort, which is what
        # lets alpha=1.0 reproduce vector-only ranking exactly.
        union: dict[int, None] = {}
        for faiss_id in vec:
            union.setdefault(faiss_id, None)
        for faiss_id in kw:
            union.setdefault(faiss_id, None)

        if method == "weighted":
            vec_n, kw_n = _minmax(vec), _minmax(kw)
            fused = {
                fid: alpha * vec_n.get(fid, 0.0) + (1.0 - alpha) * kw_n.get(fid, 0.0)
                for fid in union
            }
        elif method == "rrf":
            # Rank-based: position in each list is all that counts, so wildly
            # different score scales never need reconciling.
            vec_n = {fid: 0.0 for fid in vec}
            kw_n = {fid: 0.0 for fid in kw}
            fused = dict.fromkeys(union, 0.0)
            for leg in (vec, kw):
                ranked = sorted(leg.items(), key=lambda kv: -kv[1])
                for rank, (fid, _score) in enumerate(ranked, start=1):
                    fused[fid] += 1.0 / (RRF_K + rank)
            vec_n, kw_n = _minmax(vec), _minmax(kw)  # reported for transparency
        else:
            raise ValueError(f"unknown fusion method: {method!r} (expected 'weighted' or 'rrf')")

        merged = [
            (fid, fused[fid], vec_n.get(fid, 0.0), kw_n.get(fid, 0.0))
            for fid in union
        ]
        merged.sort(key=lambda item: -item[1])  # stable: preserves union order on ties
        return merged

    # ------------------------------------------------------------------
    # Hydration
    # ------------------------------------------------------------------

    def _hydrate(self, faiss_ids: list[int]) -> dict[int, sqlite3.Row]:
        """Fetch chunk text plus its paper's metadata for the final results.

        Done once for the top-k only, after fusion -- joining metadata for all
        `pool` candidates of both legs would be wasted work.
        """
        if not faiss_ids:
            return {}
        placeholders = ",".join("?" * len(faiss_ids))
        rows = self.conn.execute(
            f"""
            SELECT c.faiss_id, c.chunk_id, c.text, c.token_count,
                   d.arxiv_id, d.title, d.authors, d.year, d.keywords, d.url
            FROM chunks c
            JOIN documents d ON d.doc_id = c.doc_id
            WHERE c.faiss_id IN ({placeholders})
            """,
            faiss_ids,
        ).fetchall()
        return {row["faiss_id"]: row for row in rows}

    def _format(
        self,
        merged: list[tuple[int, float, float, float]],
        k: int,
        distances: dict[int, float] | None = None,
    ) -> list[dict]:
        top = merged[:k]
        hydrated = self._hydrate([fid for fid, *_ in top])
        distances = distances or {}

        results = []
        for rank, (faiss_id, fused, vec_score, kw_score) in enumerate(top, start=1):
            row = hydrated[faiss_id]
            results.append({
                "rank": rank,
                "chunk_id": row["chunk_id"],
                "paper_id": row["arxiv_id"],
                "faiss_id": faiss_id,
                "title": row["title"],
                "authors": row["authors"],
                "year": row["year"],
                "url": row["url"],
                "text": row["text"],
                # Raw FAISS L2 distance, or None for a chunk the keyword leg
                # found on its own and the vector leg never saw.
                "distance": distances.get(faiss_id),
                "vec_score": round(vec_score, 6),
                "kw_score": round(kw_score, 6),
                "fused_score": round(fused, 6),
            })
        return results

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        k: int = 3,
        method: str = "weighted",
        alpha: float = 0.5,
        kw_backend: str = "fts5",
        pool: int = DEFAULT_POOL,
    ) -> list[dict]:
        """Hybrid search: run both legs, fuse, return the top k."""
        vec, dists = self._vector_recall(query, pool)
        kw = self._keyword_recall(query, pool, kw_backend)
        merged = self._fuse(vec, kw, method, alpha)
        return self._format(merged, k, dists)

    def search_vector_only(self, query: str, k: int = 3, pool: int = DEFAULT_POOL) -> list[dict]:
        """Week 4 baseline, re-expressed through the hybrid result schema."""
        vec, dists = self._vector_recall(query, pool)
        merged = self._fuse(vec, {}, "weighted", alpha=1.0)
        return self._format(merged, k, dists)

    def search_keyword_only(
        self, query: str, k: int = 3, kw_backend: str = "fts5", pool: int = DEFAULT_POOL
    ) -> list[dict]:
        """Sparse-only baseline."""
        kw = self._keyword_recall(query, pool, kw_backend)
        merged = self._fuse({}, kw, "weighted", alpha=0.0)
        return self._format(merged, k)

    def stats(self) -> dict:
        return {
            "documents": self.n_documents,
            "chunks": self.n_chunks,
            "faiss_ntotal": self.index.ntotal,
            "faiss_dim": self.index.d,
        }

    def close(self) -> None:
        self.conn.close()
