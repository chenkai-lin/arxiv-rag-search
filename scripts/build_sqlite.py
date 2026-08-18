"""Build the SQLite half of the hybrid index (Week 5).

Week 4 left us with a FAISS index over 914 chunk embeddings and two loose JSON
files. That is enough for semantic search but has no place to put paper
metadata and no way to do exact keyword matching. This script folds both into a
single SQLite database:

    documents    one row per paper (title, authors, year, keywords, abstract)
    chunks       one row per chunk, carrying the FAISS row id
    chunks_fts   FTS5 full-text index over chunk text, for BM25 keyword search

The FAISS index itself is *not* rebuilt -- embeddings are expensive and already
correct. We only need a durable mapping from a FAISS row id back to its chunk
and its paper, which the `chunks` table provides.

A note on the assignment's starter schema: it suggests

    CREATE VIRTUAL TABLE doc_chunks USING fts5(
        content, content='documents', content_rowid='doc_id')

which does not work. An external-content FTS5 table reads its indexed column
straight out of the content table, but `documents` has no `content` column --
queries would silently return empty text (and `rebuild` would error). Chunks
also outnumber documents 914 to 50, so keying chunk text by `doc_id` cannot
represent the data in the first place. We therefore give chunks their own table
and hang FTS5 off that instead.

Usage:
    python scripts/build_sqlite.py
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
from collections import Counter
from pathlib import Path

import faiss

from text_utils import DOMAIN_STOPWORDS, to_terms

ROOT = Path(__file__).resolve().parent.parent
METADATA_PATH = ROOT / "data" / "texts" / "metadata.json"
CHUNKS_PATH = ROOT / "data" / "chunks" / "chunks.jsonl"
FAISS_PATH = ROOT / "index" / "faiss.index"
DB_PATH = ROOT / "index" / "hybrid.db"

TOP_KEYWORDS = 8


def extract_keywords(docs_terms: list[list[str]], top_n: int = TOP_KEYWORDS) -> list[str]:
    """Pick each paper's most distinctive terms by TF-IDF over the 50 abstracts.

    `metadata.json` has no keywords field (the arXiv RSS feed we fetched from
    does not carry one), but the assignment's schema asks for it -- and it is
    genuinely useful later for metadata filtering. A plain TF-IDF over the
    corpus is enough: TF favours terms the paper actually dwells on, IDF drops
    the boilerplate that every cs.CL abstract shares.
    """
    n_docs = len(docs_terms)
    doc_freq = Counter()
    for terms in docs_terms:
        doc_freq.update(set(terms))

    keywords = []
    for terms in docs_terms:
        tf = Counter(terms)
        total = sum(tf.values()) or 1
        scored = {
            term: (count / total) * math.log(n_docs / (1 + doc_freq[term]))
            for term, count in tf.items()
        }
        top = sorted(scored.items(), key=lambda kv: (-kv[1], kv[0]))[:top_n]
        keywords.append(", ".join(term for term, _ in top))
    return keywords


def load_metadata() -> list[dict]:
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


def load_chunks() -> list[dict]:
    with CHUNKS_PATH.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


_ARXIV_ID_RE = re.compile(r"^(\d{2})(\d{2})\.\d{4,5}")


def parse_year(published: str | None, arxiv_id: str = "") -> int | None:
    """Best-effort publication year.

    Prefers the `published` timestamp ('2026-08-10T17:59:51+00:00' -> 2026),
    but only 10 of our 50 papers carry one -- fetch_papers_rss.py records it
    while the plain PDF-download path does not. The arXiv identifier encodes
    the same information: '2608.07525' is YYMM.NNNNN, i.e. 2026-08. Falling
    back to it fills in the other 40 rows instead of leaving `year` NULL for
    most of the corpus and making the column useless for filtering.
    """
    if published and len(published) >= 4 and published[:4].isdigit():
        return int(published[:4])

    match = _ARXIV_ID_RE.match(arxiv_id)
    if match:
        return 2000 + int(match.group(1))
    return None


SCHEMA = """
DROP TABLE IF EXISTS chunks_fts;
DROP TABLE IF EXISTS chunks;
DROP TABLE IF EXISTS documents;

CREATE TABLE documents (
    doc_id    INTEGER PRIMARY KEY,
    arxiv_id  TEXT    UNIQUE NOT NULL,   -- matches chunks.paper_id
    title     TEXT    NOT NULL,
    authors   TEXT,                      -- author list, comma-joined
    year      INTEGER,                   -- parsed out of `published`
    published TEXT,
    keywords  TEXT,                      -- TF-IDF top terms, see extract_keywords()
    abstract  TEXT,
    url       TEXT
);

CREATE TABLE chunks (
    chunk_pk    INTEGER PRIMARY KEY,     -- assigned explicitly as faiss_id + 1
    chunk_id    TEXT    UNIQUE NOT NULL, -- e.g. "2608.07525_0"
    doc_id      INTEGER NOT NULL REFERENCES documents(doc_id),
    faiss_id    INTEGER NOT NULL UNIQUE, -- row number in index/faiss.index
    text        TEXT    NOT NULL,
    token_count INTEGER
);

CREATE INDEX idx_chunks_faiss ON chunks(faiss_id);
CREATE INDEX idx_chunks_doc   ON chunks(doc_id);

-- External-content FTS5: the index points at `chunks` rather than storing a
-- second copy of all 914 passages. `porter` stemming lets "hallucinations"
-- match a query for "hallucination"; `unicode61` handles the accented author
-- names and non-ASCII characters that survive PDF extraction.
CREATE VIRTUAL TABLE chunks_fts USING fts5(
    text,
    content='chunks',
    content_rowid='chunk_pk',
    tokenize='porter unicode61'
);
"""


def main(db_path: Path) -> None:
    metadata = load_metadata()
    chunks = load_chunks()
    print(f"Loaded {len(metadata)} papers and {len(chunks)} chunks")

    # Keywords come from title + abstract: the title is short but dense, and
    # repeating it gives its terms a little extra weight in the TF count.
    docs_terms = [
        to_terms(f"{d['title']} {d['title']} {d.get('abstract', '')}", DOMAIN_STOPWORDS)
        for d in metadata
    ]
    keywords = extract_keywords(docs_terms)

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(SCHEMA)

        # --- documents -----------------------------------------------------
        doc_id_of: dict[str, int] = {}
        doc_rows = []
        for doc_id, (meta, kws) in enumerate(zip(metadata, keywords), start=1):
            doc_id_of[meta["arxiv_id"]] = doc_id
            doc_rows.append((
                doc_id,
                meta["arxiv_id"],
                meta["title"].replace("\n", " ").strip(),
                ", ".join(meta.get("authors") or []),
                parse_year(meta.get("published"), meta["arxiv_id"]),
                meta.get("published"),
                kws,
                (meta.get("abstract") or "").strip(),
                meta.get("url"),
            ))
        conn.executemany(
            "INSERT INTO documents "
            "(doc_id, arxiv_id, title, authors, year, published, keywords, abstract, url) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            doc_rows,
        )

        # --- chunks --------------------------------------------------------
        # chunks.jsonl is written in the same order build_index.py encoded it,
        # so enumerating it reproduces the FAISS row ids exactly. We set
        # chunk_pk = faiss_id + 1 because SQLite rowids start at 1 while FAISS
        # rows start at 0 -- with that offset fixed, FAISS row <-> SQLite row
        # <-> FTS5 rowid all convert by arithmetic, with no lookup table.
        missing = {c["paper_id"] for c in chunks} - doc_id_of.keys()
        assert not missing, f"chunks reference papers absent from metadata.json: {sorted(missing)}"

        chunk_rows = [
            (
                faiss_id + 1,
                c["chunk_id"],
                doc_id_of[c["paper_id"]],
                faiss_id,
                c["text"],
                c["token_count"],
            )
            for faiss_id, c in enumerate(chunks)
        ]
        conn.executemany(
            "INSERT INTO chunks (chunk_pk, chunk_id, doc_id, faiss_id, text, token_count) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            chunk_rows,
        )

        # --- full-text index -----------------------------------------------
        # External-content tables are not populated by the INSERTs above; we
        # feed the index explicitly, matching rowid to chunk_pk.
        conn.execute("INSERT INTO chunks_fts(rowid, text) SELECT chunk_pk, text FROM chunks")
        conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('optimize')")
        conn.commit()

        verify(conn, chunks)
    finally:
        conn.close()

    size_mb = db_path.stat().st_size / 1024 / 1024
    print(f"\nSaved -> {db_path} ({size_mb:.1f} MB)")


def verify(conn: sqlite3.Connection, chunks: list[dict]) -> None:
    """Fail loudly if the SQLite side and the FAISS side disagree."""
    n_docs = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    n_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    n_fts = conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]

    index = faiss.read_index(str(FAISS_PATH))
    assert n_chunks == index.ntotal, f"chunks={n_chunks} but FAISS ntotal={index.ntotal}"
    assert n_fts == n_chunks, f"FTS5 has {n_fts} rows, chunks has {n_chunks}"

    # Spot-check the faiss_id <-> chunk alignment against the original JSONL at
    # the two ends and the middle -- an off-by-one here would silently return
    # the wrong passage for every vector hit.
    for faiss_id in (0, len(chunks) // 2, len(chunks) - 1):
        row = conn.execute(
            "SELECT chunk_id, text FROM chunks WHERE faiss_id = ?", (faiss_id,)
        ).fetchone()
        assert row is not None, f"no chunk row for faiss_id={faiss_id}"
        assert row[0] == chunks[faiss_id]["chunk_id"], f"chunk_id mismatch at faiss_id={faiss_id}"
        assert row[1] == chunks[faiss_id]["text"], f"text mismatch at faiss_id={faiss_id}"

    # And check the FTS5 index really reads through to the chunk text.
    hits = conn.execute(
        "SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH ?", ('"hallucination"',)
    ).fetchone()[0]
    assert hits > 0, "FTS5 returned no rows for a term known to be in the corpus"

    n_no_year = conn.execute("SELECT COUNT(*) FROM documents WHERE year IS NULL").fetchone()[0]
    assert n_no_year == 0, f"{n_no_year} documents have no year -- check parse_year()"

    print(f"Verified: {n_docs} documents, {n_chunks} chunks, {n_fts} FTS5 rows, "
          f"FAISS ntotal={index.ntotal}")
    print(f"  faiss_id alignment OK; FTS5 'hallucination' -> {hits} chunks")

    sample = conn.execute(
        "SELECT arxiv_id, year, keywords FROM documents ORDER BY arxiv_id LIMIT 3"
    ).fetchall()
    for arxiv_id, year, kws in sample:
        print(f"  {arxiv_id} ({year}) keywords: {kws}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DB_PATH)
    args = parser.parse_args()
    main(args.db)
