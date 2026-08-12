# arxiv-rag-search

RAG pipeline over arXiv cs.CL papers: PDF extraction, chunking, sentence-transformers embeddings, FAISS indexing, FastAPI `/search` endpoint.

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Project layout

```
data/pdfs/      50 downloaded arXiv cs.CL PDFs (gitignored, regenerate via scripts/fetch_papers_rss.py)
data/texts/     extracted plain text per paper (gitignored) + metadata.json (committed)
data/chunks/    chunks.jsonl -- 914 chunks, reference sections stripped (committed)
index/          embeddings.npy, faiss.index (gitignored, regenerate via scripts/build_index.py);
                chunks_meta.json (committed)
scripts/        pipeline scripts, plus search_lib.py (shared SearchEngine used by the
                notebook, the report generator, and the FastAPI app)
app/            FastAPI service (main.py)
notebooks/      retrieval_demo.ipynb -- 5 example queries with top-3 results, executed
reports/        retrieval_report.md -- same 5 queries, regenerate via
                scripts/generate_retrieval_report.py
```

## Pipeline scripts

Run in order:

1. `scripts/fetch_papers_rss.py --n 50` — download 50 arXiv cs.CL PDFs via the RSS feed +
   direct PDF download. Resumable (skips PDFs/metadata already on disk).
   `scripts/fetch_papers.py` (the official `export.arxiv.org` query API) is kept for
   reference but is not the recommended path -- that API rate-limited (HTTP 429)
   persistently on the network this project was built on, while the RSS feed and direct
   PDF endpoint did not.
2. `scripts/extract_text.py` — PDF -> text, and truncates each paper at its last
   References/Bibliography heading (if found past 30% into the document) so citation
   lists don't pollute retrieval. Stripped 47/50 papers cleanly; see "Known limitation"
   below for the one paper where the heuristic doesn't apply.
3. `scripts/chunk_text.py` — text -> `data/chunks/chunks.jsonl`, sentence-aware chunking
   (512-token cap, 50-token overlap)
4. `scripts/build_index.py` — chunks -> `all-MiniLM-L6-v2` embeddings + FAISS
   `IndexFlatL2` index
5. `scripts/generate_retrieval_report.py` — regenerate `reports/retrieval_report.md`
6. `app/main.py` — serve the `/search` endpoint

## Run the API

```bash
venv\Scripts\uvicorn app.main:app --reload
curl "http://localhost:8000/health"
curl "http://localhost:8000/search?q=transformer%20attention%20mechanism&k=3"
```

The model and FAISS index load once at process startup (FastAPI `lifespan`), not per
request.

## Known limitation

One paper (`2608.07525`) puts its References heading at 26% into the document because a
large appendix follows the reference list, below the 30% position guard in
`strip_references()`. Its citation list wasn't stripped and can surface in results (see
Query 1 in `reports/retrieval_report.md`). Affects 1/50 papers in this corpus.
