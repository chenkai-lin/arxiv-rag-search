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
data/pdfs/      50 downloaded arXiv cs.CL PDFs
data/texts/     extracted plain text per paper + metadata.json
data/chunks/    chunks.jsonl (chunked text ready for embedding)
index/          embeddings.npy, faiss.index, chunks_meta.json
app/            FastAPI service (main.py)
notebooks/      retrieval demo notebook
reports/        retrieval_report.md (5+ example queries)
```

## Pipeline scripts

Run in order (each step documented further as it's implemented):

1. `scripts/fetch_papers.py` — download 50 arXiv cs.CL PDFs
2. `scripts/extract_text.py` — PDF -> text
3. `scripts/chunk_text.py` — text -> chunks.jsonl
4. `scripts/build_index.py` — chunks -> embeddings + FAISS index
5. `app/main.py` — serve `/search` endpoint

## Run the API

```bash
venv\Scripts\uvicorn app.main:app --reload
curl "http://localhost:8000/search?q=transformer attention mechanism"
```
