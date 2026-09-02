# arxiv-rag-search

RAG pipeline over arXiv cs.CL papers: PDF extraction, chunking, sentence-transformers
embeddings, FAISS indexing, FastAPI endpoints.

Week 4 built the semantic half (FAISS + `/search`). Week 5 added the sparse half — paper
metadata and an FTS5 full-text index in SQLite — and fuses the two into hybrid retrieval
(`/hybrid_search`). The week 4 embeddings are reused unchanged; nothing is re-encoded.

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
data/eval/      queries.json -- 15 queries with chunk-level relevance labels (committed)
data/sft/       week 7 synthetic Q&A: raw_qa.json (generator output), synthetic_qa.jsonl
                + synthetic_qa_val.jsonl (chat-formatted train/val), dataset_stats.json
                (all committed); paper_contexts.json is a regeneratable inspection dump
index/          embeddings.npy, faiss.index, hybrid.db (gitignored, regenerate via
                scripts/build_index.py and scripts/build_sqlite.py);
                chunks_meta.json (committed)
scripts/        pipeline scripts, plus search_lib.py (week 4 SearchEngine) and
                hybrid_lib.py (week 5 HybridSearchEngine, used by the notebook, the
                evaluation, and the FastAPI app)
app/            FastAPI service (main.py)
notebooks/      retrieval_demo.ipynb -- 5 example queries with top-3 results, executed
                hybrid_eval.ipynb   -- hybrid vs. baselines on the 15-query set, executed
reports/        retrieval_report.md     -- 5 queries, regenerate via
                                           scripts/generate_retrieval_report.py
                hybrid_eval_report.md   -- hybrid evaluation, regenerate via
                                           scripts/evaluate_retrieval.py
```

## The hybrid index

Two stores, joined by arithmetic rather than a lookup table:

| | |
|---|---|
| `index/faiss.index` | 914 chunk embeddings (`all-MiniLM-L6-v2`, `IndexFlatL2`) |
| `index/hybrid.db` | `documents` (paper metadata), `chunks` (text + its FAISS row id), `chunks_fts` (FTS5 index over chunk text) |

`chunks.chunk_pk` is pinned to `faiss_id + 1`, because SQLite rowids start at 1 and FAISS
rows at 0. With that offset fixed, a FAISS hit, a SQLite row and an FTS5 rowid all convert
between each other by adding or subtracting one.

Note that the assignment's starter schema (`fts5(content, content='documents',
content_rowid='doc_id')`) is deliberately not followed: an external-content FTS5 table
reads its indexed column out of the content table, and `documents` has no `content`
column — and with 914 chunks against 50 documents, chunk text cannot be keyed by `doc_id`
anyway. See the docstring in `scripts/build_sqlite.py`.

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
5. `scripts/build_sqlite.py` — chunks + metadata -> `index/hybrid.db` (documents table,
   chunks table carrying FAISS row ids, FTS5 full-text index). Reads the existing
   `faiss.index` only to assert the two stay the same size.
6. `scripts/generate_retrieval_report.py` — regenerate `reports/retrieval_report.md`
7. `scripts/evaluate_retrieval.py` — regenerate `reports/hybrid_eval_report.md`
8. `app/main.py` — serve the endpoints

## Run the API

```bash
venv\Scripts\uvicorn app.main:app --reload
curl "http://localhost:8000/health"

# week 4: dense vectors only
curl "http://localhost:8000/search?q=transformer%20attention%20mechanism&k=3"

# week 5: hybrid
curl "http://localhost:8000/hybrid_search?query=MoE%20load%20balancing&k=3"
curl "http://localhost:8000/hybrid_search?query=MoE%20load%20balancing&k=3&method=rrf"
curl "http://localhost:8000/hybrid_search?query=AraSSM&k=3&kw_backend=bm25&alpha=0.3"
```

`/hybrid_search` takes `method` (`weighted` | `rrf`), `alpha` (vector weight, weighted
only), `kw_backend` (`fts5` | `bm25`) and `pool` (candidates per leg before fusion).
Each result carries `vec_score` and `kw_score` next to `fused_score`, so you can see
which leg is responsible for a hit.

`/search` keeps its week 4 response shape exactly. Both endpoints are served by one
`HybridSearchEngine`, so the model, the FAISS index and the SQLite connection load once
at process startup (FastAPI `lifespan`), not per request and not once per endpoint.

## Week 7: synthetic Q&A for fine-tuning

Week 7 reuses this corpus for the opposite purpose — distilling its knowledge into a
model's weights instead of retrieving it. Two scripts, kept separate because generation
is slow, paid and vendor-specific while formatting is fast, free and deterministic:

- `scripts/generate_synthetic_qa.py` — for each paper in `metadata.json`, builds a context
  (abstract + a budgeted digest of each detected section; `--dump-contexts` writes them
  to `data/sft/paper_contexts.json` without any API call) and asks an LLM for 5 Q&A pairs
  (4 answerable + 1 edge case whose premise is false or whose detail the paper never
  reports). Output: `data/sft/raw_qa.json`. Resumable — papers already in the output are
  skipped. **Provider is configuration, not code**: `LLM_PROVIDER` (`openai` |
  `anthropic`), `LLM_MODEL`, `LLM_API_KEY`, `LLM_BASE_URL` — the `openai` path is the
  wire protocol, not the company, so `LLM_BASE_URL` points it at DeepSeek, Qwen,
  OpenRouter, Ollama, vLLM, … See the config block in the script for known-good
  combinations. Put them in a git-ignored `.env` at the repo root.
- `scripts/build_sft_dataset.py` — `raw_qa.json` → chat-formatted JSONL. Applies the
  `<|system|>…<|user|>…<|assistant|>…` template once, dedups on the normalised question,
  and splits **by paper** (`--val-split`, default 0.1) so no validation question has a
  sibling in training. Output: `synthetic_qa.jsonl` (450), `synthetic_qa_val.jsonl` (50),
  `dataset_stats.json`.

The committed `raw_qa.json` for the 100-paper corpus was authored with Claude (Sonnet)
rather than GPT-4 — an equivalent synthetic-data generator — from the same
abstract-plus-section-digest context the script builds. Re-run `build_sft_dataset.py` to
regenerate the JSONL; re-run `generate_synthetic_qa.py` with a provider configured to
regenerate `raw_qa.json` from scratch.

## Hybrid retrieval: does it help?

`reports/hybrid_eval_report.md` and `notebooks/hybrid_eval.ipynb` compare five
configurations on 15 queries with chunk-level relevance labels:

| Method | Chunk Hit@1 | Chunk Hit@3 | Chunk MRR | Paper Hit@3 |
|---|---|---|---|---|
| vector-only | 53.3% | 73.3% | 0.639 | 100.0% |
| fts5-only | 20.0% | 60.0% | 0.419 | 93.3% |
| bm25okapi-only | 26.7% | 80.0% | 0.521 | 86.7% |
| **hybrid-weighted (α=0.5)** | **60.0%** | **80.0%** | **0.691** | 100.0% |
| hybrid-rrf | 53.3% | 73.3% | 0.657 | 100.0% |

Yes, but narrowly, and the honest caveats matter:

- **The alpha sweep is the real evidence.** Chunk Hit@3 runs 60.0% at pure keyword, peaks
  at 80.0% in the middle (α=0.4–0.5), and settles at 73.3% at pure vector. An interior
  maximum is what hybrid retrieval predicts, and unlike the headline numbers it does not
  rest on any single query.
- **The headline gap is one query.** At n=15 each query is worth 6.7 points, so
  hybrid's Hit@1 lead over vectors is a single query changing its mind.
- **Paper-level scoring is saturated** — three methods tie at 100%. A first version of
  this evaluation labelled relevance per paper and every method scored an identical
  83.3%; the corpus is 50 papers on disjoint topics, so finding the right *paper* is
  trivial. The labels were moved to chunk level, which is also the granularity a RAG
  pipeline actually consumes.
- **One hypothesis failed.** Literal identifiers were expected to favour BM25, but
  vectors beat FTS5 there (100% vs 75%): MiniLM splits a coined name like `AraSSM` into
  subwords that are frequent in the paper that coined it, and nothing on a 50-paper
  corpus competes for them. Dense retrieval does fail on rare literal strings — it needs
  a corpus with real lexical competition to show it.

## Known limitation

One paper (`2608.07525`) puts its References heading at 26% into the document because a
large appendix follows the reference list, below the 30% position guard in
`strip_references()`. Its citation list wasn't stripped and can surface in results (see
Query 1 in `reports/retrieval_report.md`). Affects 1/50 papers in this corpus.
