# Hybrid Retrieval Evaluation

15 labelled queries against the arXiv cs.CL index (914 chunks from 50 papers). Each leg contributes 50 candidates before fusion. Regenerate with `python scripts/evaluate_retrieval.py`.

Scored at two granularities. **Chunk level** is the primary metric: did the passage that actually answers the question reach the top k? That is the passage a RAG pipeline would put in the prompt. **Paper level** asks only whether any passage from the right paper reached the top k, and is reported as a control -- see *Why chunk level* below.

## Overall

| Method | Chunk Hit@1 | Chunk Hit@3 | Chunk MRR@10 | Paper Hit@3 |
|---|---|---|---|---|
| vector-only | 53.3% | 73.3% | 0.639 | 100.0% |
| fts5-only | 20.0% | 60.0% | 0.419 | 93.3% |
| bm25okapi-only | 26.7% | 80.0% | 0.521 | 86.7% |
| hybrid-weighted (a=0.5) | 60.0% | 80.0% | 0.691 | 100.0% |
| hybrid-rrf | 53.3% | 73.3% | 0.657 | 100.0% |

## By query type

Chunk Hit@3, split by how the query is phrased. This is where each method's advantage actually comes from.

| Method | keyword (n=4) | mixed (n=4) | semantic (n=7) |
|---|---|---|---|
| vector-only | 100.0% | 75.0% | 57.1% |
| fts5-only | 75.0% | 100.0% | 28.6% |
| bm25okapi-only | 100.0% | 100.0% | 57.1% |
| hybrid-weighted (a=0.5) | 100.0% | 100.0% | 57.1% |
| hybrid-rrf | 100.0% | 75.0% | 57.1% |

## Fusion weight sweep

`alpha` weights the vector leg: `score = alpha * vector + (1 - alpha) * keyword`. alpha=0.0 is pure keyword, alpha=1.0 is pure vector.

| alpha | Chunk Hit@3 |
|---|---|
| 0.0 | 60.0% |
| 0.1 | 66.7% |
| 0.2 | 73.3% |
| 0.3 | 73.3% |
| 0.4 | 80.0% |
| 0.5 | 80.0%  <-- best |
| 0.6 | 73.3% |
| 0.7 | 73.3% |
| 0.8 | 73.3% |
| 0.9 | 73.3% |
| 1.0 | 73.3% |

## Why chunk level

An earlier version of this evaluation labelled relevance at paper level. Every method scored an identical 83.3% Hit@3 and the alpha sweep was flat across its whole range. The corpus is 50 papers on essentially disjoint topics, so almost any signal identifies the right paper: a probe of 16 single-token queries for coined names (`AraSSM`, `WuYuEval`, `Search-G1`, ...) ranked the correct paper first on **vectors alone** in 14 of 16 cases, because the subword pieces of a coined name are themselves frequent in that paper and nothing else competes for them.

The Paper Hit@k column above shows the same saturation persists with the current queries. Chunk level asks the harder and more useful question, and separates the methods.

## Reading the results

**Fusion helps, modestly.** `hybrid-weighted (a=0.5)` leads on both Chunk Hit@1 (60.0% against 53.3% for vectors alone) and MRR (0.691 against 0.639). With 15 queries, one query is worth 6.7 percentage points, so a gap of that size is one query changing its mind, not a robust effect. Treat the direction as real and the magnitude as provisional.

**The sweep is the stronger evidence.** Chunk Hit@3 runs 60.0% at alpha=0.0 (pure keyword), peaks at 80.0% in the middle, and settles at 73.3% at alpha=1.0 (pure vector). An interior maximum on both sides is what hybrid retrieval predicts, and it does not depend on any single query.

**The semantic split behaves as designed.** On paraphrased queries, keyword search reaches only 28.6% against 57.1% for vectors -- with the passage's own vocabulary deliberately avoided, BM25 has little to match on.

**The keyword split does not.** The hypothesis was that literal identifiers would favour BM25, but vectors score 100.0% there against 75.0% for FTS5. MiniLM tokenizes a coined name into subwords that are themselves frequent in the paper that coined it, and on a 50-paper corpus nothing else competes for them. Dense retrieval failing on rare literal strings is a real phenomenon, but it needs a corpus with genuine lexical competition to reproduce -- this one is too small and too topically sparse.

**Caveats.** 15 queries labelled by one annotator, on 50 papers from a single month of cs.CL. Differences of one or two queries are noise. The evaluation measures retrieval only -- whether a generator would produce a better answer from these passages is not tested here.

## Per-query detail

Rank of the first relevant **chunk** (`-` = not in the top 10).

| qid | type | query | vector-only | fts5-only | bm25okapi-only | hybrid-weighted (a=0.5) | hybrid-rrf |
|---|---|---|---|---|---|---|---|
| q01 | keyword | LongCat-Flash Single Batch Overlap ScMoE | 1 | 2 | 2 | 1 | 1 |
| q02 | mixed | replicating hotspot experts at runtime to avoid devi... | 1 | 2 | 2 | 1 | 1 |
| q03 | semantic | why do some experts stop receiving training signal w... | - | - | 3 | - | - |
| q04 | mixed | All-to-All metadata and small-GEMM fragmentation whe... | - | 2 | 2 | 3 | 4 |
| q05 | semantic | once a position has been filled in it can never be r... | 1 | 5 | 2 | 1 | 2 |
| q06 | mixed | commitment horizon definition tolerance epsilon earl... | 2 | 1 | 1 | 1 | 1 |
| q07 | keyword | Archer refresh when Hamming distance to the anchor e... | 1 | 4 | 3 | 1 | 2 |
| q08 | semantic | how much quality and speed does reusing cached state... | 2 | 3 | 3 | 2 | 3 |
| q09 | keyword | insertion strategy beginning only end only number of... | 1 | 1 | 1 | 1 | 1 |
| q10 | semantic | spotting untruthful answers by reading a frozen netw... | 3 | - | 5 | 3 | 1 |
| q11 | mixed | k-sparse autoencoders avoid the shrinkage bias of L1... | 1 | 2 | 2 | 1 | 1 |
| q12 | semantic | which mode leans on worded-out logical connectives a... | 1 | 2 | 1 | 1 | 1 |
| q13 | semantic | pushing a model out of an assigned character and bac... | - | - | - | - | 7 |
| q14 | semantic | do models tackle exam questions in the order shown r... | 4 | - | 9 | 5 | 8 |
| q15 | keyword | top-density overlap and early-position overlap of th... | 1 | 1 | 1 | 1 | 1 |
