# Hybrid Retrieval Evaluation

12 labelled queries against the arXiv cs.CL index (914 chunks from 50 papers). Relevance is paper-level: a query counts as a hit if any top-3 chunk comes from a paper listed in `data/eval/queries.json`. Each leg contributes 50 candidates before fusion. Regenerate with `python scripts/evaluate_retrieval.py`.

## Overall

| Method | Hit@3 | Hit@5 | MRR@10 |
|---|---|---|---|
| vector-only | 83.3% | 83.3% | 0.792 |
| fts5-only | 83.3% | 83.3% | 0.833 |
| bm25okapi-only | 83.3% | 83.3% | 0.806 |
| hybrid-weighted (a=0.5) | 83.3% | 83.3% | 0.833 |
| hybrid-rrf | 83.3% | 83.3% | 0.858 |
| hybrid-weighted (a=0.0) | 83.3% | 83.3% | 0.833 |

## By query type

Where each method's advantage actually comes from.

| Method | keyword (Hit@3) | mixed (Hit@3) | semantic (Hit@3) |
|---|---|---|---|
| vector-only | 100.0% | 100.0% | 50.0% |
| fts5-only | 100.0% | 100.0% | 50.0% |
| bm25okapi-only | 100.0% | 100.0% | 50.0% |
| hybrid-weighted (a=0.5) | 100.0% | 100.0% | 50.0% |
| hybrid-rrf | 100.0% | 100.0% | 50.0% |
| hybrid-weighted (a=0.0) | 100.0% | 100.0% | 50.0% |

## Fusion weight sweep

`alpha` weights the vector leg: `score = alpha * vector + (1 - alpha) * keyword`. alpha=0.0 is pure keyword, alpha=1.0 is pure vector.

| alpha | Hit@3 |
|---|---|
| 0.0 | 83.3%  <-- best |
| 0.1 | 83.3% |
| 0.2 | 83.3% |
| 0.3 | 83.3% |
| 0.4 | 83.3% |
| 0.5 | 83.3% |
| 0.6 | 83.3% |
| 0.7 | 83.3% |
| 0.8 | 83.3% |
| 0.9 | 83.3% |
| 1.0 | 83.3% |

## Per-query detail

Rank of the first relevant chunk (`-` = not found in the top 10).

| qid | type | query | vector-only | fts5-only | bm25okapi-only | hybrid-weighted (a=0.5) | hybrid-rrf | hybrid-weighted (a=0.0) |
|---|---|---|---|---|---|---|---|---|
| q01 | semantic | how can we tell when a chatbot is making up facts it cann... | - | - | 6 | - | 8 | - |
| q02 | semantic | do reasoning systems spend the same effort on easy and ha... | 1 | 1 | 1 | 1 | 1 | 1 |
| q03 | semantic | stopping automatic translation from defaulting to male fo... | 1 | 1 | 2 | 1 | 1 | 1 |
| q04 | semantic | training a smaller network to imitate a larger one on its... | - | - | - | - | 6 | - |
| q05 | keyword | Search-G1 intrinsic rewards | 1 | 1 | 1 | 1 | 1 | 1 |
| q06 | keyword | AraSSM Arabic state-space encoder | 1 | 1 | 1 | 1 | 1 | 1 |
| q07 | keyword | WuYuEval benchmark | 1 | 1 | 1 | 1 | 1 | 1 |
| q08 | keyword | RA-FinBERT rule-aware LoRA financial sentiment | 1 | 1 | 1 | 1 | 1 | 1 |
| q09 | mixed | Mixture-of-Experts routing and load balancing | 1 | 1 | 1 | 1 | 1 | 1 |
| q10 | mixed | sparse autoencoders for interpreting chain-of-thought rea... | 1 | 1 | 1 | 1 | 1 | 1 |
| q11 | mixed | classifier-free guidance in masked diffusion language models | 2 | 1 | 1 | 1 | 1 | 1 |
| q12 | mixed | safety benchmark for Indian languages | 1 | 1 | 1 | 1 | 1 | 1 |
