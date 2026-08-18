"""Compare vector-only, keyword-only and hybrid retrieval (Week 5).

Runs the 12 labelled queries in data/eval/queries.json through five retrieval
configurations and reports hit rate at 3 and 5 plus MRR, both overall and
broken down by query type. Also sweeps the fusion weight alpha from 0 to 1 to
show the trade-off curve between the two legs.

Relevance is paper-level: a query is a hit at k if any of the top-k chunks
belongs to one of its `relevant_paper_ids`.

Usage:
    python scripts/evaluate_retrieval.py
    python scripts/evaluate_retrieval.py --k 3 --pool 50
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from hybrid_lib import DEFAULT_POOL, HybridSearchEngine

ROOT = Path(__file__).resolve().parent.parent
QUERIES_PATH = ROOT / "data" / "eval" / "queries.json"
REPORT_PATH = ROOT / "reports" / "hybrid_eval_report.md"

# alpha=1.0 is pure vector, alpha=0.0 is pure keyword; the sweep runs the whole
# range so the two endpoints double as a check that fusion degrades gracefully.
ALPHA_SWEEP = [round(i / 10, 1) for i in range(11)]

MRR_DEPTH = 10


def load_queries(path: Path = QUERIES_PATH) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["queries"]


def build_methods(engine: HybridSearchEngine, alpha: float, pool: int) -> dict:
    """Name -> callable(query, k) -> results. Five configurations to compare."""
    return {
        "vector-only": lambda q, k: engine.search_vector_only(q, k=k, pool=pool),
        "fts5-only": lambda q, k: engine.search_keyword_only(q, k=k, kw_backend="fts5", pool=pool),
        "bm25okapi-only": lambda q, k: engine.search_keyword_only(q, k=k, kw_backend="bm25", pool=pool),
        f"hybrid-weighted (a={alpha})": lambda q, k: engine.search(
            q, k=k, method="weighted", alpha=alpha, kw_backend="fts5", pool=pool
        ),
        "hybrid-rrf": lambda q, k: engine.search(
            q, k=k, method="rrf", kw_backend="fts5", pool=pool
        ),
    }


def rank_of_first_hit(results: list[dict], relevant: set[str]) -> int | None:
    """1-based rank of the first chunk from a relevant paper, or None."""
    for result in results:
        if result["paper_id"] in relevant:
            return result["rank"]
    return None


def evaluate(method_fn, queries: list[dict], k: int) -> dict:
    """Hit@3, Hit@5 and MRR for one retrieval configuration."""
    depth = max(k, 5, MRR_DEPTH)
    per_query = []
    for spec in queries:
        relevant = set(spec["relevant_paper_ids"])
        results = method_fn(spec["query"], depth)
        hit_rank = rank_of_first_hit(results, relevant)
        per_query.append({
            "qid": spec["qid"],
            "type": spec["type"],
            "query": spec["query"],
            "hit_rank": hit_rank,
            f"hit@{k}": hit_rank is not None and hit_rank <= k,
            "hit@5": hit_rank is not None and hit_rank <= 5,
            # Reciprocal rank, truncated at MRR_DEPTH: rewards ranking the
            # right paper first, not merely somewhere in the list.
            "rr": 1.0 / hit_rank if hit_rank and hit_rank <= MRR_DEPTH else 0.0,
            "top_paper": results[0]["paper_id"] if results else None,
        })

    n = len(per_query)
    summary = {
        f"hit@{k}": sum(r[f"hit@{k}"] for r in per_query) / n,
        "hit@5": sum(r["hit@5"] for r in per_query) / n,
        "mrr": sum(r["rr"] for r in per_query) / n,
    }

    by_type: dict[str, float] = {}
    grouped = defaultdict(list)
    for r in per_query:
        grouped[r["type"]].append(r)
    for qtype, rows in grouped.items():
        by_type[qtype] = sum(r[f"hit@{k}"] for r in rows) / len(rows)

    return {"summary": summary, "by_type": by_type, "per_query": per_query}


def sweep_alpha(engine, queries, k, pool, kw_backend="fts5") -> list[tuple[float, float]]:
    """Hit@k as the fusion weight shifts from pure keyword to pure vector."""
    out = []
    for alpha in ALPHA_SWEEP:
        report = evaluate(
            lambda q, kk, a=alpha: engine.search(
                q, k=kk, method="weighted", alpha=a, kw_backend=kw_backend, pool=pool
            ),
            queries,
            k,
        )
        out.append((alpha, report["summary"][f"hit@{k}"]))
    return out


def fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def render_report(results: dict, sweep: list[tuple[float, float]], queries: list[dict],
                  k: int, pool: int, best_alpha: float) -> str:
    lines = [
        "# Hybrid Retrieval Evaluation",
        "",
        f"{len(queries)} labelled queries against the arXiv cs.CL index "
        f"(914 chunks from 50 papers). Relevance is paper-level: a query counts as a hit "
        f"if any top-{k} chunk comes from a paper listed in "
        "`data/eval/queries.json`. Each leg contributes "
        f"{pool} candidates before fusion. Regenerate with "
        "`python scripts/evaluate_retrieval.py`.",
        "",
        "## Overall",
        "",
        f"| Method | Hit@{k} | Hit@5 | MRR@{MRR_DEPTH} |",
        "|---|---|---|---|",
    ]
    for name, report in results.items():
        s = report["summary"]
        lines.append(
            f"| {name} | {fmt_pct(s[f'hit@{k}'])} | {fmt_pct(s['hit@5'])} | {s['mrr']:.3f} |"
        )

    qtypes = sorted({q["type"] for q in queries})
    lines += [
        "",
        "## By query type",
        "",
        "Where each method's advantage actually comes from.",
        "",
        f"| Method | " + " | ".join(f"{t} (Hit@{k})" for t in qtypes) + " |",
        "|---" * (len(qtypes) + 1) + "|",
    ]
    for name, report in results.items():
        cells = [fmt_pct(report["by_type"].get(t, 0.0)) for t in qtypes]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Fusion weight sweep",
        "",
        "`alpha` weights the vector leg: `score = alpha * vector + (1 - alpha) * keyword`. "
        "alpha=0.0 is pure keyword, alpha=1.0 is pure vector.",
        "",
        f"| alpha | Hit@{k} |",
        "|---|---|",
    ]
    for alpha, score in sweep:
        marker = "  <-- best" if alpha == best_alpha else ""
        lines.append(f"| {alpha} | {fmt_pct(score)}{marker} |")

    lines += [
        "",
        "## Per-query detail",
        "",
        "Rank of the first relevant chunk (`-` = not found in the top "
        f"{max(k, 5, MRR_DEPTH)}).",
        "",
        "| qid | type | query | " + " | ".join(results) + " |",
        "|---" * (len(results) + 3) + "|",
    ]
    for i, spec in enumerate(queries):
        cells = []
        for report in results.values():
            rank = report["per_query"][i]["hit_rank"]
            cells.append(str(rank) if rank else "-")
        query_text = spec["query"] if len(spec["query"]) <= 60 else spec["query"][:57] + "..."
        lines.append(
            f"| {spec['qid']} | {spec['type']} | {query_text} | " + " | ".join(cells) + " |"
        )

    lines.append("")
    return "\n".join(lines)


def main(k: int, pool: int, alpha: float, report_path: Path) -> None:
    queries = load_queries()
    print(f"Loaded {len(queries)} evaluation queries")

    engine = HybridSearchEngine()
    print(f"Engine: {engine.stats()}\n")

    print(f"Sweeping alpha to pick the best fusion weight (Hit@{k})...")
    sweep = sweep_alpha(engine, queries, k, pool)
    for a, score in sweep:
        print(f"  alpha={a:.1f}  Hit@{k}={fmt_pct(score)}")
    # Ties go to the smallest alpha that achieves the best score, i.e. the
    # setting that leans on keywords no more than it has to.
    best_alpha = max(sweep, key=lambda item: (item[1], -item[0]))[0]
    print(f"  best alpha = {best_alpha}\n")

    # Report the requested alpha, but tell the reader if the sweep found better.
    methods = build_methods(engine, alpha, pool)
    if best_alpha != alpha:
        methods[f"hybrid-weighted (a={best_alpha})"] = lambda q, kk: engine.search(
            q, k=kk, method="weighted", alpha=best_alpha, kw_backend="fts5", pool=pool
        )

    results = {}
    for name, fn in methods.items():
        results[name] = evaluate(fn, queries, k)
        s = results[name]["summary"]
        print(f"{name:32} Hit@{k}={fmt_pct(s[f'hit@{k}']):>7}  "
              f"Hit@5={fmt_pct(s['hit@5']):>7}  MRR={s['mrr']:.3f}")

    # Endpoint sanity check: with the keyword leg switched off entirely, hybrid
    # must reproduce the vector baseline. If this drifts, the fusion or the
    # normalization is wrong.
    endpoint = evaluate(
        lambda q, kk: engine.search(q, k=kk, method="weighted", alpha=1.0, pool=pool),
        queries, k,
    )
    assert endpoint["summary"] == results["vector-only"]["summary"], (
        "alpha=1.0 does not reproduce vector-only -- fusion logic is broken"
    )
    print("\n[check] alpha=1.0 reproduces the vector-only baseline")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        render_report(results, sweep, queries, k, pool, best_alpha), encoding="utf-8"
    )
    print(f"Saved -> {report_path}")
    engine.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=3, help="cutoff for the headline hit rate")
    parser.add_argument("--pool", type=int, default=DEFAULT_POOL, help="candidates per leg")
    parser.add_argument("--alpha", type=float, default=0.5, help="fusion weight to report")
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    args = parser.parse_args()
    main(args.k, args.pool, args.alpha, args.report)
