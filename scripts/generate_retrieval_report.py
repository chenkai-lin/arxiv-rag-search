"""Generate reports/retrieval_report.md by running the same demo queries as
notebooks/retrieval_demo.ipynb through the shared SearchEngine.

Usage:
    python scripts/generate_retrieval_report.py
"""
from __future__ import annotations

from pathlib import Path

from search_lib import SearchEngine

ROOT = Path(__file__).resolve().parent.parent
REPORT_PATH = ROOT / "reports" / "retrieval_report.md"

QUERIES = [
    "How can hidden states be used to detect hallucinations in large language models?",
    "What are the routing and load balancing strategies in Mixture-of-Experts architectures?",
    "How do reasoning models allocate test-time compute across different questions?",
    "What techniques help with low-resource and multilingual machine translation?",
    "How can gender bias be mitigated in machine translation systems?",
]


def main() -> None:
    engine = SearchEngine()

    lines = [
        "# Retrieval Report",
        "",
        f"5 example queries against the arXiv cs.CL FAISS index "
        f"({engine.index.ntotal} chunks from 50 papers, reference sections stripped). "
        "Distance is FAISS L2 over `all-MiniLM-L6-v2` embeddings -- lower is more similar. "
        "Regenerate with `python scripts/generate_retrieval_report.py`.",
        "",
    ]

    for i, query in enumerate(QUERIES, start=1):
        results = engine.search(query, k=3)
        lines.append(f"## Query {i}: {query}")
        lines.append("")
        for r in results:
            snippet = " ".join(r["text"].split())[:400]
            lines.append(f"**[{r['rank']}] `{r['chunk_id']}`** (distance={r['distance']:.4f})")
            lines.append("")
            lines.append(f"> {snippet}...")
            lines.append("")

    lines.append("## Known limitation")
    lines.append("")
    lines.append(
        "Query 1's rank-1 and rank-2 results (`2608.07525_12`, `2608.07525_15`) are "
        "citation-list fragments, not body text -- only rank-3 is a genuine passage. "
        "`extract_text.py` only strips a References/Bibliography heading found past 30% into "
        "the document (`_MIN_REFERENCES_POSITION`), to avoid cutting real body text when an "
        "early match is a table-of-contents entry or cross-reference. In this one paper the "
        "heading legitimately sits at 26% because a large appendix follows the reference list, "
        "so the guard skips it and the reference list stays in the corpus. Affects 1/50 papers "
        "in this corpus; the other 46 papers with a detected heading were stripped correctly "
        "(see the commit that introduced `strip_references()`)."
    )
    lines.append("")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
