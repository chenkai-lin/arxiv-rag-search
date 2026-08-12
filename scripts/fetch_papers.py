"""Download N arXiv cs.CL papers and write their metadata.

Resumable: safe to re-run after a rate-limit failure. Already-downloaded
PDFs and already-recorded metadata are skipped, and metadata is saved
incrementally so a crash never loses prior progress.

Usage:
    python scripts/fetch_papers.py --n 50
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import arxiv

ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = ROOT / "data" / "pdfs"
METADATA_PATH = ROOT / "data" / "texts" / "metadata.json"


def load_existing_metadata() -> list[dict]:
    if METADATA_PATH.exists():
        return json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return []


def save_metadata(metadata: list[dict]) -> None:
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def fetch(n: int, sleep_seconds: float, page_size: int, page_delay: float) -> None:
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    metadata = load_existing_metadata()
    seen_ids = {m["arxiv_id"] for m in metadata}
    print(f"Resuming with {len(metadata)} papers already recorded", flush=True)

    if len(metadata) >= n:
        print(f"Already have {len(metadata)}/{n}, nothing to do.")
        return

    # export.arxiv.org has been aggressively rate-limiting/blocking this
    # network (429s after a handful of requests, occasional timeouts).
    # Keep pages small and spaced well apart; on a page-fetch failure, save
    # what we have and exit cleanly (exit code 2) instead of crashing --
    # re-running the script picks up where it left off.
    client = arxiv.Client(page_size=page_size, delay_seconds=page_delay, num_retries=3)
    search = arxiv.Search(
        query="cat:cs.CL",
        max_results=n * 2,  # over-fetch a bit in case some downloads fail
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )

    try:
        for result in client.results(search):
            if len(metadata) >= n:
                break

            arxiv_id = result.get_short_id().split("v")[0]
            if arxiv_id in seen_ids:
                continue

            pdf_path = PDF_DIR / f"{arxiv_id}.pdf"
            if not pdf_path.exists():
                try:
                    result.download_pdf(dirpath=str(PDF_DIR), filename=pdf_path.name)
                    time.sleep(sleep_seconds)
                except Exception as exc:  # noqa: BLE001 - log and skip on any download failure
                    print(f"  skip {arxiv_id}: download failed ({exc})", flush=True)
                    continue

            if pdf_path.stat().st_size < 10_000:
                print(f"  skip {arxiv_id}: file too small, likely a failed download", flush=True)
                pdf_path.unlink(missing_ok=True)
                continue

            metadata.append(
                {
                    "arxiv_id": arxiv_id,
                    "title": result.title.strip(),
                    "authors": [a.name for a in result.authors],
                    "abstract": result.summary.strip(),
                    "published": result.published.isoformat(),
                    "pdf_file": pdf_path.name,
                    "url": result.entry_id,
                }
            )
            seen_ids.add(arxiv_id)
            save_metadata(metadata)  # persist after every success, not just at the end
            print(f"  [{len(metadata)}/{n}] {arxiv_id}: {result.title.strip()[:70]}", flush=True)
    except arxiv.HTTPError as exc:
        print(f"\nPage fetch failed ({exc}); stopping here with {len(metadata)}/{n} saved.", flush=True)
        print("Re-run this script later to top up the rest.", flush=True)
        save_metadata(metadata)
        raise SystemExit(2)

    save_metadata(metadata)
    print(f"\nSaved {len(metadata)} papers -> {PDF_DIR}")
    print(f"Metadata -> {METADATA_PATH}")

    if len(metadata) < n:
        print(f"WARNING: only got {len(metadata)}/{n} papers, re-run to top up")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--sleep-seconds", type=float, default=2.0)
    parser.add_argument("--page-size", type=int, default=10)
    parser.add_argument("--page-delay", type=float, default=20.0)
    args = parser.parse_args()
    fetch(args.n, args.sleep_seconds, args.page_size, args.page_delay)
