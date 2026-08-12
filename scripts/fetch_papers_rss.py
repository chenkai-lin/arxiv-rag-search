"""Fetch arXiv cs.CL papers via the RSS feed instead of the query API.

export.arxiv.org's query API rate-limits this network hard (persistent 429s
even on a first request), but the RSS feed and the PDF download endpoint
both respond normally. This script parses the cs.CL RSS feed for paper IDs
and metadata, then downloads PDFs directly.

Resumable: existing PDFs and already-recorded metadata are skipped, and
metadata is saved after every successful download.

Usage:
    python scripts/fetch_papers_rss.py --n 50
"""
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = ROOT / "data" / "pdfs"
METADATA_PATH = ROOT / "data" / "texts" / "metadata.json"

RSS_URL = "https://rss.arxiv.org/rss/cs.CL"
USER_AGENT = "arxiv-rag-search/0.1 (course project; contact via GitHub linchenk1992)"

_ID_RE = re.compile(r"arXiv:(\d{4}\.\d{4,5})")


def http_get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def parse_rss() -> list[dict]:
    """Return [{arxiv_id, title, authors, abstract}] from the cs.CL feed."""
    root = ET.fromstring(http_get(RSS_URL).decode("utf-8"))
    papers = []
    for item in root.iter("item"):
        description = (item.findtext("description") or "").strip()
        match = _ID_RE.search(description)
        if not match:
            continue
        abstract = description.split("Abstract:", 1)[-1].strip() if "Abstract:" in description else description
        creator = item.findtext("{http://purl.org/dc/elements/1.1/}creator") or ""
        papers.append(
            {
                "arxiv_id": match.group(1),
                "title": " ".join((item.findtext("title") or "").split()),
                "authors": [a.strip() for a in creator.split(",") if a.strip()],
                "abstract": abstract,
                "url": item.findtext("link") or f"https://arxiv.org/abs/{match.group(1)}",
            }
        )
    return papers


def load_existing_metadata() -> list[dict]:
    if METADATA_PATH.exists():
        return json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return []


def save_metadata(metadata: list[dict]) -> None:
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def main(n: int, sleep_seconds: float) -> None:
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    metadata = load_existing_metadata()
    seen_ids = {m["arxiv_id"] for m in metadata}
    print(f"Resuming with {len(metadata)} papers already recorded", flush=True)
    if len(metadata) >= n:
        print(f"Already have {len(metadata)}/{n}, nothing to do.")
        return

    candidates = parse_rss()
    print(f"RSS feed listed {len(candidates)} cs.CL papers", flush=True)

    for paper in candidates:
        if len(metadata) >= n:
            break
        arxiv_id = paper["arxiv_id"]
        if arxiv_id in seen_ids:
            continue

        pdf_path = PDF_DIR / f"{arxiv_id}.pdf"
        if not pdf_path.exists():
            try:
                pdf_path.write_bytes(http_get(f"https://arxiv.org/pdf/{arxiv_id}"))
                time.sleep(sleep_seconds)
            except Exception as exc:  # noqa: BLE001
                print(f"  skip {arxiv_id}: download failed ({exc})", flush=True)
                pdf_path.unlink(missing_ok=True)
                continue

        if pdf_path.stat().st_size < 10_000:
            print(f"  skip {arxiv_id}: file too small, likely a failed download", flush=True)
            pdf_path.unlink(missing_ok=True)
            continue

        paper["pdf_file"] = pdf_path.name
        metadata.append(paper)
        seen_ids.add(arxiv_id)
        save_metadata(metadata)
        print(f"  [{len(metadata)}/{n}] {arxiv_id}: {paper['title'][:70]}", flush=True)

    save_metadata(metadata)
    print(f"\nSaved {len(metadata)} papers -> {PDF_DIR}")
    if len(metadata) < n:
        print(f"WARNING: only got {len(metadata)}/{n} papers, re-run to top up")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--sleep-seconds", type=float, default=2.0)
    args = parser.parse_args()
    main(args.n, args.sleep_seconds)
