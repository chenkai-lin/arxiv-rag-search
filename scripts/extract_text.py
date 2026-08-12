"""Extract plain text from every downloaded PDF.

Usage:
    python scripts/extract_text.py
"""
from __future__ import annotations

from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = ROOT / "data" / "pdfs"
TEXT_DIR = ROOT / "data" / "texts"


def extract_text_from_pdf(pdf_path: Path) -> str:
    doc = pymupdf.open(pdf_path)
    pages = [page.get_text() for page in doc]
    doc.close()
    return "\n".join(pages).strip()


def main() -> None:
    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_paths = sorted(PDF_DIR.glob("*.pdf"))
    print(f"Found {len(pdf_paths)} PDFs in {PDF_DIR}")

    ok, failed = 0, []
    for pdf_path in pdf_paths:
        arxiv_id = pdf_path.stem
        text_path = TEXT_DIR / f"{arxiv_id}.txt"
        try:
            text = extract_text_from_pdf(pdf_path)
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL {arxiv_id}: {exc}")
            failed.append(arxiv_id)
            continue

        if len(text) < 500:
            print(f"  WARN {arxiv_id}: only {len(text)} chars extracted")
            failed.append(arxiv_id)
            continue

        text_path.write_text(text, encoding="utf-8")
        ok += 1
        print(f"  [{ok}] {arxiv_id}: {len(text)} chars -> {text_path.name}")

    print(f"\nExtracted {ok}/{len(pdf_paths)} papers -> {TEXT_DIR}")
    if failed:
        print(f"Failed/short: {failed}")


if __name__ == "__main__":
    main()
