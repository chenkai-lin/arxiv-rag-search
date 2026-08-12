"""Extract plain text from every downloaded PDF.

Usage:
    python scripts/extract_text.py
"""
from __future__ import annotations

import re
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = ROOT / "data" / "pdfs"
TEXT_DIR = ROOT / "data" / "texts"

# A standalone "References"/"Bibliography" line marks the end of the body text.
# Everything after it is a citation list: keyword-dense, so it scores well on
# semantic similarity, but useless as a retrieved passage. 49/50 papers in this
# corpus expose the heading this way.
_REFERENCES_HEADING = re.compile(r"^[ \t]*(references|bibliography)[ \t]*$", re.M | re.I)

# Only trust the heading if it appears late in the document -- an early match is
# more likely a table-of-contents entry or a cross-reference than the real list.
_MIN_REFERENCES_POSITION = 0.30


def strip_references(text: str) -> tuple[str, bool]:
    matches = [m for m in _REFERENCES_HEADING.finditer(text)]
    if not matches:
        return text, False
    last = matches[-1]
    if last.start() / len(text) < _MIN_REFERENCES_POSITION:
        return text, False
    return text[: last.start()].rstrip(), True


def extract_text_from_pdf(pdf_path: Path) -> tuple[str, bool]:
    doc = pymupdf.open(pdf_path)
    pages = [page.get_text() for page in doc]
    doc.close()
    return strip_references("\n".join(pages).strip())


def main() -> None:
    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_paths = sorted(PDF_DIR.glob("*.pdf"))
    print(f"Found {len(pdf_paths)} PDFs in {PDF_DIR}")

    ok, failed, stripped_count = 0, [], 0
    for pdf_path in pdf_paths:
        arxiv_id = pdf_path.stem
        text_path = TEXT_DIR / f"{arxiv_id}.txt"
        try:
            text, stripped = extract_text_from_pdf(pdf_path)
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
        stripped_count += stripped
        note = "" if stripped else "  (no references section found)"
        print(f"  [{ok}] {arxiv_id}: {len(text)} chars -> {text_path.name}{note}")

    print(f"\nExtracted {ok}/{len(pdf_paths)} papers -> {TEXT_DIR}")
    print(f"References section stripped from {stripped_count}/{ok}")
    if failed:
        print(f"Failed/short: {failed}")


if __name__ == "__main__":
    main()
