"""Turn the generated Q&A into the instruction-tuning JSONL a trainer wants (Week 7).

Kept separate from `generate_synthetic_qa.py` on purpose. Generation is slow,
costs money and depends on whichever vendor is configured; formatting is fast,
free and deterministic. Splitting them means the chat template, the dedup rules
and the train/val split can be changed and re-run in a second, and it means
hand-written Q&A can enter the pipeline through the same door as generated Q&A
as long as it matches the `raw_qa.json` schema.

Three things happen here that are easy to get wrong by hand:

*Dedup is on the normalised question.* Asking a model the same question twice
with two different gold answers teaches it that the answer is arbitrary. Across
100 independently prompted papers, generic questions ("What is the main
contribution?") do collide.

*The split is grouped by paper, not by pair.* Five questions about one paper
share its facts almost completely, so splitting at pair level would put a near
copy of every validation item into training and report a validation loss that
means nothing. Whole papers go to one side or the other.

*The chat template is applied once, here.* `<|system|>`, `<|user|>` and
`<|assistant|>` are written into a single `text` field because that is what
`SFTTrainer(dataset_text_field="text")` consumes. The provenance fields beside
it are ignored by the trainer and are there so a suspicious-looking answer can
be traced back to its paper.

Usage:
    python scripts/build_sft_dataset.py
    python scripts/build_sft_dataset.py --val-split 0.0     # train on everything
    python scripts/build_sft_dataset.py --seed 7 --val-split 0.2
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SFT_DIR = ROOT / "data" / "sft"
RAW_QA_PATH = SFT_DIR / "raw_qa.json"
TRAIN_PATH = SFT_DIR / "synthetic_qa.jsonl"
VAL_PATH = SFT_DIR / "synthetic_qa_val.jsonl"
STATS_PATH = SFT_DIR / "dataset_stats.json"

SYSTEM_PROMPT = "You are a helpful academic Q&A assistant specialized in scholarly content."
VAL_SPLIT = 0.1
SEED = 42

MIN_QUESTION_CHARS = 15
MIN_ANSWER_CHARS = 40
MAX_ANSWER_CHARS = 2000

_NORMALISE = re.compile(r"[^a-z0-9 ]+")


def load_raw_qa() -> dict:
    return json.loads(RAW_QA_PATH.read_text(encoding="utf-8"))


def normalise(question: str) -> str:
    """'What is  SA-PASS?' -> 'what is sapass', so near-duplicates collide."""
    return " ".join(_NORMALISE.sub("", question.lower()).split())


def format_example(question: str, answer: str) -> str:
    """The chat template the fine-tune trains on, and inference must mirror."""
    return f"<|system|>{SYSTEM_PROMPT}<|user|>{question}<|assistant|>{answer}"


def collect_records(payload: dict) -> tuple[list[dict], dict[str, int]]:
    """Flatten papers into records, dropping the ones that fail a quality gate."""
    records: list[dict] = []
    seen: set[str] = set()
    dropped = {"short_question": 0, "short_answer": 0, "long_answer": 0, "duplicate": 0}

    for paper in payload["papers"]:
        for qa in paper["qa_pairs"]:
            question, answer = qa["question"].strip(), qa["answer"].strip()

            if len(question) < MIN_QUESTION_CHARS:
                dropped["short_question"] += 1
                continue
            if len(answer) < MIN_ANSWER_CHARS:
                dropped["short_answer"] += 1
                continue
            if len(answer) > MAX_ANSWER_CHARS:
                dropped["long_answer"] += 1
                continue
            key = normalise(question)
            if key in seen:
                dropped["duplicate"] += 1
                continue
            seen.add(key)

            records.append(
                {
                    "text": format_example(question, answer),
                    "arxiv_id": paper["arxiv_id"],
                    "type": qa["type"],
                }
            )
    return records, dropped


def split_by_paper(records: list[dict], val_split: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Hold out whole papers, so no validation question has a sibling in training."""
    if val_split <= 0:
        return records, []

    paper_ids = sorted({r["arxiv_id"] for r in records})
    random.Random(seed).shuffle(paper_ids)
    n_val = max(1, round(len(paper_ids) * val_split))
    val_ids = set(paper_ids[:n_val])

    train = [r for r in records if r["arxiv_id"] not in val_ids]
    val = [r for r in records if r["arxiv_id"] in val_ids]
    return train, val


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def summarise(records: list[dict]) -> dict:
    lengths = sorted(len(r["text"]) for r in records)
    types: dict[str, int] = {}
    for r in records:
        types[r["type"]] = types.get(r["type"], 0) + 1
    return {
        "examples": len(records),
        "papers": len({r["arxiv_id"] for r in records}),
        "by_type": dict(sorted(types.items())),
        "text_chars": {
            "min": lengths[0],
            "median": lengths[len(lengths) // 2],
            "max": lengths[-1],
        } if lengths else {},
    }


def main(val_split: float, seed: int) -> None:
    payload = load_raw_qa()
    records, dropped = collect_records(payload)
    train, val = split_by_paper(records, val_split, seed)

    print(f"Loaded {len(payload['papers'])} papers from {RAW_QA_PATH.name}")
    print(f"  kept {len(records)} pairs, dropped {sum(dropped.values())} "
          f"({', '.join(f'{k}={v}' for k, v in dropped.items() if v)})"
          if sum(dropped.values()) else f"  kept {len(records)} pairs, dropped none")

    write_jsonl(train, TRAIN_PATH)
    if val:
        write_jsonl(val, VAL_PATH)

    stats = {
        "generator": payload.get("generator", "unknown"),
        "system_prompt": SYSTEM_PROMPT,
        "val_split": val_split,
        "seed": seed,
        "dropped": dropped,
        "train": summarise(train),
        "val": summarise(val),
    }
    STATS_PATH.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    # Sanity checks
    train_ids = {r["arxiv_id"] for r in train}
    val_ids = {r["arxiv_id"] for r in val}
    assert not (train_ids & val_ids), "a paper leaked across the train/val split"
    assert all(r["text"].startswith("<|system|>") for r in train), "an example is missing its template"
    assert len({r["text"] for r in records}) == len(records), "duplicate example survived dedup"

    print(f"\nTrain: {stats['train']['examples']} pairs over {stats['train']['papers']} papers "
          f"{stats['train']['by_type']}")
    if val:
        print(f"Val:   {stats['val']['examples']} pairs over {stats['val']['papers']} papers "
              f"{stats['val']['by_type']}")
    print(f"Saved -> {TRAIN_PATH}")
    if val:
        print(f"Saved -> {VAL_PATH}")
    print(f"Saved -> {STATS_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-split", type=float, default=VAL_SPLIT)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    main(args.val_split, args.seed)
