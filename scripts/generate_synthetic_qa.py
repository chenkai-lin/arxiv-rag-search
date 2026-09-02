"""Generate synthetic academic Q&A pairs from the paper corpus with an LLM (Week 7).

Week 4-5 turned these papers into a retrieval index. Week 7 uses the same corpus
for the opposite purpose: instead of looking knowledge up at query time, we distil
it into training data and fine-tune it into the weights.

Two decisions worth defending:

*Context is abstract + a digest of every section*, not the abstract alone and not
the whole paper. The abstract alone yields shallow questions that all sound the
same; the whole paper does not fit in a prompt and costs a fortune across 100
papers. Instead `build_paper_context()` takes the clean abstract from
`metadata.json` and appends the opening of each detected section. Sections are
found by walking candidate headings and keeping one only when its number is the
next one expected -- a numbered list item or a table row breaks the sequence and
is dropped, which is what separates "3 Approach" from "25 Table 2: Statistics".
77/100 papers parse this way; the rest fall back to head/middle/tail slices.

*The provider is configuration, not code.* The assignment says "use GPT-4", but
nothing here is OpenAI-specific, and paying per token is a poor reason to be
locked to one vendor. `LLM_PROVIDER` picks between the OpenAI-compatible path --
which covers OpenAI, DeepSeek, Qwen/DashScope, OpenRouter, and any local Ollama
or vLLM server, because they all speak the same /chat/completions shape -- and
Anthropic's native Messages API, which does not.

Output is `data/sft/raw_qa.json`, one record per paper. That file is the handoff
to `build_sft_dataset.py`, which is what actually writes the training JSONL. The
split exists so that the generator can be swapped (or replaced by hand-written
Q&A) without touching the formatting, splitting and dedup logic.

Resumable: papers already present in the output file are skipped, so an
interrupted run -- or a rate limit halfway through -- costs only what it got.

Usage:
    python scripts/generate_synthetic_qa.py --dump-contexts   # no API calls, inspect prompts first
    python scripts/generate_synthetic_qa.py --limit 3         # cheap smoke test
    python scripts/generate_synthetic_qa.py                   # the full 100-paper run
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
METADATA_PATH = ROOT / "data" / "texts" / "metadata.json"
TEXT_DIR = ROOT / "data" / "texts"
SFT_DIR = ROOT / "data" / "sft"
RAW_QA_PATH = SFT_DIR / "raw_qa.json"
CONTEXTS_PATH = SFT_DIR / "paper_contexts.json"

QA_PER_PAPER = 5            # 4 answerable + 1 edge case, per the assignment's ~500 total
EDGE_CASES_PER_PAPER = 1
CONTEXT_CHARS = 6000        # body budget per paper, on top of the abstract
MIN_SECTION_CHARS = 400     # never slice a section thinner than this
SLEEP_SECONDS = 1.0         # be polite to whichever endpoint is configured

# --- LLM provider configuration -------------------------------------------
#
# Every knob below is an environment variable, read once in `llm_settings()`.
# Put them in a `.env` file at the repo root (already gitignored) and they are
# picked up automatically; a real shell environment variable wins over the file.
#
#   LLM_PROVIDER   Which client to use. "openai" (default) or "anthropic".
#                  "openai" here means *the OpenAI wire protocol*, not the
#                  company -- most vendors and every local server implement it.
#   LLM_MODEL      Model id, passed through verbatim. No default that costs
#                  money is assumed; see the table below.
#   LLM_API_KEY    The credential. Falls back to OPENAI_API_KEY /
#                  ANTHROPIC_API_KEY so an existing setup keeps working.
#   LLM_BASE_URL   Endpoint override. Leave unset for OpenAI or Anthropic
#                  proper; set it to point at anyone else.
#   LLM_TEMPERATURE  Default 0.7. Some diversity helps -- 5 questions at
#                    temperature 0 tend to rephrase each other.
#   LLM_MAX_TOKENS   Default 2000. Enough for 5 Q&A pairs as JSON.
#
# Known-good combinations:
#
#   Vendor        LLM_PROVIDER  LLM_BASE_URL                              LLM_MODEL
#   ------------  ------------  ----------------------------------------  ----------------------
#   OpenAI        openai        (unset)                                   gpt-4o
#   Anthropic     anthropic     (unset)                                   claude-sonnet-5
#   DeepSeek      openai        https://api.deepseek.com/v1               deepseek-chat
#   Qwen          openai        https://dashscope.aliyuncs.com/compatible-mode/v1   qwen-plus
#   Moonshot      openai        https://api.moonshot.cn/v1                moonshot-v1-8k
#   OpenRouter    openai        https://openrouter.ai/api/v1              anthropic/claude-sonnet-4
#   Ollama        openai        http://localhost:11434/v1                 llama3.1  (LLM_API_KEY=ollama)
#   vLLM          openai        http://localhost:8000/v1                  <served model name>
#
# Local servers still want *some* key string even though they ignore it, hence
# the `or "not-needed"` fallback in `llm_settings()`.
#
DEFAULT_MODELS = {"openai": "gpt-4o", "anthropic": "claude-sonnet-5"}

PROMPT_TEMPLATE = """You are a research assistant who reads academic papers and creates quiz questions.

Below is the abstract and key sections of a research paper. Read them and generate \
exactly {n_qa} question-answer pairs that a student might ask after reading this paper.

Requirements:
- {n_answerable} of the pairs must be answerable from the text. Cover the paper's key \
points: its problem, method, findings and numbers. Mix factual questions (what was \
measured, what the result was) with conceptual ones (why the approach works, what a \
term means).
- Answers must be detailed, 2-4 sentences, and grounded only in the text below. Use the \
paper's own terminology. Never invent a number that is not stated.
- Each question must stand alone. Do not write "the paper" or "this study" without \
naming what it is about, because the question will be asked without the text in front \
of the model.
- The final {n_edge} pair must be an EDGE CASE: a question with a false premise, or one \
asking for a detail the paper does not contain. Its answer must correct the premise or \
state plainly that the paper does not report it, and then say what the paper does cover \
instead. Do not fabricate an answer to it.
- Avoid trivial or ambiguous questions.

Paper:
{context}

Output a JSON array of exactly {n_qa} objects, each with the fields "question", \
"answer", and "type". Use "type": "factual" or "conceptual" for the answerable pairs \
and "type": "edge_case" for the final one. Output the JSON array only, with no \
markdown fence and no commentary."""

# A heading is its number alone on a line, then the title on the next line -- what
# a two-column PDF flattens to. Anchored to line starts so mid-sentence digits miss.
_HEADING_RE = re.compile(r"^[ \t]*(\d{1,2})[.)]?[ \t]*\n[ \t]*([A-Z][^\n]{2,70})[ \t]*$", re.M)
_SENTENCE_END = re.compile(r"[.!?:;]\s*$")
_JSON_ARRAY = re.compile(r"\[.*\]", re.S)


def load_metadata() -> list[dict]:
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


def llm_settings(provider: str, model: str | None) -> dict:
    """Resolve provider settings from the environment (and .env, if present)."""
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:  # python-dotenv is optional; real env vars still work
        pass

    fallback_key = "OPENAI_API_KEY" if provider == "openai" else "ANTHROPIC_API_KEY"
    return {
        "provider": provider,
        "model": model or os.getenv("LLM_MODEL") or DEFAULT_MODELS[provider],
        "api_key": os.getenv("LLM_API_KEY") or os.getenv(fallback_key) or "not-needed",
        "base_url": os.getenv("LLM_BASE_URL") or None,
        "temperature": float(os.getenv("LLM_TEMPERATURE", "0.7")),
        "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "2000")),
    }


def dewrap(text: str) -> str:
    """Undo the PDF's hard line wrapping so sentences read as sentences."""
    lines = [ln.strip() for ln in text.split("\n")]
    out: list[str] = []
    for line in lines:
        if not line:
            out.append("\n")
        elif out and out[-1] != "\n" and not _SENTENCE_END.search(out[-1]):
            out[-1] = f"{out[-1].rstrip('-')} {line}" if not out[-1].endswith("-") else out[-1][:-1] + line
        else:
            out.append(line)
    return " ".join(out).replace(" \n ", "\n").strip()


def find_sections(text: str) -> list[tuple[str, str]]:
    """Return [(title, body)] for headings numbered 1, 2, 3, ... in order.

    Keeping a heading only when its number is the one expected next is what
    rejects '25 Table 2: Statistics of ...' while accepting '3 Approach'.
    """
    marks: list[tuple[str, int]] = []
    expected = 1
    for m in _HEADING_RE.finditer(text):
        if int(m.group(1)) != expected:
            continue
        marks.append((m.group(2).strip(), m.end()))
        expected += 1

    sections = []
    for i, (title, start) in enumerate(marks):
        end = marks[i + 1][1] - 200 if i + 1 < len(marks) else len(text)
        sections.append((title, text[start:max(start, end)]))
    return sections


def build_paper_context(paper: dict, body: str, budget: int = CONTEXT_CHARS) -> str:
    """Abstract plus a budgeted digest of the paper's sections."""
    parts = [
        f"Title: {paper['title']}",
        f"Abstract: {' '.join(paper['abstract'].split())}",
    ]

    sections = find_sections(body)
    if len(sections) >= 3:
        # Head of every section, so a method section named after the system it
        # introduces ("4 SHADOWBENCH") is kept as readily as one named "3 Method".
        per_section = max(MIN_SECTION_CHARS, budget // len(sections))
        for title, section_body in sections:
            excerpt = dewrap(section_body)[:per_section]
            if excerpt:
                parts.append(f"[{title}] {excerpt}")
    else:
        # No parseable structure: three slices still beat the first N characters,
        # which on these PDFs is mostly the author block and the abstract again.
        clean = dewrap(body)
        third = budget // 3
        mid = len(clean) // 2
        for label, slice_ in (
            ("Opening", clean[:third]),
            ("Middle", clean[mid:mid + third]),
            ("End", clean[-third:]),
        ):
            if slice_.strip():
                parts.append(f"[{label}] {slice_}")

    return "\n\n".join(parts)


def build_prompt(context: str, n_qa: int = QA_PER_PAPER, n_edge: int = EDGE_CASES_PER_PAPER) -> str:
    return PROMPT_TEMPLATE.format(
        n_qa=n_qa, n_edge=n_edge, n_answerable=n_qa - n_edge, context=context
    )


def call_llm(prompt: str, settings: dict) -> str:
    """Send one prompt, return the raw text reply."""
    if settings["provider"] == "anthropic":
        import anthropic

        client = anthropic.Anthropic(
            api_key=settings["api_key"], base_url=settings["base_url"]
        )
        reply = client.messages.create(
            model=settings["model"],
            max_tokens=settings["max_tokens"],
            temperature=settings["temperature"],
            messages=[{"role": "user", "content": prompt}],
        )
        return reply.content[0].text

    from openai import OpenAI

    client = OpenAI(api_key=settings["api_key"], base_url=settings["base_url"])
    reply = client.chat.completions.create(
        model=settings["model"],
        max_tokens=settings["max_tokens"],
        temperature=settings["temperature"],
        messages=[{"role": "user", "content": prompt}],
    )
    return reply.choices[0].message.content


def parse_qa_response(raw: str) -> list[dict]:
    """Pull the Q&A array out of a reply, tolerating fences and stray prose."""
    match = _JSON_ARRAY.search(raw)
    if not match:
        raise ValueError("no JSON array in the reply")
    items = json.loads(match.group(0))

    pairs = []
    for item in items:
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        if not question or not answer:
            continue
        qa_type = str(item.get("type", "factual")).strip().lower()
        pairs.append(
            {
                "question": question,
                "answer": answer,
                "type": qa_type if qa_type in {"factual", "conceptual", "edge_case"} else "factual",
            }
        )
    if not pairs:
        raise ValueError("the reply parsed but held no usable pairs")
    return pairs


def load_existing() -> dict[str, dict]:
    if not RAW_QA_PATH.exists():
        return {}
    payload = json.loads(RAW_QA_PATH.read_text(encoding="utf-8"))
    return {rec["arxiv_id"]: rec for rec in payload.get("papers", [])}


def save_raw_qa(records: dict[str, dict], generator: str) -> None:
    payload = {
        "_readme": [
            "Synthetic academic Q&A generated from the week 4-5 arXiv cs.CL corpus.",
            "",
            f"One record per paper, {QA_PER_PAPER} pairs each: {QA_PER_PAPER - EDGE_CASES_PER_PAPER}",
            "answerable from the paper plus 1 edge case whose premise is false or whose",
            "detail the paper never reports. The edge cases are here so the fine-tuned",
            "model learns to say 'the paper does not report that' instead of inventing a",
            "number -- a fine-tune on answerable questions alone teaches the opposite.",
            "",
            "Context fed to the generator was the abstract plus a digest of each section;",
            "see build_paper_context() in scripts/generate_synthetic_qa.py.",
            "",
            "This is the generator's output, not the training file. Run",
            "scripts/build_sft_dataset.py to turn it into the chat-formatted JSONL.",
        ],
        "generator": generator,
        "papers": sorted(records.values(), key=lambda r: r["arxiv_id"]),
    }
    RAW_QA_PATH.parent.mkdir(parents=True, exist_ok=True)
    RAW_QA_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_body(arxiv_id: str) -> str:
    path = TEXT_DIR / f"{arxiv_id}.txt"
    return path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""


def dump_contexts(papers: list[dict], budget: int) -> None:
    """Write every prompt context without calling a model.

    Worth running first: it is free, and it shows exactly what the model will be
    charged to read.
    """
    contexts = [
        {
            "arxiv_id": p["arxiv_id"],
            "title": p["title"],
            "context": build_paper_context(p, read_body(p["arxiv_id"]), budget),
        }
        for p in papers
    ]
    CONTEXTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONTEXTS_PATH.write_text(json.dumps(contexts, indent=2, ensure_ascii=False), encoding="utf-8")

    sizes = [len(c["context"]) for c in contexts]
    print(f"Built {len(contexts)} contexts")
    print(f"  chars: min {min(sizes)}, median {sorted(sizes)[len(sizes) // 2]}, max {max(sizes)}")
    print(f"  ~{sum(sizes) // 4:,} prompt tokens for the whole corpus (rough 4 chars/token)")
    print(f"Saved -> {CONTEXTS_PATH}")


def main(
    provider: str,
    model: str | None,
    limit: int | None,
    budget: int,
    sleep_seconds: float,
    dump_only: bool,
) -> None:
    papers = load_metadata()
    if limit is not None:
        papers = papers[:limit]

    if dump_only:
        dump_contexts(papers, budget)
        return

    settings = llm_settings(provider, model)
    records = load_existing()
    todo = [p for p in papers if p["arxiv_id"] not in records]

    print(f"Found {len(papers)} papers, {len(records)} already generated, {len(todo)} to go")
    print(f"Provider {settings['provider']}, model {settings['model']}"
          f"{', base_url ' + settings['base_url'] if settings['base_url'] else ''}")

    done = 0
    for paper in todo:
        arxiv_id = paper["arxiv_id"]
        body = read_body(arxiv_id)
        prompt = build_prompt(build_paper_context(paper, body, budget))
        try:
            pairs = parse_qa_response(call_llm(prompt, settings))
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {arxiv_id}: generation failed ({exc})", flush=True)
            continue

        records[arxiv_id] = {
            "arxiv_id": arxiv_id,
            "title": paper["title"],
            "generator": settings["model"],
            "qa_pairs": pairs,
        }
        done += 1
        # Save every time: a rate limit at paper 80 should not cost the first 79.
        save_raw_qa(records, settings["model"])
        print(f"  [{done}/{len(todo)}] {arxiv_id}: {len(pairs)} pairs "
              f"({sum(1 for p in pairs if p['type'] == 'edge_case')} edge)", flush=True)
        time.sleep(sleep_seconds)

    total = sum(len(r["qa_pairs"]) for r in records.values())
    print(f"\nGenerated {done} papers this run, {len(records)} papers and {total} pairs in total")
    print(f"Saved -> {RAW_QA_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["openai", "anthropic"], default=os.getenv("LLM_PROVIDER", "openai"))
    parser.add_argument("--model", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--context-chars", type=int, default=CONTEXT_CHARS)
    parser.add_argument("--sleep-seconds", type=float, default=SLEEP_SECONDS)
    parser.add_argument("--dump-contexts", action="store_true")
    args = parser.parse_args()
    main(args.provider, args.model, args.limit, args.context_chars, args.sleep_seconds, args.dump_contexts)
