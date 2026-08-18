"""Tokenization shared by index construction and query time.

`build_sqlite.py` uses this to pick each paper's TF-IDF keywords, and
`hybrid_lib.py` uses it to turn a natural-language question into an FTS5 MATCH
expression and into rank_bm25 query terms. They have to agree: if the index
were built from one vocabulary and queried with another, terms would drop out
on one side only and the keyword comparison in the evaluation would be
measuring the tokenizer difference rather than the retrieval method.
"""
from __future__ import annotations

import re

STOPWORDS = {
    "a", "about", "above", "across", "after", "against", "all", "also", "an", "and",
    "any", "are", "as", "at", "be", "because", "been", "before", "being", "below",
    "between", "both", "but", "by", "can", "could", "did", "do", "does", "doing",
    "during", "each", "few", "for", "from", "further", "had", "has", "have", "having",
    "he", "her", "here", "hers", "him", "his", "how", "however", "i", "if", "in",
    "into", "is", "it", "its", "itself", "just", "may", "me", "might", "more", "most",
    "must", "my", "no", "nor", "not", "now", "of", "off", "on", "once", "only", "or",
    "other", "our", "ours", "out", "over", "own", "same", "shall", "she", "should",
    "so", "some", "such", "than", "that", "the", "their", "theirs", "them", "then",
    "there", "these", "they", "this", "those", "through", "to", "too", "under",
    "until", "up", "very", "was", "we", "were", "what", "when", "where", "which",
    "while", "who", "whom", "why", "will", "with", "would", "you", "your", "yours",
}

# Filler that is technically contentful but appears in nearly every cs.CL
# abstract and every question asked of this corpus, so it separates nothing.
# Dropped on top of STOPWORDS when extracting a paper's keywords; kept at query
# time, where a user typing "model" may genuinely mean it.
DOMAIN_STOPWORDS = {
    "using", "used", "use", "show", "shows", "propose", "proposed", "paper",
    "results", "result", "model", "models", "method", "methods", "approach",
    "approaches", "task", "tasks", "based", "work", "new", "study",
}

# Words, plus internal hyphens/underscores so "chain-of-thought", "Search-G1"
# and "load_balancing" survive as single terms instead of shattering into
# fragments that match far too much.
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*")


def to_terms(text: str, extra_stopwords: set[str] | None = None) -> list[str]:
    """Lowercase word tokens, with stopwords and single characters removed."""
    stop = STOPWORDS | (extra_stopwords or set())
    return [
        t for t in (m.group(0).lower() for m in _TOKEN_RE.finditer(text))
        if len(t) > 1 and t not in stop
    ]


def to_fts_query(query: str) -> str | None:
    """Turn a natural-language question into a safe FTS5 MATCH expression.

    FTS5 MATCH takes a query *language*, not free text. Passing a raw question
    through raises sqlite3.OperationalError on the punctuation ("...models?"),
    and a bare word like "not" or "and" is parsed as a boolean operator rather
    than a search term. So: tokenize, drop stopwords, wrap every term in double
    quotes (which forces it to be read as a literal string), and OR them
    together.

    OR rather than AND is deliberate -- requiring every term of a 12-word
    question to appear in one 512-token chunk would return almost nothing.
    With OR, BM25 still ranks chunks matching more (and rarer) terms highest,
    which is exactly the behaviour we want.

    Returns None when nothing survives filtering, so callers can skip the
    keyword leg instead of issuing an empty MATCH.
    """
    terms = to_terms(query)
    if not terms:
        return None
    # Deduplicate but keep order, so the generated expression is stable and
    # readable when it shows up in the evaluation report.
    seen: dict[str, None] = {}
    for t in terms:
        seen.setdefault(t, None)
    return " OR ".join(f'"{t}"' for t in seen)
