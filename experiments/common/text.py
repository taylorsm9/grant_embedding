"""Shared text cleaning and query/document construction (Plan 1 steps 1 and 5).

Every plan builds query and document text through this module so the boilerplate
rules, instruction template and length caps stay identical. Rules are documented in
experiments/common/boilerplate_rules.md.
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent

TASK = ("Given a funding opportunity announcement, retrieve abstracts of research "
        "projects funded under it")
# Model card format: no space after "Query:" (as in the model card's get_detailed_instruct).
QUERY_PROMPT = f"Instruct: {TASK}\nQuery:"
QUERY_CAP = 512   # tokens including the prompt and the appended <|endoftext|>
DOC_CAP = 768     # tokens including <|endoftext|>

# --- abstracts -------------------------------------------------------------

UNREADABLE = re.compile(r"\[unreadable\]", re.I)
# Leading section headers ("DESCRIPTION (provided by applicant):", "Project Summary/Abstract", ...),
# possibly repeated. They are era-specific, so they could leak the award year.
ABSTRACT_HEADER = re.compile(
    r"^\s*(?:(?:description|project\s+summary|summary|abstract|project\s+abstract|overall|"
    r"project\s+description)\s*(?:/\s*abstract)?\s*(?:\(provided\s+by\s+(?:the\s+)?applicant\))?"
    r"\s*(?:abstract)?\s*[:.\-]?\s*)+",
    re.I)
# Trailing relevance / narrative statements (start at ~80% of the text in >98% of cases).
ABSTRACT_TRAILER = re.compile(
    r"\s(?:public\s+health\s+relevance|relevance\s*\(see\s+instructions\)|project\s+narrative|"
    r"narrative)\s*(?:\(provided\s+by\s+(?:the\s+)?applicant\))?\s*:.*$",
    re.I | re.S)
END_OF_ABSTRACT = re.compile(r"\(end\s+of\s+abstract\)", re.I)


def squash(s):
    return re.sub(r"[ \t]+", " ", re.sub(r"\s*\n\s*", "\n", s)).strip()


def clean_abstract(text):
    t = UNREADABLE.sub(" ", text or "")
    t = END_OF_ABSTRACT.sub(" ", t)
    t = squash(t)
    t = ABSTRACT_HEADER.sub("", t)
    head = ABSTRACT_TRAILER.sub("", t)
    # Keep the trailer only if cutting it would leave almost nothing.
    if len(head) >= 0.4 * len(t):
        t = head
    return squash(t)


# --- FOAs ------------------------------------------------------------------

# Hidden template comments from the NIH Guide HTML that the parser kept as text.
PARSER_ARTIFACTS = re.compile(
    r"This variable defines that we need to start a new row\.|"
    r"Closing the div tags for a table element, if a vertical element starts after table element\.|"
    r"If there are no labels defined for table element, we treat that as 1 row and append the text\.|"
    r"</?div>|End Row",
    re.I)
SENT_SPLIT = re.compile(r"(?<=[.!?:])\s+|\n+")
_frequent = None


def norm_sentence(s):
    return re.sub(r"\s+", " ", s.strip().lower())


def foa_sentences(text):
    text = PARSER_ARTIFACTS.sub(" ", text or "")
    return [s.strip() for s in SENT_SPLIT.split(text) if s.strip()]


def frequent_sentences():
    """Normalised FOA sentences found in >= 2% of FOAs (built by build_text.py)."""
    global _frequent
    if _frequent is None:
        f = HERE / "foa_boilerplate_sentences.json"
        _frequent = set(json.loads(f.read_text())["sentences"]) if f.exists() else set()
    return _frequent


def clean_foa_field(text):
    freq = frequent_sentences()
    keep = [s for s in foa_sentences(text) if norm_sentence(s) not in freq]
    return squash("\n".join(keep))


def clean_title(title):
    return squash(PARSER_ARTIFACTS.sub(" ", title or ""))


def query_text(row, variant):
    """Variant A: title + purpose. Variant B: title + purpose + Section I (truncated later)."""
    parts = [clean_title(row["foa_title"]), clean_foa_field(row["foa_purpose"])]
    if variant == "B":
        parts.append(clean_foa_field(row["foa_description"]))
    elif variant != "A":
        raise ValueError(variant)
    return "\n".join(p for p in parts if p)


def doc_text(row):
    return squash(f"{row['project_title'] or ''}\n{clean_abstract(row['abstract'])}")


def truncate_tokens(texts, tokenizer, max_tokens):
    """Cut texts to max_tokens tokens (no special tokens), so BM25 and dense see the same text."""
    enc = tokenizer.encode_batch(list(texts), add_special_tokens=False)
    out = []
    for t, e in zip(texts, enc):
        if len(e.ids) <= max_tokens:
            out.append(t)
        else:
            out.append(t[: e.offsets[max_tokens - 1][1]])
    return out
