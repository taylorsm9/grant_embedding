"""Build cleaned query and document texts (Plan 1 step 1).

Usage: python -I build_text.py <pairs.parquet> <tokenizer.json> <out_dir>

1. Counts normalised sentences over the purpose + Section I of every FOA in pairs
   (one count per FOA). Sentences in >= 2% of FOAs are boilerplate; the list is
   written to experiments/common/foa_boilerplate_sentences.json. This uses only
   unlabeled FOA text, no award labels.
2. Writes <out_dir>/queries.parquet (foa_number, query_A, query_B, token counts)
   and <out_dir>/docs.parquet (application_id, doc, token counts), truncated to the
   caps in text.py so every model and BM25 see the same text, plus
   <out_dir>/foa_meta.parquet (type, size, first-award FY, release date).
"""
import collections
import json
import sys
from pathlib import Path

import pandas as pd
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import text  # noqa: E402

MIN_FRAC = 0.02
MIN_CHARS = 20


def find_boilerplate(foas):
    counts = collections.Counter()
    for _, r in foas.iterrows():
        sents = text.foa_sentences(r.foa_purpose) + text.foa_sentences(r.foa_description)
        counts.update({text.norm_sentence(s) for s in sents if len(s) >= MIN_CHARS})
    n = len(foas)
    freq = sorted((s for s, k in counts.items() if k >= MIN_FRAC * n), key=lambda s: -counts[s])
    return {"n_foas": n, "min_frac": MIN_FRAC, "min_chars": MIN_CHARS,
            "sentences": freq, "foa_counts": [counts[s] for s in freq]}


def main(pairs_path, tok_path, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pairs = pd.read_parquet(pairs_path)
    tok = Tokenizer.from_file(tok_path)

    foas = pairs.drop_duplicates("foa_number")
    bp = find_boilerplate(foas)
    (text.HERE / "foa_boilerplate_sentences.json").write_text(json.dumps(bp, indent=1))
    text._frequent = None
    print(f"boilerplate sentences: {len(bp['sentences'])} (>= {MIN_FRAC:.0%} of {bp['n_foas']} FOAs)")

    prompt_len = len(tok.encode(text.QUERY_PROMPT, add_special_tokens=False).ids)
    q_max = text.QUERY_CAP - prompt_len - 1
    d_max = text.DOC_CAP - 1
    q = pd.DataFrame({"foa_number": foas.foa_number.values})
    for v in "AB":
        full = [text.query_text(r, v) for _, r in foas.iterrows()]
        q[f"query_{v}_full_tokens"] = [len(e.ids) for e in tok.encode_batch(full, add_special_tokens=False)]
        q[f"query_{v}"] = text.truncate_tokens(full, tok, q_max)
    docs = pairs[["application_id", "project_title", "abstract"]].copy()
    full = [text.doc_text(r) for _, r in docs.iterrows()]
    docs["doc_full_tokens"] = [len(e.ids) for e in tok.encode_batch(full, add_special_tokens=False)]
    docs["doc"] = text.truncate_tokens(full, tok, d_max)
    docs["raw_tokens"] = [len(e.ids) for e in tok.encode_batch(
        [f"{a}\n{b}" for a, b in zip(docs.project_title, docs.abstract)], add_special_tokens=False)]
    docs = docs.drop(columns=["project_title", "abstract"])

    q.to_parquet(out / "queries.parquet", index=False)
    foas[["foa_number", "foa_type", "foa_n_awards", "foa_first_award_fy", "foa_release_date"]].to_parquet(
        out / "foa_meta.parquet", index=False)
    docs.to_parquet(out / "docs.parquet", index=False)
    pct = lambda s: s.quantile([.5, .9, .95, .99]).round().astype(int).tolist()
    stats = {
        "prompt_tokens": prompt_len, "query_text_max": q_max, "doc_text_max": d_max,
        "query_A_p50_90_95_99": pct(q.query_A_full_tokens),
        "query_A_frac_truncated": round(float((q.query_A_full_tokens > q_max).mean()), 4),
        "query_B_p50_90_95_99": pct(q.query_B_full_tokens),
        "query_B_frac_truncated": round(float((q.query_B_full_tokens > q_max).mean()), 4),
        "doc_raw_p50_90_95_99": pct(docs.raw_tokens),
        "doc_clean_p50_90_95_99": pct(docs.doc_full_tokens),
        "doc_frac_truncated": round(float((docs.doc_full_tokens > d_max).mean()), 4),
        "empty_query_A": int((q.query_A.str.len() < 20).sum()),
        "empty_doc": int((docs.doc.str.len() < 50).sum()),
    }
    (out / "text_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:4])
