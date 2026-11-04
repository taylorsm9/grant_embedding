"""Mine one hard negative per training pair (Plan 1 step 6), from the train split only.

Usage: python -I mine_negatives.py --data DIR --variant A [--model PATH] [--tag NAME]

Corpus: train-split abstracts from non-held-out institutions (deduplicated by text).
For each train FOA, candidates are the union of BM25 and zero-shot dense ranks
RANK_LO..RANK_HI. Filters: same FOA family, any winning contact PI of that family,
identical abstract text to any of the family's winners. Each training pair (FOA, positive)
draws one surviving candidate at random (seeded), without reuse within an FOA where possible.
Writes <DIR>/hard_negatives_<variant>.parquet (application_id, foa_number, neg_application_id)
and prints filter counts.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval import BM25, Data, Dense, SEED  # noqa: E402

RANK_LO, RANK_HI = 20, 100


def top_ranks(S, lo, hi):
    order = np.argsort(-S, axis=1)[:, :hi]
    return order[:, lo - 1:hi]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--variant", default="A")
    ap.add_argument("--model", default="models/Qwen3-Embedding-0.6B")
    ap.add_argument("--tag", default="qwen3-0.6b-zs")
    a = ap.parse_args()
    data = Data(a.data)
    s = data.splits
    train_all = s[s.split == "train"].drop_duplicates("abstract_hash")   # same order as near_dups.py
    pairs = s[(s.split == "train") & s.in_train_cap]
    corpus = train_all[~train_all.heldout_inst].reset_index(drop=True)
    foas = np.array(sorted(pairs.foa_number.unique()))
    q_text = data.queries.loc[foas, f"query_{a.variant}"].tolist()
    d_text = data.docs.loc[corpus.application_id, "doc"].tolist()

    enc = Dense(a.model, tag=a.tag, cache=Path(a.data) / "cache", d_batch=32)
    # Reuse the near_dups.py embeddings of all train abstracts, then drop held-out rows.
    all_emb = enc.encode(data.docs.loc[train_all.application_id, "doc"].tolist(), False)
    d_emb = all_emb[~train_all.heldout_inst.values]
    q_emb = enc.encode(q_text, True)

    cand = {}
    for name, scorer in [("bm25", lambda qs: BM25().score([q_text[i] for i in qs], d_text)),
                         ("dense", lambda qs: q_emb[qs] @ d_emb.T)]:
        for i in range(0, len(foas), 512):
            qs = list(range(i, min(i + 512, len(foas))))
            top = top_ranks(scorer(qs), RANK_LO, RANK_HI)
            for k, q in enumerate(qs):
                cand.setdefault(q, set()).update(top[k].tolist())
        print(f"  ranked with {name}", flush=True)

    fam_of = dict(zip(s.foa_number, s.family_id))
    fam_pis = s[s.split == "train"].groupby("family_id").contact_pi_id.agg(set).to_dict()
    fam_hash = s[s.split == "train"].groupby("family_id").abstract_hash.agg(set).to_dict()
    c_fam = corpus.family_id.values
    c_pi = corpus.contact_pi_id.values
    c_hash = corpus.abstract_hash.values
    rng = np.random.default_rng(SEED)
    counts = {"candidates": 0, "same_family": 0, "same_pi": 0, "same_abstract": 0, "kept": 0}
    rows, short = [], 0
    for qi, foa in enumerate(foas):
        fam = fam_of[foa]
        c = np.array(sorted(cand[qi]))
        counts["candidates"] += len(c)
        m_fam = c_fam[c] == fam
        m_pi = ~m_fam & np.isin(c_pi[c], list(fam_pis[fam]))
        m_hash = ~m_fam & ~m_pi & np.isin(c_hash[c], list(fam_hash[fam]))
        counts["same_family"] += int(m_fam.sum())
        counts["same_pi"] += int(m_pi.sum())
        counts["same_abstract"] += int(m_hash.sum())
        keep = c[~(m_fam | m_pi | m_hash)]
        counts["kept"] += len(keep)
        pos = pairs[pairs.foa_number == foa].application_id.values
        if len(keep) == 0:
            short += len(pos)
            continue
        pick = rng.permutation(keep)
        if len(pick) < len(pos):
            pick = np.concatenate([pick, rng.choice(keep, len(pos) - len(pick))])
        for p, n in zip(pos, pick[:len(pos)]):
            rows.append((p, foa, corpus.application_id[n]))
    out = pd.DataFrame(rows, columns=["application_id", "foa_number", "neg_application_id"])
    out.to_parquet(Path(a.data) / f"hard_negatives_{a.variant}.parquet", index=False)
    counts.update({"pairs": len(pairs), "pairs_with_negative": len(out), "pairs_without_negative": short,
                   "rank_window": [RANK_LO, RANK_HI], "corpus_docs": len(corpus), "foas": len(foas)})
    (Path(a.data) / f"hard_negatives_{a.variant}.json").write_text(json.dumps(counts, indent=2))
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
