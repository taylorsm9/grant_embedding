"""Embed every query and document used in training or evaluation, for analysis and figures
(1 GPU, ~20 min for the 4B).

Rows: everything in the train, val, test and test_reissue splits (positives, hard negatives,
distractor pools). Rows dropped from the design (drop_after_cutoff, drop_dup_abstract,
drop_fy2025) are excluded. Texts are exactly the model inputs: query variant B (title +
purpose + start of Section I) with the instruction prompt, truncated at 512 tokens; documents
are title + abstract truncated at 768 tokens.
Same prompt, caps, pooling and normalisation as eval.py's Dense, so these vectors match the
evaluation ones. Join with splits.parquet (application_id, foa_number) for split, FY, type.

Output: <data>/cache/export/<tag>_used_<variant>.npz
  foa_number (n_q,), q_split (n_q,), q (n_q, dim) float32;
  doc_id (n_d,) application_id, d_split (n_d,), d (n_d, dim) float32.
Vectors and ids only, no text; lives under cache/ (git-ignored). Skips if the output exists.
Vectors that eval.py already cached for this tag (val/test/test_reissue pools) are reused, not re-encoded.

Usage: python -I experiments/retriever_4b/encode_all.py --data experiments/common/frozen \
           --model models/Qwen3-Embedding-4B --tag qwen3-4b-zs
"""
import argparse
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/common"))
import text  # noqa: E402
from eval import Data  # noqa: E402

USED = ["train", "val", "test", "test_reissue"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--variant", default="B")
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()
    d = Path(a.data)
    dst = d / "cache" / "export" / f"{a.tag}_used_{a.variant}.npz"
    if dst.exists():
        print(f"{dst} exists; skipping")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(a.model, device="cuda", model_kwargs={"attn_implementation": "sdpa", "dtype": torch.bfloat16})
    s = pd.read_parquet(d / "splits.parquet", columns=["application_id", "foa_number", "split"])
    s = s[s.split.isin(USED)]
    assert s.application_id.is_unique
    hn = pd.read_parquet(d / "hard_negatives_B.parquet")
    assert hn.neg_application_id.isin(s.application_id).all(), "a hard negative is outside the used rows"
    q_split = s.drop_duplicates("foa_number").set_index("foa_number").split
    q = pd.read_parquet(d / "queries.parquet", columns=["foa_number", f"query_{a.variant}"])
    q = q[q.foa_number.isin(q_split.index)].reset_index(drop=True)
    docs = pd.read_parquet(d / "docs.parquet", columns=["application_id", "doc"])
    docs = docs[docs.application_id.isin(s.application_id)].reset_index(drop=True)
    d_split = docs.application_id.map(s.set_index("application_id").split)
    print(f"{len(q)} queries, {len(docs)} docs ({d_split.value_counts().to_dict()})", flush=True)

    def enc(texts, cap, prompt):
        m.max_seq_length = cap
        t0 = time.time()
        e = m.encode(list(texts), prompt=prompt, batch_size=a.batch, normalize_embeddings=True,
                     convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
        print(f"  [{a.tag}] encoded {len(texts)} in {time.time() - t0:.0f}s", flush=True)
        return e

    # vectors eval.py cached for this tag, keyed the way Dense.encode keys them
    have = {True: {}, False: {}}
    data = Data(d)
    for split in ("val", "test", "test_reissue"):
        foas, q_text, doc_ids, d_text, *_ = data.eval_a(split, a.variant)
        for ids, texts, isq in ((foas, q_text, True), (doc_ids, d_text, False)):
            key = hashlib.sha1(("\x00".join([a.tag, str(isq)] + list(texts))).encode()).hexdigest()[:20]
            f = d / "cache" / f"{a.tag}_{'q' if isq else 'd'}_{key}.npy"
            if f.exists():
                have[isq].update(zip(ids, np.load(f)))
    del data

    def fill(ids, texts, isq):
        miss = [i for i, x in enumerate(ids) if x not in have[isq]]
        print(f"  [{a.tag}] {'queries' if isq else 'docs'}: {len(ids) - len(miss)} reused from eval cache, "
              f"{len(miss)} to encode", flush=True)
        new = enc([texts[i] for i in miss], text.QUERY_CAP if isq else text.DOC_CAP,
                  text.QUERY_PROMPT if isq else None) if miss else None
        E = np.stack([have[isq][x] for x in ids]) if not miss else np.empty((len(ids), new.shape[1]), np.float32)
        if miss:
            hit = [i for i, x in enumerate(ids) if x in have[isq]]
            E[hit] = np.stack([have[isq][ids[i]] for i in hit]) if hit else E[hit]
            E[miss] = new
        return E.astype(np.float32)

    Q = fill(list(q.foa_number), list(q[f"query_{a.variant}"]), True)
    D = fill(list(docs.application_id), list(docs["doc"]), False)
    tmp = dst.with_name(dst.stem + ".tmp.npz")
    np.savez(tmp, foa_number=q.foa_number.to_numpy(), q_split=q.foa_number.map(q_split).to_numpy().astype(str), q=Q,
             doc_id=docs.application_id.to_numpy(), d_split=d_split.to_numpy().astype(str), d=D)
    tmp.rename(dst)
    print(f"wrote {dst}  q {Q.shape}  d {D.shape}")


if __name__ == "__main__":
    main()
