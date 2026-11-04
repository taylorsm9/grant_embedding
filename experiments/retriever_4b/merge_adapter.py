"""Merge a trained LoRA adapter into the 4B base and save a plain sentence-transformers model.

Usage: python -I merge_adapter.py --model models/Qwen3-Embedding-4B --adapter RUN/best_adapter --out RUN/merged
       [--data experiments/common/frozen]

Plan 2 step 4.1: with --data, also checks that adapter-attached and merged models give the
same embeddings on 32 val queries and 32 val documents (max abs difference printed; bf16,
so expect about 1e-2 or less on normalised vectors, and identical rankings).
The merged directory is what eval.py takes as --model.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/common"))
import text  # noqa: E402


def load(path, fp32=False):
    from sentence_transformers import SentenceTransformer
    cuda = torch.cuda.is_available()
    return SentenceTransformer(path, device="cuda" if cuda else "cpu",
                               model_kwargs={"attn_implementation": "sdpa",
                                             "dtype": torch.bfloat16 if cuda and not fp32 else torch.float32})


def enc(m, texts, query):
    m.max_seq_length = text.QUERY_CAP if query else text.DOC_CAP
    return m.encode(texts, prompt=text.QUERY_PROMPT if query else None, batch_size=8,
                    normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)


def main():
    from peft import PeftModel
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=None)
    ap.add_argument("--fp32", action="store_true",
                    help="merge and check in float32 on GPU too (removes bf16 near-tie noise from the check; "
                         "the merged model is saved in float32, and eval.py still loads it in bf16)")
    a = ap.parse_args()
    m = load(a.model, a.fp32)
    m[0].model = PeftModel.from_pretrained(m[0].model, a.adapter)   # .auto_model is a read-only alias
    sample = None
    if a.data:
        from eval import Data
        d = Data(a.data)
        foas, q_text, doc_ids, d_text, R, info = d.eval_a("val", "B")
        sample = (q_text[:32], d_text[:32])
        before = [enc(m, sample[0], True), enc(m, sample[1], False)]
    m[0].model = m[0].model.merge_and_unload()
    m.save(a.out)
    print(f"merged model saved to {a.out}")
    if sample:
        del m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()   # two 4B models in fp32 do not fit in 24 GB together
        merged = load(a.out, a.fp32)
        after = [enc(merged, sample[0], True), enc(merged, sample[1], False)]
        diff = max(float(np.abs(x - y).max()) for x, y in zip(before, after))
        rows_same = (np.argsort(-(before[0] @ before[1].T), 1) == np.argsort(-(after[0] @ after[1].T), 1))[:, :5].all(1)
        same_rank = bool(rows_same.all())
        print(f"adapter vs merged: max |diff| {diff:.4g}; top-5 rankings identical on sample: {same_rank} "
              f"({int((~rows_same).sum())}/{len(rows_same)} query rows differ; dtype {'fp32' if a.fp32 or not torch.cuda.is_available() else 'bf16'})")


if __name__ == "__main__":
    main()
