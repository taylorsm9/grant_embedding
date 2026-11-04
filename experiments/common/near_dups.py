"""Flag val/test abstracts that are near-duplicates of a training abstract (master "additional checks").

Usage: python -I near_dups.py --data DIR --split test [--model PATH] [--tag NAME]

Encodes every train-split abstract (all institutions) and the split's abstracts with
the zero-shot model, and records each split abstract's highest cosine to any train
abstract. Writes <DIR>/near_dups_<split>.parquet (application_id, max_cos,
nearest_train_id) and prints the distribution plus the closest pairs to eyeball,
so the threshold can be chosen and recorded. eval.py --near-dup-max T then reports
metrics without split abstracts whose max_cos >= T.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval import Data, Dense  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--model", default="models/Qwen3-Embedding-0.6B")
    ap.add_argument("--tag", default="qwen3-0.6b-zs")
    a = ap.parse_args()
    data = Data(a.data)
    s = data.splits
    train_ids = s[s.split == "train"].drop_duplicates("abstract_hash").application_id.values
    split_ids = s[s.split == a.split].application_id.unique()
    enc = Dense(a.model, tag=a.tag, cache=Path(a.data) / "cache", d_batch=32)
    tr = enc.encode(data.docs.loc[train_ids, "doc"].tolist(), False)
    te = enc.encode(data.docs.loc[split_ids, "doc"].tolist(), False)
    best, arg = np.empty(len(te), np.float32), np.empty(len(te), np.int64)
    for i in range(0, len(te), 2048):
        sim = te[i:i + 2048] @ tr.T
        arg[i:i + 2048] = sim.argmax(1)
        best[i:i + 2048] = sim.max(1)
    out = pd.DataFrame({"application_id": split_ids, "max_cos": best, "nearest_train_id": train_ids[arg]})
    out.to_parquet(Path(a.data) / f"near_dups_{a.split}.parquet", index=False)
    print(f"{a.split}: {len(out)} abstracts vs {len(train_ids)} train abstracts")
    print("max_cos percentiles 50/90/99/99.9:", np.percentile(best, [50, 90, 99, 99.9]).round(4))
    for t in (0.90, 0.92, 0.94, 0.95, 0.96, 0.98):
        print(f"  >= {t}: {int((best >= t).sum())} ({(best >= t).mean():.2%})")
    meta = pd.read_parquet(Path(a.data) / "docs.parquet").set_index("application_id").doc
    for lo, hi in [(0.97, 1.01), (0.94, 0.97), (0.90, 0.94)]:
        band = out[(out.max_cos >= lo) & (out.max_cos < hi)]
        print(f"\n=== {lo}-{hi}: {len(band)}; 3 examples ===")
        for r in band.sample(min(3, len(band)), random_state=0).itertuples():
            print(f"{r.max_cos:.3f} {r.application_id} vs train {r.nearest_train_id}")
            print("   S:", meta[r.application_id][:220].replace("\n", " "))
            print("   T:", meta[r.nearest_train_id][:220].replace("\n", " "))


if __name__ == "__main__":
    main()
