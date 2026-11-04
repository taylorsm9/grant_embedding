"""Export cached eval embeddings with their IDs, for CPU work without a GPU (CPU only, seconds).

eval.py's Dense caches embeddings as <data>/cache/<tag>_{q,d}_<hash of texts>.npy, with no IDs.
This rebuilds each split's Eval A query/corpus lists exactly as eval.py does, finds the
matching cache files and writes one self-describing file per (tag, split):

  <data>/cache/export/<tag>_<split>_<variant>.npz
    foa_number  (n_q,)        query ids, row order of Q and R
    q           (n_q, dim)    float32, L2-normalised query embeddings (instruction prompt applied)
    doc_id      (n_d,)        application_id, row order of D and columns of R
    d           (n_d, dim)    float32, L2-normalised document embeddings
    relevant    (n_q, n_d)    bool, Eval A relevance (awards under the FOA)
  Cosine scores are q @ d.T. The near-duplicate runs use subsets of the same corpus; filter
  with near_dups_<split>.parquet (max_cos >= 0.95) if needed.

The export holds vectors and ids only, no text. It lives under cache/ (git-ignored); like
the cache, do not copy it off this machine as a results artefact.

Usage: python -I experiments/retriever_4b/export_embeddings.py --data experiments/common/frozen \
           [--tags qwen3-4b-zs,qwen3-4b-ft-B] [--splits val,test,test_reissue]
"""
import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/common"))
from eval import Data  # noqa: E402


def cache_file(cache, tag, texts, is_query):
    # Must match eval.Dense.encode exactly.
    key = hashlib.sha1(("\x00".join([tag, str(is_query)] + list(texts))).encode()).hexdigest()[:20]
    return cache / f"{tag}_{'q' if is_query else 'd'}_{key}.npy"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--tags", default="qwen3-4b-zs,qwen3-4b-ft-B")
    ap.add_argument("--splits", default="val,test,test_reissue")
    ap.add_argument("--variant", default="B")
    a = ap.parse_args()
    data = Data(a.data)
    cache = Path(a.data) / "cache"
    out = cache / "export"
    out.mkdir(parents=True, exist_ok=True)
    for split in a.splits.split(","):
        foas, q_text, doc_ids, d_text, R, _ = data.eval_a(split, a.variant)
        for tag in a.tags.split(","):
            fq, fd = cache_file(cache, tag, q_text, True), cache_file(cache, tag, d_text, False)
            if not (fq.exists() and fd.exists()):
                print(f"skip {tag} {split}: not cached (queries {fq.exists()}, docs {fd.exists()})")
                continue
            Q, D = np.load(fq), np.load(fd)
            assert Q.shape[0] == len(foas) and D.shape[0] == len(doc_ids), (Q.shape, D.shape)
            dst = out / f"{tag}_{split}_{a.variant}.npz"
            np.savez(dst, foa_number=np.asarray(foas), q=Q, doc_id=np.asarray(doc_ids), d=D, relevant=R)
            print(f"wrote {dst}  q {Q.shape}  d {D.shape}")


if __name__ == "__main__":
    main()
