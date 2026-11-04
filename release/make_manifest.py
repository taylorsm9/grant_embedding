"""Write the shareable manifest for the frozen dataset: identifiers, labels and hashes only.

Usage: python -I release/make_manifest.py <frozen_dir> <grants_dir> <manifest_dir>

No abstract text, project titles or PI names are written; abstracts belong to their
authors, so the dataset is shared as IDs + hashes + code (REPRODUCE.md). FOA text
is federal work, but it is also rebuilt from source so that one script covers everything.

Writes into <manifest_dir>:
  splits.csv.gz          application_id, foa_number, family_id, core_project_num, fy, split,
                         in_train_cap, abstract_sha1 (normalised raw abstract), doc_sha1 (cleaned,
                         truncated document text the models see)
  foas.csv.gz            foa_number, foa_text_source, source_file_sha256, query_A_sha1, query_B_sha1
  foa_links.csv.gz       direct FOA links used by the training mask
  hard_negatives_<v>.csv.gz, near_dups_<split>.csv.gz   (zero-shot model outputs, canonical copies)
  exporter_files.csv     the ExPORTER zips used (FY2008-2025) with size and SHA-256
  foa_fetch_list.txt, grantsgov_list.txt   the FOA numbers fetched
  build_info.json        git commit, constants, fetch dates, counts
"""
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/common"))
import make_splits  # noqa: E402
import text  # noqa: E402
import foa_families  # noqa: E402


def sha1(s):
    return hashlib.sha1(s.encode()).hexdigest()


def file_sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest() if Path(p).exists() else ""


def main(frozen, grants, out):
    frozen, grants, out = Path(frozen), Path(grants), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    s = pd.read_parquet(frozen / "splits.parquet")
    docs = pd.read_parquet(frozen / "docs.parquet").set_index("application_id")
    q = pd.read_parquet(frozen / "queries.parquet").set_index("foa_number")

    m = s[["application_id", "foa_number", "family_id", "core_project_num", "fy", "split", "in_train_cap",
           "abstract_hash"]].rename(columns={"abstract_hash": "abstract_sha1"})
    m["doc_sha1"] = m.application_id.map(docs.doc.map(sha1))
    m.sort_values("application_id").to_csv(out / "splits.csv.gz", index=False)

    src = (pd.read_parquet(grants / "derived/pairs.parquet", columns=["foa_number", "foa_text_source"])
           .drop_duplicates("foa_number").set_index("foa_number").foa_text_source)
    f = pd.DataFrame({"foa_number": q.index})
    f["foa_text_source"] = f.foa_number.map(src)
    f["source_file_sha256"] = [
        file_sha256(grants / ("raw/nih_guide" if t == "nih_guide" else "raw/grantsgov") /
                    (f"{n}.html" if t == "nih_guide" else f"{n}.json"))
        for n, t in zip(f.foa_number, f.foa_text_source)]
    for v in ("A", "B"):
        f[f"query_{v}_sha1"] = f.foa_number.map(q[f"query_{v}"].map(sha1))
    f.to_csv(out / "foas.csv.gz", index=False)

    pd.read_parquet(frozen / "foa_links.parquet").to_csv(out / "foa_links.csv.gz", index=False)
    for p in sorted(frozen.glob("hard_negatives_*.parquet")):
        pd.read_parquet(p).to_csv(out / f"{p.stem}.csv.gz", index=False)
    for p in sorted(frozen.glob("near_dups_*.parquet")):
        d = pd.read_parquet(p)
        d.round({"max_cos": 4}).to_csv(out / f"{p.stem}.csv.gz", index=False)

    ex = pd.read_csv(grants / "raw/nih_exporter/manifest.csv", dtype=str)
    ex = ex[ex.fy.astype(int).between(2008, 2025)]
    ex[["group", "fy", "file_name", "size", "sha256", "source_url", "downloaded_at"]].to_csv(
        out / "exporter_files.csv", index=False)
    shutil.copy(grants / "logs/foa_fetch_all.txt", out / "foa_fetch_list.txt")
    shutil.copy(grants / "logs/grantsgov_list.txt", out / "grantsgov_list.txt")

    fetch = pd.read_csv(grants / "logs/foa_fetch.csv", dtype=str)
    try:
        commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = None
    info = {
        "git_commit": commit,
        "exporter_downloaded": [ex.downloaded_at.min(), ex.downloaded_at.max()],
        "nih_guide_fetched": [fetch.fetched_at.min(), fetch.fetched_at.max()] if "fetched_at" in fetch else None,
        "fy_range": [2008, 2025],
        "split_constants": {"seed": make_splits.SEED, "val_frac": make_splits.VAL_FRAC, "cap": make_splits.CAP,
                            "test_fy": list(make_splits.TEST_FY), "award_cutoff_fy": make_splits.AWARD_CUTOFF_FY,
                            "heldout_institutions": make_splits.HELDOUT},
        "family_threshold": json.loads((frozen / "freeze.json").read_text())["family_threshold"]
        if (frozen / "freeze.json").exists() else None,
        "family_title_threshold": foa_families.TITLE_T,
        "text": {"query_cap": text.QUERY_CAP, "doc_cap": text.DOC_CAP, "query_prompt": text.QUERY_PROMPT},
        "counts": {"rows": len(m), "by_split": m.split.value_counts().to_dict(),
                   "train_used": int(m.in_train_cap.sum()), "foas": len(f)},
        "hashes": "abstract_sha1 = sha1 of the raw abstract, whitespace-collapsed and lower-cased "
                  "(make_splits.text_hash); doc_sha1 / query_*_sha1 = sha1 of the exact model input text",
    }
    (out / "build_info.json").write_text(json.dumps(info, indent=2, default=str))
    print(json.dumps(info["counts"], indent=2))
    print(f"manifest written to {out}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
