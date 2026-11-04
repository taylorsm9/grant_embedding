"""Check a rebuilt dataset against the released manifest, and optionally pin it to the manifest.

Usage: python -I release/verify.py <rebuilt_dir> <manifest_dir> [--pin <out_dir>]

<rebuilt_dir> is the output of experiments/common/build_all.sh (splits, queries, docs, ...).
Reports, per split, rows missing from or extra to the rebuild, and for shared rows any
difference in family, abstract hash, model-input document text or FOA query text.
Exit status 0 means an exact match.

Upstream sources can drift (NIH occasionally revises records or amends an FOA page).
--pin writes <out_dir> with the rebuilt queries/docs/meta, but with split, in_train_cap
and family_id taken from the manifest and the manifest's FOA links, hard negatives and
near-duplicate scores, so training and evaluation use the canonical sets. Rows the
manifest has but the rebuild lacks are listed in pin_missing.csv; rebuilt rows the
manifest lacks are marked split = drop_not_in_manifest.
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pandas as pd


def sha1(s):
    return hashlib.sha1(s.encode()).hexdigest()


def compare(rb, man):
    s = pd.read_parquet(rb / "splits.parquet")
    docs = pd.read_parquet(rb / "docs.parquet").set_index("application_id")
    q = pd.read_parquet(rb / "queries.parquet").set_index("foa_number")
    m = pd.read_csv(man / "splits.csv.gz", dtype={"application_id": str, "core_project_num": str})
    mf = pd.read_csv(man / "foas.csv.gz", dtype=str).set_index("foa_number")
    s["application_id"] = s.application_id.astype(str)
    docs.index = docs.index.astype(str)

    rep = {"splits": {}}
    for name in sorted(set(m.split) | set(s.split)):
        a = set(m.application_id[m.split == name])
        b = set(s.application_id[s.split == name])
        rep["splits"][name] = {"manifest": len(a), "rebuilt": len(b), "missing": len(a - b), "extra": len(b - a)}
    j = m.merge(s[["application_id", "family_id", "abstract_hash", "in_train_cap"]], on="application_id",
                how="inner", suffixes=("", "_rb"))
    j["doc_sha1_rb"] = j.application_id.map(docs.doc.map(sha1))
    rep["shared_rows"] = len(j)
    rep["family_mismatch"] = int((j.family_id != j.family_id_rb).sum())
    rep["abstract_hash_mismatch"] = int((j.abstract_sha1 != j.abstract_hash).sum())
    rep["doc_text_mismatch"] = int((j.doc_sha1 != j.doc_sha1_rb).sum())
    rep["train_cap_mismatch"] = int((j.in_train_cap.astype(bool) != j.in_train_cap_rb.astype(bool)).sum())
    for v in ("A", "B"):
        rb_h = q[f"query_{v}"].map(sha1)
        common = mf.index.intersection(rb_h.index)
        rep[f"query_{v}_mismatch"] = int((mf.loc[common, f"query_{v}_sha1"] != rb_h.loc[common]).sum())
    rep["foas_missing"] = int(len(mf.index.difference(q.index)))

    for p in sorted(man.glob("hard_negatives_*.csv.gz")):
        rb_p = rb / f"{p.name.replace('.csv.gz', '.parquet')}"
        if rb_p.exists():
            a = pd.read_csv(p, dtype=str).set_index("application_id").neg_application_id
            b = pd.read_parquet(rb_p).astype(str).set_index("application_id").neg_application_id
            c = a.index.intersection(b.index)
            rep[f"{p.name.split('.')[0]}_agreement"] = round(float((a.loc[c] == b.loc[c]).mean()), 4) if len(c) else None

    exact = (all(v["missing"] == 0 and v["extra"] == 0 for v in rep["splits"].values()) and
             all(rep[k] == 0 for k in ["family_mismatch", "abstract_hash_mismatch", "doc_text_mismatch",
                                        "train_cap_mismatch", "query_A_mismatch", "query_B_mismatch",
                                        "foas_missing"]))
    rep["exact_match"] = exact
    return rep, s, m


def pin(rb, man, out, s, m):
    out.mkdir(parents=True, exist_ok=True)
    for f in ["queries.parquet", "docs.parquet", "foa_meta.parquet"]:
        shutil.copy(rb / f, out / f)
    keep = m.set_index("application_id")[["split", "in_train_cap", "family_id"]]
    p = s.drop(columns=["split", "in_train_cap", "family_id"]).join(keep, on="application_id")
    p["in_train_cap"] = p.in_train_cap.fillna(False).astype(bool)
    p["split"] = p.split.fillna("drop_not_in_manifest")
    p["family_id"] = p.family_id.fillna("")
    p.to_parquet(out / "splits.parquet", index=False)
    m[~m.application_id.isin(set(s.application_id))].to_csv(out / "pin_missing.csv", index=False)
    pd.read_csv(man / "foa_links.csv.gz", dtype=str).to_parquet(out / "foa_links.parquet", index=False)
    for pth in man.glob("hard_negatives_*.csv.gz"):
        h = pd.read_csv(pth, dtype=str)
        h = h[h.application_id.isin(set(p.application_id)) & h.neg_application_id.isin(set(p.application_id))]
        h.to_parquet(out / pth.name.replace(".csv.gz", ".parquet"), index=False)
    for pth in man.glob("near_dups_*.csv.gz"):
        pd.read_csv(pth, dtype={"application_id": str}).to_parquet(
            out / pth.name.replace(".csv.gz", ".parquet"), index=False)
    print(f"pinned dataset written to {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rebuilt")
    ap.add_argument("manifest")
    ap.add_argument("--pin", default=None)
    a = ap.parse_args()
    rb, man = Path(a.rebuilt), Path(a.manifest)
    rep, s, m = compare(rb, man)
    print(json.dumps(rep, indent=2))
    (rb / "verify_report.json").write_text(json.dumps(rep, indent=2))
    if a.pin:
        pin(rb, man, Path(a.pin), s, m)
    sys.exit(0 if rep["exact_match"] else 1)


if __name__ == "__main__":
    main()
