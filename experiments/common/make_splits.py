"""Build the train / val / test splits (Plan 1 step 3, master decision 2).

Usage: python -I make_splits.py <pairs.parquet> <foa_families.parquet> <out_dir>

One row per (FOA, project) pair in pairs.parquet, with:
  split          train | val | test | test_reissue | drop_fy2025 | drop_after_cutoff |
                 drop_dup_abstract | drop_heldout_shared (multi-site copy of a held-out abstract)
  heldout_inst   project is at a held-out institution (none since 2026-11-03; kept for Eval B later)
  in_train_cap   train row kept by the per-FOA cap (training uses split == train & in_train_cap)

Rules:
  - FOA period from foa_first_award_fy: <= 2021 train period, 2022-2024 test, 2025 dropped.
  - Families: a family whose earliest member is train-period is a train family and
    its test-period members form the test_reissue slice; otherwise it is a test family
    (the new-call slice).
  - Train/val rows awarded in FY >= AWARD_CUTOFF_FY are dropped: a model deployed at the
    start of the test period could not have seen them.
  - VAL_FRAC of train families (seeded) become validation.
  - Held-out institutions (ORG_IPF_CODE) are removed from train and val in every year.
  - Per-FOA cap on train rows, sampled with a fixed seed.
  - A train/val row from another institution whose abstract is identical to a held-out
    institution's (multi-site consortium awards) is dropped.
  - A val/test/test_reissue row whose abstract text or core project also occurs in train
    is dropped; likewise test/test_reissue against val, and test_reissue against test.
Writes <out_dir>/splits.parquet and <out_dir>/split_report.json, and prints the
assertion output.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20261102
VAL_FRAC = 0.07
CAP = 40
TEST_FY = (2022, 2024)
AWARD_CUTOFF_FY = TEST_FY[0]
# No institution is held out (2026-11-03: train on all institutions, lab-level Eval B
# deferred). The earlier choice was UCSF 577508, Johns Hopkins 4134401, U. Michigan 1506502.
HELDOUT = {}


def text_hash(s):
    return hashlib.sha1(" ".join(s.split()).lower().encode()).hexdigest()


def assign(pairs, fams):
    p = pairs.merge(fams, on="foa_number", how="left", validate="many_to_one")
    assert p.family_id.notna().all(), "FOA without a family"
    fy = p.foa_first_award_fy
    period = np.where(fy <= TEST_FY[0] - 1, "train", np.where(fy <= TEST_FY[1], "test", "fy2025"))
    p["period"] = period
    fam_first = p.groupby("family_id").foa_first_award_fy.min()
    p["family_period"] = np.where(p.family_id.map(fam_first) < TEST_FY[0], "train", "test")

    split = np.where(p.period == "fy2025", "drop_fy2025",
             np.where((p.family_period == "train") & (p.period == "test"), "test_reissue",
             p.family_period))
    p["split"] = split

    rng = np.random.default_rng(SEED)
    train_fams = np.array(sorted(p.loc[p.split == "train", "family_id"].unique()))
    n_val = int(round(VAL_FRAC * len(train_fams)))
    val_fams = set(rng.choice(train_fams, size=n_val, replace=False))
    p.loc[(p.split == "train") & p.family_id.isin(val_fams), "split"] = "val"
    p.loc[p.split.isin(["train", "val"]) & (p.fy >= AWARD_CUTOFF_FY), "split"] = "drop_after_cutoff"

    p["heldout_inst"] = p.org_ipf_code.isin(HELDOUT)
    p["abstract_hash"] = p.abstract.map(text_hash)
    ho = p[p.heldout_inst]
    shared = (p.split.isin(["train", "val"]) & ~p.heldout_inst &
              (p.abstract_hash.isin(set(ho.abstract_hash)) | p.core_project_num.isin(set(ho.core_project_num))))
    p.loc[shared, "split"] = "drop_heldout_shared"

    # Val/test rows whose abstract or core project is also a train row (any institution).
    train = p[p.split == "train"]
    seen_abs, seen_proj = set(train.abstract_hash), set(train.core_project_num)
    dup = (p.split.isin(["val", "test", "test_reissue"]) &
           (p.abstract_hash.isin(seen_abs) | p.core_project_num.isin(seen_proj)))
    p.loc[dup, "split"] = "drop_dup_abstract"
    # Same check for later splits against val, then the reissue slice against new-call test.
    for ref, later in [("val", ["test", "test_reissue"]), ("test", ["test_reissue"])]:
        r = p[p.split == ref]
        dup = p.split.isin(later) & (p.abstract_hash.isin(set(r.abstract_hash)) |
                                     p.core_project_num.isin(set(r.core_project_num)))
        p.loc[dup, "split"] = "drop_dup_abstract"

    # Per-FOA cap on the rows training actually uses (train split, not held-out).
    p["in_train_cap"] = False
    usable = p[(p.split == "train") & ~p.heldout_inst]
    keep = (usable.sample(frac=1.0, random_state=SEED)
            .groupby("foa_number").head(CAP).index)
    p.loc[keep, "in_train_cap"] = True
    return p


def check(p):
    """Leakage assertions; returns printable lines. Raises on failure."""
    out = []
    train = p[(p.split == "train") & ~p.heldout_inst]
    val = p[(p.split == "val") & ~p.heldout_inst]
    test = p[p.split == "test"]
    reissue = p[p.split == "test_reissue"]
    groups = {"train": train, "val": val, "test": test, "reissue": reissue}
    for key in ["application_id", "core_project_num", "abstract_hash", "foa_number", "family_id"]:
        for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
            n = len(set(groups[a][key]) & set(groups[b][key]))
            out.append(f"overlap {key:<17} {a:>5}/{b:<5}: {n}")
            assert n == 0, out[-1]
    # The reissue slice shares families with train/val by design, but never an award.
    for key in ["application_id", "core_project_num", "abstract_hash", "foa_number"]:
        for a in ["train", "val", "test"]:
            n = len(set(groups[a][key]) & set(reissue[key]))
            out.append(f"overlap {key:<17} {a:>5}/reissue: {n}")
            assert n == 0, out[-1]
    n = int(p[p.split.isin(["train", "val"])].fy.ge(AWARD_CUTOFF_FY).sum())
    out.append(f"train/val rows awarded >= FY{AWARD_CUTOFF_FY}: {n}")
    assert n == 0, out[-1]
    n = int((reissue.foa_first_award_fy < TEST_FY[0]).sum() + (reissue.foa_first_award_fy > TEST_FY[1]).sum())
    out.append(f"test_reissue rows outside FY{TEST_FY[0]}-{TEST_FY[1]}: {n}")
    assert n == 0
    for name in ["train", "val"]:
        n = int(groups[name].org_ipf_code.isin(HELDOUT).sum())
        out.append(f"held-out institution rows in {name}: {n}")
        assert n == 0, out[-1]
    # Held-out institutions' train-period abstracts must not leak into training via another org row.
    ho = set(p[p.heldout_inst].abstract_hash) | set(p[p.heldout_inst].core_project_num)
    n = int((train.abstract_hash.isin(ho) | train.core_project_num.isin(ho)).sum())
    out.append(f"train rows sharing abstract/project with held-out institutions: {n}")
    assert n == 0, out[-1]
    fam_splits = p[p.split.isin(["train", "val", "test"])].groupby("family_id").split.nunique()
    out.append(f"families in more than one split: {int((fam_splits > 1).sum())}")
    assert (fam_splits <= 1).all(), out[-1]
    n = int((test.foa_first_award_fy < TEST_FY[0]).sum() + (test.foa_first_award_fy > TEST_FY[1]).sum())
    out.append(f"test rows outside FY{TEST_FY[0]}-{TEST_FY[1]}: {n}")
    assert n == 0
    n = int(p[p.split.isin(["train", "val"])].foa_first_award_fy.ge(TEST_FY[0]).sum())
    out.append(f"train/val rows first awarded >= FY{TEST_FY[0]}: {n}")
    assert n == 0
    caps = p[p.in_train_cap].groupby("foa_number").size()
    out.append(f"max train pairs per FOA after cap: {int(caps.max())} (cap {CAP})")
    assert caps.max() <= CAP
    out.append("ALL SPLIT ASSERTIONS PASSED")
    return out


def report(p):
    def summ(d):
        return {"pairs": len(d), "foas": int(d.foa_number.nunique()), "families": int(d.family_id.nunique())}
    r = {"seed": SEED, "val_frac": VAL_FRAC, "cap": CAP, "test_fy": TEST_FY,
         "award_cutoff_fy": AWARD_CUTOFF_FY, "heldout": HELDOUT,
         "by_split": {s: summ(d) for s, d in p.groupby("split")},
         "train_used": summ(p[p.in_train_cap]),
         "train_uncapped_non_heldout": summ(p[(p.split == "train") & ~p.heldout_inst]),
         "val_non_heldout": summ(p[(p.split == "val") & ~p.heldout_inst]),
         "test_non_heldout": summ(p[(p.split == "test") & ~p.heldout_inst]),
         "heldout_rows_by_split": {f"{HELDOUT[k][:30]}": p[p.org_ipf_code == k].split.value_counts().to_dict()
                                   for k in HELDOUT},
         "heldout_org_names": p[p.heldout_inst].groupby("org_ipf_code").org_name.unique()
                               .map(list).to_dict()}
    fam = p[p.in_train_cap].groupby("family_id").size()
    r["train_families"] = int(fam.size)
    r["largest_train_family_pairs"] = {k: int(v) for k, v in fam.nlargest(5).items()}
    test = p[(p.split == "test") & ~p.heldout_inst]
    r["test_foas_by_type"] = test.drop_duplicates("foa_number").foa_type.value_counts().to_dict()
    for name in ["test", "test_reissue"]:
        d = p[p.split == name].drop_duplicates("foa_number")
        r[f"{name}_foas_by_first_award_fy"] = d.foa_first_award_fy.value_counts().sort_index().to_dict()
    r["test_queries_with_heldout_winners"] = int(p[(p.split == "test") & p.heldout_inst].foa_number.nunique())
    return r


def main(pairs_path, fam_path, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = assign(pd.read_parquet(pairs_path), pd.read_parquet(fam_path))
    lines = check(p)
    print("\n".join(lines))
    cols = ["foa_number", "family_id", "application_id", "core_project_num", "contact_pi_id",
            "org_ipf_code", "abstract_hash", "foa_type", "foa_first_award_fy", "fy",
            "split", "heldout_inst", "in_train_cap"]
    p[cols].to_parquet(out / "splits.parquet", index=False)
    r = report(p)
    r["assertions"] = lines
    (out / "split_report.json").write_text(json.dumps(r, indent=2, default=str))
    print(json.dumps({k: v for k, v in r.items() if k != "assertions"}, indent=2, default=str))


if __name__ == "__main__":
    main(*sys.argv[1:4])
