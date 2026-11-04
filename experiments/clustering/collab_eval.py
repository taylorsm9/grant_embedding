"""Do fine-tuned embeddings put future collaborators closer together than off-the-shelf ones? CPU only.

Collaborations: pairs of PIs listed together on a new NIH project first funded in FY2022-24 (any
mechanism), where both PIs have pre-FY2022 abstracts and had never shared a pre-FY2022 project.

Fixed design, decided before looking at any result:
- PI profile = normalised mean of the PI's pre-FY2022 abstract embeddings (the backtest's history
  corpus); the pool is every PI with history (44,376).
- For each pair, each direction counts once: the rank of the partner among all other PIs by cosine
  to the PI's profile. Reported: R@10/100/1000, median rank; chance R@k = k / (pool - 1).
- Split by institution: same vs different institution (each PI's latest pre-FY2022 institution).
- "Unnoticed": among directions the off-the-shelf model ranks beyond 1,000, the share the fine-tuned
  model ranks in its top 100, and the same the other way round.

Usage (from the repo root):
  CUDA_VISIBLE_DEVICES= .venv-train/bin/python -I experiments/clustering/collab_eval.py
Writes experiments/clustering/results/collab_eval.json (numbers only).
"""
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "common"))
sys.path.insert(0, str(ROOT / "experiments" / "pi_backtest"))
from eval import Data, Dense  # noqa: E402
from backtest import history, pi_ids  # noqa: E402

DATA = ROOT / "experiments" / "common" / "frozen"
MODELS = {"off the shelf": "qwen3-0.6b-zs", "fine-tuned": "qwen3-0.6b-ft-B"}
KS = (10, 100, 1000)
FAR, NEAR = 1000, 100
OUT = Path(__file__).parent / "results"


def profiles(E_hist, hp, pis):
    P = np.zeros((len(pis), E_hist.shape[1]), dtype=np.float32)
    idx = np.searchsorted(pis, hp.pi.values)
    np.add.at(P, idx, E_hist[hp.row.values])
    return P / np.linalg.norm(P, axis=1, keepdims=True)


def partner_ranks(P, a, b, batch=1024):
    """Rank (1-based) of b[i] among all PIs except a[i], by cosine to a[i]."""
    out = np.empty(len(a), dtype=np.int64)
    for i in range(0, len(a), batch):
        S = P[a[i:i + batch]] @ P.T
        S[np.arange(len(S)), a[i:i + batch]] = -np.inf
        target = S[np.arange(len(S)), b[i:i + batch]]
        out[i:i + batch] = (S > target[:, None]).sum(1) + 1
    return out


def summary(r, pool):
    return {**{f"R@{k}": round(float((r <= k).mean()), 3) for k in KS},
            "median_rank": float(np.median(r)), "n": int(len(r)),
            **{f"chance_R@{k}": round(k / (pool - 1), 5) for k in KS}}


def main():
    data = Data(DATA)
    proj = pd.read_parquet(ROOT / "grants" / "derived" / "projects.parquet",
                           columns=["APPLICATION_ID", "PI_IDS", "FY"])
    proj = proj.set_index(proj.APPLICATION_ID.astype(str))

    # history and profiles for both models (same rows: the cache keys share one text list)
    ref = None
    P = {}
    for name, tag in MODELS.items():
        hist, E_hist, hp = history(data, Dense("unused", tag=tag, cache=DATA / "cache"), proj)
        if ref is None:
            ref = (hist, hp)
        assert hist.application_id.equals(ref[0].application_id)
        pis = np.unique(hp.pi.values)
        P[name] = profiles(E_hist, hp, pis)
    hist, hp = ref
    pi_pos = {p: i for i, p in enumerate(pis)}
    latest_org = (hp.assign(fy=hist.fy.values[hp.row.values]).sort_values(["pi", "fy"])
                  .groupby("pi").org_ipf_code.last())

    # prior collaborations: any pre-FY2022 project listing both PIs
    prior = set()
    for ids in proj.PI_IDS[proj.FY <= 2021]:
        ids = sorted(set(pi_ids(ids)))
        if len(ids) > 1:
            prior.update(combinations(ids, 2))

    # new collaborations: new projects first funded FY2022-24 with 2+ PIs who both have history
    pairs = set()
    n_projects = 0
    for ids in proj.PI_IDS[proj.FY.between(2022, 2024)]:
        ids = sorted({p for p in pi_ids(ids) if p in pi_pos})
        if len(ids) > 1:
            n_projects += 1
            pairs.update(c for c in combinations(ids, 2) if c not in prior)
    pairs = np.array(sorted(pairs))
    a = np.array([pi_pos[p] for p in pairs[:, 0]])
    b = np.array([pi_pos[p] for p in pairs[:, 1]])
    src, dst = np.concatenate([a, b]), np.concatenate([b, a])        # both directions
    same = (latest_org.loc[pis[src]].values == latest_org.loc[pis[dst]].values)
    print(f"{n_projects} FY2022-24 multi-PI projects; {len(pairs)} new pairs; {len(src)} directions; "
          f"{same.mean():.1%} same institution", flush=True)

    ranks = {name: partner_ranks(P[name], src, dst) for name in MODELS}
    pool = len(pis)
    out = {"design": __doc__.split("Usage")[0].strip(), "pool_PIs": pool,
           "multi_PI_projects_with_2plus_history_PIs": n_projects, "new_pairs": len(pairs),
           "prior_pairs_excluded_from": len(prior)}
    for group, m in (("all", np.ones_like(same)), ("same_institution", same), ("different_institution", ~same)):
        out[group] = {name: summary(r[m], pool) for name, r in ranks.items()}
        zs, ft = ranks["off the shelf"][m], ranks["fine-tuned"][m]
        out[group]["fine-tuned ranks higher"] = round(float((ft < zs).mean()), 3)
        out[group]["unnoticed"] = {
            f"zs>{FAR} & ft<={NEAR}": round(float(((zs > FAR) & (ft <= NEAR)).sum() / max((zs > FAR).sum(), 1)), 3),
            f"ft>{FAR} & zs<={NEAR}": round(float(((ft > FAR) & (zs <= NEAR)).sum() / max((ft > FAR).sum(), 1)), 3),
            f"n zs>{FAR}": int((zs > FAR).sum()), f"n ft>{FAR}": int((ft > FAR).sum())}
        print(group, json.dumps(out[group]), flush=True)
    OUT.mkdir(exist_ok=True)
    (OUT / "collab_eval.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
