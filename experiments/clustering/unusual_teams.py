"""Unusual teams: new collaborations between topically distant PIs, and whether a call-fit model could
have suggested them. CPU only, cached embeddings.

Fixed design, decided before looking at any result:
- PIs and profiles as in collab_eval.py (pre-FY2022 history; profile = mean abstract embedding).
- Topic distance uses the OFF-THE-SHELF profiles. A pair is "distant" when neither PI is among the
  other's 1,000 nearest PIs, "close" when at least one is among the other's 100 nearest; the rest
  are "middle".
- New collaborations: PI pairs on a project first funded FY2022-24 who never shared a pre-FY2022
  project (both with history).
Part A, describe: distant vs close pairs by mechanism family, call type, same institution, team
  size, and share of large multi-component awards (U, P mechanisms).
Part B, the call as a bridge: for pairs whose joint project is under one of the 769 test calls,
  each PI's national rank for that call (max over their abstracts, as in the email backtest), by
  model. Reported: share of pairs with both PIs in the top K, K = 100, 500, 1000.
Part C, suggest teams: for each test call, candidates = topically distant pairs among the call's
  top K PIs (K = 100, 500) by each model's call fit. Reported: candidates per call, realized
  distant pairs under that call found among them (recall), and precision against the random rate
  for a distant pair.
Examples of suggested teams that did form go to examples/ (git-ignored: titles and PI IDs).

Usage (from the repo root):
  CUDA_VISIBLE_DEVICES= .venv-train/bin/python -I experiments/clustering/unusual_teams.py [--size 4b]
Writes experiments/clustering/results/{unusual_teams.json, unusual_teams_bridge.svg} (numbers only);
--size 4b uses the 4B models and writes unusual_teams_4b.json, unusual_teams_bridge_4b.svg.
"""
import argparse
import json
import subprocess
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "common"))
sys.path.insert(0, str(ROOT / "experiments" / "pi_backtest"))
sys.path.insert(0, str(Path(__file__).parent))
from eval import Data, Dense, type_group  # noqa: E402
from backtest import cached, history, pi_ids  # noqa: E402
from collab_eval import DATA, profiles  # noqa: E402

MODELS = {"off the shelf": "qwen3-0.6b-zs", "fine-tuned": "qwen3-0.6b-ft-B"}
SLICES = ("test", "test_reissue")
FAR, NEAR = 1000, 100
KS_B = (100, 500, 1000)
KS_C = (100, 500)
OUT = Path(__file__).parent / "results"
EXAMPLES = Path(__file__).parent / "examples"


def kth_cos(P, k, batch=2048):
    """Cosine of each PI's k-th nearest other PI."""
    out = np.empty(len(P), dtype=np.float32)
    for i in range(0, len(P), batch):
        S = P[i:i + batch] @ P.T
        S[np.arange(len(S)), np.arange(i, i + len(S))] = -np.inf
        out[i:i + batch] = -np.partition(-S, k - 1, axis=1)[:, k - 1]
    return out


def setup(data, proj):
    """Shared state: history, PI profiles, call fit, and new FY2022-24 collaboration pairs."""
    # ---- history, PI profiles and call-fit matrices for both models (shared row order)
    P, S_pi, E_m, Q_m, ref = {}, {}, {}, {}, None
    for name, tag in MODELS.items():
        enc = Dense("unused", tag=tag, cache=DATA / "cache")
        hist, E, hp = history(data, enc, proj)
        if ref is None:
            ref = (hist, hp)
        assert hist.application_id.equals(ref[0].application_id)
        pis, starts = np.unique(hp.pi.values, return_index=True)
        P[name] = profiles(E, hp, pis)
        calls, Q = [], []
        for sl in SLICES:
            foas, q_text, *_ = data.eval_a(sl, "B")
            calls += list(foas)
            Q.append(cached(enc, q_text, True))
        E_m[name], Q_m[name] = E, np.concatenate(Q)
        S_doc = Q_m[name] @ E.T
        S_pi[name] = np.maximum.reduceat(S_doc[:, hp.row.values], starts, axis=1)   # calls x PIs
        if name == "fine-tuned":
            S_doc_ft = S_doc
    hist, hp = ref
    N = len(pis)
    pos = {p: i for i, p in enumerate(pis)}
    call_pos = {f: i for i, f in enumerate(calls)}
    latest_org = (hp.assign(fy=hist.fy.values[hp.row.values]).sort_values(["pi", "fy"])
                  .groupby("pi").org_ipf_code.last()).loc[pis].values

    Pz = P["off the shelf"]
    far_thr, near_thr = kth_cos(Pz, FAR), kth_cos(Pz, NEAR)

    def distance(a, b):
        c = np.einsum("ij,ij->i", Pz[a], Pz[b])
        far = (c < far_thr[a]) & (c < far_thr[b])
        near = (c >= near_thr[a]) | (c >= near_thr[b])
        return np.where(far, "distant", np.where(near, "close", "middle"))

    # ---- new collaborations
    prior = set()
    for ids in proj.PI_IDS[proj.FY <= 2021]:
        prior.update(combinations(sorted(set(pi_ids(ids))), 2))
    new = proj[proj.FY.between(2022, 2024)]
    rows = []
    for r in new.itertuples():
        all_ids = set(pi_ids(r.PI_IDS))
        ids = sorted(p for p in all_ids if p in pos)
        for c in combinations(ids, 2):
            if c not in prior:
                rows.append((pos[c[0]], pos[c[1]], r.FOA_NUMBER, r.ACTIVITY, r.ADMINISTERING_IC, len(all_ids), r.Index))
    pairs = pd.DataFrame(rows, columns=["a", "b", "foa", "activity", "ic", "team_size", "app"])
    pairs = pairs.drop_duplicates(["a", "b"]).reset_index(drop=True)
    pairs["kind"] = distance(pairs.a.values, pairs.b.values)
    pairs["same_inst"] = latest_org[pairs.a] == latest_org[pairs.b]
    pairs["mech"] = pairs.activity.str[0]
    pairs["call_type"] = pairs.foa.fillna("").map(lambda f: type_group(f.split("-")[0]) if f else "none")
    return dict(hist=hist, hp=hp, pis=pis, starts=starts, N=N, pos=pos, calls=calls, call_pos=call_pos,
                P=P, S_pi=S_pi, S_doc_ft=S_doc_ft, E=E_m, Q=Q_m, distance=distance, pairs=pairs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", choices=["0.6b", "4b"], default="4b")
    a = ap.parse_args()
    sfx = "" if a.size == "0.6b" else f"_{a.size}"
    for name in MODELS:
        MODELS[name] = MODELS[name].replace("0.6b", a.size)
    data = Data(DATA)
    proj = pd.read_parquet(ROOT / "grants" / "derived" / "projects.parquet",
                           columns=["APPLICATION_ID", "PI_IDS", "FY", "FOA_NUMBER", "ACTIVITY",
                                    "ADMINISTERING_IC", "PROJECT_TITLE", "FULL_PROJECT_NUM"])
    proj = proj.set_index(proj.APPLICATION_ID.astype(str))

    st = setup(data, proj)
    hist, hp, pis, starts, N, pos, calls, call_pos, P, S_pi, S_doc_ft, distance, pairs = (
        st[k] for k in ("hist", "hp", "pis", "starts", "N", "pos", "calls", "call_pos", "P", "S_pi",
                        "S_doc_ft", "distance", "pairs"))
    Pz = P["off the shelf"]
    out = {"design": __doc__.split("Usage")[0].strip(), "pool_PIs": N, "new_pairs": len(pairs),
           "pairs_by_kind": pairs.kind.value_counts().to_dict()}

    # ---- Part A: describe
    A = {}
    for kind in ("distant", "middle", "close"):
        d = pairs[pairs.kind == kind]
        A[kind] = {"pairs": len(d),
                   "same_institution": round(float(d.same_inst.mean()), 3),
                   "median_team_size": float(d.team_size.median()),
                   "U_or_P_mechanism": round(float(d.mech.isin(["U", "P"]).mean()), 3),
                   "mechanism_family": d.mech.value_counts(normalize=True).round(3).head(6).to_dict(),
                   "top_activity_codes": d.activity.value_counts(normalize=True).round(3).head(8).to_dict(),
                   "call_type": d.call_type.value_counts(normalize=True).round(3).to_dict(),
                   "top_institutes": d.ic.value_counts(normalize=True).round(3).head(6).to_dict()}
    out["A_describe"] = A

    # ---- Part B: the call as a bridge
    tp = pairs[pairs.foa.isin(call_pos)].copy()
    tp["ci"] = tp.foa.map(call_pos)
    B = {"pairs_under_test_calls": len(tp), "by_kind": tp.kind.value_counts().to_dict()}
    for name in MODELS:
        S = S_pi[name]
        rank = lambda ci, j: (S[ci] > S[ci, j][:, None]).sum(1) + 1
        ra, rb = rank(tp.ci.values, tp.a.values), rank(tp.ci.values, tp.b.values)
        tp[f"ra_{name}"], tp[f"rb_{name}"] = ra, rb
        B[name] = {kind: {f"both_top{k}": round(float(((ra <= k) & (rb <= k))[m].mean()), 3) for k in KS_B}
                   | {"median_worse_rank": float(np.median(np.maximum(ra, rb)[m]))}
                   for kind in ("distant", "middle", "close") for m in [tp.kind.values == kind]}
    B["chance_both_top100"] = round((100 / N) ** 2, 6)
    out["B_bridge"] = B

    # ---- Part C: suggest distant teams per call
    realized = {}
    for r in tp[tp.kind == "distant"].itertuples():
        realized.setdefault(r.ci, set()).add((min(r.a, r.b), max(r.a, r.b)))
    n_real = sum(map(len, realized.values()))
    frac_distant = float(np.mean(distance(*np.random.default_rng(0).integers(0, N, (2, 200_000)))  == "distant"))
    C = {"realized_distant_pairs": n_real, "calls_with_one": len(realized),
         "share_of_all_pairs_distant": round(frac_distant, 3)}
    examples = []
    for name in MODELS:
        S = S_pi[name]
        for k in KS_C:
            n_cand = found = 0
            for ci in range(len(calls)):
                top = np.argpartition(-S[ci], k - 1)[:k]
                ia, ib = np.triu_indices(k, 1)
                a, b = top[ia], top[ib]
                d = distance(a, b) == "distant"
                n_cand += int(d.sum())
                if ci in realized:
                    lo, hi = np.minimum(a[d], b[d]), np.maximum(a[d], b[d])
                    hit = set(zip(lo.tolist(), hi.tolist())) & realized[ci]
                    found += len(hit)
                    if name == "fine-tuned" and k == 100:
                        examples += [(ci, x, y) for x, y in hit]
            base = n_real / (len(calls) * N * (N - 1) / 2 * frac_distant)
            C[f"{name} top{k}"] = {"candidates_per_call": round(n_cand / len(calls), 1),
                                   "found": found, "recall": round(found / n_real, 3),
                                   "precision": round(found / n_cand, 6),
                                   "lift_over_random_distant_pair": round(found / n_cand / base, 1)}
    out["C_suggest"] = C
    OUT.mkdir(exist_ok=True)
    out["models"] = dict(MODELS)
    (OUT / f"unusual_teams{sfx}.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps({k: v for k, v in out.items() if k != "design"}, indent=1, default=str))

    # ---- recall figure (Part B): both partners in the call's top K, by K
    svg_recall(tp, S_pi, OUT / f"unusual_teams_bridge{sfx}.svg", N)

    # ---- examples (git-ignored)
    if subprocess.run(["git", "-C", str(ROOT), "check-ignore", "-q", str(EXAMPLES / "x.txt")]).returncode:
        raise SystemExit(f"{EXAMPLES} must be git-ignored")
    EXAMPLES.mkdir(exist_ok=True)
    titles = pd.read_parquet(ROOT / "grants" / "derived" / "foa_text.parquet", columns=["foa_number", "title"]
                             ).drop_duplicates("foa_number").set_index("foa_number").title
    hp_rows = {j: hp.row.values[starts[j]:(starts[j + 1] if j + 1 < len(starts) else len(hp))] for j in range(N)}
    lines = []
    for ci, x, y in examples:
        f = calls[ci]
        lines += [f"{f}: {titles.get(f, '')}"]
        for j in (x, y):
            rr = hp_rows[j]
            d = rr[np.argmax(S_doc_ft[ci, rr])]
            pr = proj.loc[hist.application_id[d]]
            lines += [f"  PI {pis[j]} (rank {int((S_pi['fine-tuned'][ci] > S_pi['fine-tuned'][ci, j]).sum()) + 1}): "
                      f"{pr.FULL_PROJECT_NUM} {pr.PROJECT_TITLE}"]
        lines += [f"  topic cosine {float(Pz[x] @ Pz[y]):.3f}", ""]
    (EXAMPLES / f"suggested_teams_that_formed{sfx}.txt").write_text("\n".join(lines))
    print(f"{len(examples)} examples -> {EXAMPLES}")


def svg_recall(tp, S_pi, path, N):
    Ks = np.unique(np.round(np.logspace(1, np.log10(5000), 40)).astype(int))
    W, H, l, b, t = 760, 420, 70, 60, 70
    X = lambda k: l + (np.log10(k) - 1) / (np.log10(5000) - 1) * (W - l - 40)
    Y = lambda v: H - b - v * (H - b - t)
    col = {"fine-tuned": "#2f6db3", "off the shelf": "#9a9a9a"}
    dash = {"distant": "", "close": "6 4"}
    parts = [f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {W} {H}' font-family='Helvetica, Arial, sans-serif' font-size='13'>",
             "<rect width='100%' height='100%' fill='white'/>",
             "<text x='24' y='28' font-size='17' font-weight='600'>Distant teams rarely have both partners near the top for their call</text>",
             "<text x='24' y='50' fill='#555'>Share of new FY2022–24 pairs with both PIs in the call's top K (of 44,376 PIs)</text>"]
    for v in (0, .25, .5, .75, 1):
        parts.append(f"<line x1='{l}' x2='{W - 40}' y1='{Y(v):.0f}' y2='{Y(v):.0f}' stroke='#e5e5e5'/>"
                     f"<text x='{l - 8}' y='{Y(v) + 4:.0f}' text-anchor='end' fill='#555'>{int(v * 100)}%</text>")
    for k in (10, 100, 1000):
        parts.append(f"<text x='{X(k):.0f}' y='{H - b + 20}' text-anchor='middle' fill='#555'>{k:,}</text>")
    parts.append(f"<text x='{(l + W - 40) / 2}' y='{H - 14}' text-anchor='middle' fill='#555'>K (log scale)</text>")
    ends = []
    for name in ("off the shelf", "fine-tuned"):
        ra, rb = tp[f"ra_{name}"].values, tp[f"rb_{name}"].values
        worst = np.maximum(ra, rb)
        for kind in ("close", "distant"):
            w = worst[tp.kind.values == kind]
            ys = [(w <= k).mean() for k in Ks]
            pts = " ".join(f"{X(k):.1f},{Y(v):.1f}" for k, v in zip(Ks, ys))
            parts.append(f"<polyline points='{pts}' fill='none' stroke='{col[name]}' stroke-width='2.2' stroke-dasharray='{dash[kind]}'/>")
            ends.append((Y(ys[-1]), f"{name}, {kind} pairs (n={len(w)})", col[name], dash[kind]))
    for j, (_, label, c, d) in enumerate(ends):
        y = t + 14 + 20 * j
        parts.append(f"<line x1='{l + 14}' x2='{l + 44}' y1='{y - 4}' y2='{y - 4}' stroke='{c}' stroke-width='2.2' stroke-dasharray='{d}'/>"
                     f"<text x='{l + 52}' y='{y}'>{label}</text>")
    parts.append("</svg>")
    path.write_text("\n".join(parts))


if __name__ == "__main__":
    main()
