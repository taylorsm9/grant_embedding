"""PI-level backtest of a fine-tuned retriever (0.6B by default, --size 4b), run on cached embeddings (CPU only).

Question: if the tool had been run at the start of FY2022 over every PI's earlier (pre-FY2022)
abstracts, would the FY2022-24 calls those PIs went on to win have been recommended to them?

Fixed design, decided before looking at any result (test data, run once):
- History corpus: train-split and val-split abstracts (all awarded before FY2022), one row per
  distinct abstract, linked to every PI listed on the project (PI_IDS, not only the contact PI).
- Calls: all test-period calls in both slices (test = new calls, test_reissue = reissues).
- PI score for a call = max cosine over the PI's history abstracts; the arg-max abstract is the
  "matched prior grant" shown in the email.
- Winners: every PI listed on a test-slice award. Winners with no history abstract cannot be
  scored; their share is reported as coverage.
Three views:
  national     call -> rank of each winning PI among all PIs with history
  institution  call -> rank of each winning PI among PIs with history at the winner's
               institution (the research-office view); movers without history there are counted
  email        PI -> rank of the won call among all test-period calls (what the email would list)

Usage (from the repo root):
  CUDA_VISIBLE_DEVICES= .venv-train/bin/python -I experiments/pi_backtest/backtest.py [--size 4b]
Writes experiments/pi_backtest/results/backtest.json (backtest_4b.json for --size 4b; numbers only).
Emails contain project titles and PI IDs, so they go to --emails (default
experiments/pi_backtest/emails/, plus a 4b/ subdirectory for --size 4b), which git must ignore.
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
COMMON = ROOT / "experiments" / "common"
sys.path.insert(0, str(COMMON))
from eval import Data, Dense, type_group  # noqa: E402

DATA = COMMON / "frozen"
TAGS = {"0.6b": "qwen3-0.6b-ft-B", "4b": "qwen3-4b-ft-B"}
SLICES = ("test", "test_reissue")
KS_PI = (10, 50, 100, 500)
KS_INST = (1, 3, 5, 10)
KS_EMAIL = (1, 3, 10)


def cached(enc, texts, is_query, ids=None):
    """Cached embeddings only: a cache miss means the inputs changed, so fail instead of encoding.
    With ids (foa_number for queries, application_id for docs), a miss falls back to the ID-keyed
    export of retriever_4b/encode_all.py (<cache>/export/<tag>_used_B.npz), which encodes the same
    texts with the same prompt and caps."""
    import hashlib
    key = hashlib.sha1(("\x00".join([enc.name, str(is_query)] + list(texts))).encode()).hexdigest()[:20]
    f = enc.cache / f"{enc.name}_{'q' if is_query else 'd'}_{key}.npy"
    if f.exists():
        return np.load(f)
    x = enc.cache / "export" / f"{enc.name}_used_B.npz"
    if ids is not None and x.exists():
        z = np.load(x, allow_pickle=True)
        k, v = ("foa_number", "q") if is_query else ("doc_id", "d")
        pos = pd.Series(np.arange(len(z[k])), index=z[k].astype(str))
        return z[v][pos.loc[[str(i) for i in ids]].values]
    raise SystemExit(f"no cached embeddings for this input ({f.name}); this script is CPU-only")


def pi_ids(s):
    return [int(x) for x in re.findall(r"\d+", s or "")]


def rank_of(scores, i):
    """1-based rank of column i in a score vector (ties counted against it)."""
    return int((scores > scores[i]).sum()) + 1


def recall(ranks, ks):
    r = np.asarray(ranks)
    out = {f"R@{k}": round(float((r <= k).mean()), 3) for k in ks}
    out.update({"MRR": round(float((1 / r).mean()), 3), "median_rank": float(np.median(r)), "n": len(r)})
    return out


def history(data, enc, proj):
    """The pre-FY2022 history corpus (train + val abstracts, one row per distinct abstract) from the
    embedding cache, and its doc -> PI pairs (every listed PI) sorted by PI for reduceat."""
    s = data.splits
    train = s[s.split == "train"].drop_duplicates("abstract_hash").reset_index(drop=True)
    E_train = cached(enc, data.docs.loc[train.application_id, "doc"].tolist(), False, train.application_id)
    _, _, val_ids, val_text, R_val, _ = data.eval_a("val", "B")
    E_val_all = cached(enc, val_text, False, val_ids)
    n_val = int(R_val.any(0).sum())                       # relevant docs come first in eval_a
    assert R_val[:, n_val:].sum() == 0 and R_val[:, :n_val].any(0).all()
    val = s[s.application_id.isin(set(val_ids[:n_val]))].drop_duplicates("application_id")
    val = val.set_index("application_id").loc[val_ids[:n_val]].reset_index()
    keep = ~val.abstract_hash.isin(set(train.abstract_hash))
    hist = pd.concat([train, val[keep.values]], ignore_index=True)
    E_hist = np.concatenate([E_train, E_val_all[:n_val][keep.values]])
    assert (hist.fy <= 2021).all() and len(hist) == len(E_hist)

    hp = hist[["application_id", "org_ipf_code"]].assign(row=np.arange(len(hist)))
    hp["pi"] = hp.application_id.map(lambda x: pi_ids(proj.PI_IDS.get(x, "")))
    hp = hp.explode("pi").dropna(subset=["pi"]).astype({"pi": int}).sort_values(["pi", "row"]).reset_index(drop=True)
    return hist, E_hist, hp


def email_framing(S_pi, win, pi_pos, hp, hist, ns=(3, 5), office_min_pis=50):
    """Results counted per email sent. W[c, p] = PI p won call c (winners with history only).
    PI email: every PI with history gets their top-n calls. Office email: for each call, each
    institution with at least office_min_pis PIs gets its top-n local PIs (a PI belongs to the
    institution of their latest history award; smaller institutions would list nearly everyone)."""
    W = np.zeros(S_pi.shape, dtype=bool)
    w = win[win.has_hist]
    W[w.ci.values, w.pi.map(pi_pos).values] = True
    out = {"base_rate_per_suggestion": float(W.mean())}
    winners = W.any(0)
    for n in ns:
        top = np.argpartition(-S_pi, n - 1, axis=0)[:n]                       # n x PIs
        hits = np.take_along_axis(W, top, axis=0)
        per_email = hits.sum(0)
        out[f"pi_email_top{n}"] = {
            "emails_all_PIs": int(S_pi.shape[1]),
            "suggestions_won_all_PIs": round(float(hits.mean()), 4),
            "lift_over_random": round(float(hits.mean() / W.mean()), 1),
            "emails_to_test_winners": int(winners.sum()),
            "share_of_winner_emails_with_a_won_call": round(float((per_email[winners] > 0).mean()), 3),
            "suggestions_won_winner_emails": round(float(hits[:, winners].mean()), 3),
            "winner_emails_by_calls_won": {int(k): int(v) for k, v in
                                           zip(*np.unique(per_email[winners], return_counts=True))},
        }
    latest = hp.assign(fy=hist.fy.values[hp.row.values]).sort_values(["pi", "fy"]).groupby("pi").org_ipf_code.last()
    org_of = pd.Series(latest.values, index=[pi_pos[p] for p in latest.index])
    groups = [np.array(ix) for ix in org_of.groupby(org_of).groups.values() if len(ix) >= office_min_pis]
    for n in ns:
        listed = won = lists = lists_hit = winner_lists = 0
        for cols in groups:
            sc, Wg = S_pi[:, cols], W[:, cols]
            k = min(n, len(cols))
            top = np.argpartition(-sc, k - 1, axis=1)[:, :k]
            h = np.take_along_axis(Wg, top, axis=1)
            listed += h.size
            won += int(h.sum())
            has = Wg.any(1)                      # calls this institution's PIs won
            winner_lists += int(has.sum())
            lists_hit += int(h.any(1)[has].sum())
            lists += sc.shape[0]
        out[f"office_email_top{n}"] = {
            "institutions": len(groups), "min_PIs": office_min_pis,
            "PIs_covered": int(sum(map(len, groups))), "lists": lists,
            "suggestions_won": round(won / listed, 4),
            "lift_over_random": round(won / listed / W.mean(), 1),
            "lists_where_institution_had_a_winner": winner_lists,
            "share_of_those_lists_naming_a_winner": round(lists_hit / winner_lists, 3),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emails", type=Path, default=Path(__file__).parent / "emails",
                    help="git-ignored directory for example emails")
    ap.add_argument("--n-emails", type=int, default=3)
    ap.add_argument("--size", choices=sorted(TAGS), default="4b")
    a = ap.parse_args()
    if a.size != "0.6b":
        a.emails = a.emails / a.size
    e = a.emails.resolve()
    if ROOT in e.parents and subprocess.run(["git", "-C", str(ROOT), "check-ignore", "-q", str(e / "x.txt")]).returncode:
        raise SystemExit(f"{e} is inside the repository but not git-ignored (emails contain titles and PI IDs)")

    data = Data(DATA)
    s = data.splits
    enc = Dense("unused", tag=TAGS[a.size], cache=DATA / "cache")
    proj = pd.read_parquet(ROOT / "grants" / "derived" / "projects.parquet",
                           columns=["APPLICATION_ID", "PI_IDS", "PI_NAMES", "FULL_PROJECT_NUM", "PROJECT_TITLE", "FY"])
    proj = proj.set_index(proj.APPLICATION_ID.astype(str))

    hist, E_hist, hp = history(data, enc, proj)
    pis, starts = np.unique(hp.pi.values, return_index=True)
    pi_pos = {p: i for i, p in enumerate(pis)}
    pi_orgs = hp.groupby("pi").org_ipf_code.agg(set)

    # ---- calls: both test slices, cached query embeddings
    calls, Q = [], []
    for sl in SLICES:
        foas, q_text, *_ = data.eval_a(sl, "B")
        calls += [(f, sl) for f in foas]
        Q.append(cached(enc, q_text, True, foas))
    calls = pd.DataFrame(calls, columns=["foa_number", "slice"])
    Q = np.concatenate(Q)

    S_doc = Q @ E_hist.T                                  # calls x history docs
    S_pairs = S_doc[:, hp.row.values]
    S_pi = np.maximum.reduceat(S_pairs, starts, axis=1)   # calls x PIs (max over the PI's abstracts)
    print(f"history: {len(hist)} abstracts, {len(pis)} PIs; calls: {len(calls)}", flush=True)

    # ---- winners: every PI on a test-slice award
    win = s[s.split.isin(SLICES)][["foa_number", "split", "application_id", "org_ipf_code", "foa_type"]].copy()
    win["pi"] = win.application_id.map(lambda x: pi_ids(proj.PI_IDS.get(x, "")))
    win = win.explode("pi").dropna(subset=["pi"]).astype({"pi": int})
    win = win.drop_duplicates(["foa_number", "pi"]).reset_index(drop=True)
    call_pos = {f: i for i, f in enumerate(calls.foa_number)}
    win["ci"] = win.foa_number.map(call_pos)
    win["has_hist"] = win.pi.isin(pi_pos)
    win["has_hist_at_org"] = [p in pi_pos and o in pi_orgs[p] for p, o in zip(win.pi, win.org_ipf_code)]

    # ---- views
    rows = []
    for w in win[win.has_hist].itertuples():
        sc, j = S_pi[w.ci], pi_pos[w.pi]
        r = {"slice": w.split, "type": type_group(w.foa_type), "nat": rank_of(sc, j),
             "email": rank_of(S_pi[:, j], w.ci), "inst": None, "inst_pool": None}
        if w.has_hist_at_org:
            pool = np.array([pi_pos[p] for p in hp.pi[hp.org_ipf_code == w.org_ipf_code].unique()])
            r["inst"] = int((sc[pool] > sc[j]).sum()) + 1
            r["inst_pool"] = len(pool)
        rows.append(r)
    R = pd.DataFrame(rows)

    def summary(df, w):
        ins = df.dropna(subset=["inst"])
        return {
            "winner_pairs": len(w), "with_history": int(w.has_hist.sum()),
            "coverage": round(float(w.has_hist.mean()), 3),
            "with_history_at_award_institution": int(w.has_hist_at_org.sum()),
            "national": {**recall(df.nat, KS_PI), "pool_PIs": len(pis),
                         "chance_R@100": round(100 / len(pis), 4)},
            "institution": {**recall(ins.inst, KS_INST),
                            "median_pool_PIs": float(ins.inst_pool.median()),
                            "chance_R@3_median_pool": round(3 / ins.inst_pool.median(), 3)},
            "email": {**recall(df.email, KS_EMAIL), "calls_ranked": len(calls),
                      "chance_R@3": round(3 / len(calls), 4)},
        }

    out = {"design": __doc__.split("Usage")[0].strip(), "model": TAGS[a.size], "history_abstracts": len(hist), "history_PIs": len(pis),
           "calls": {sl: int((calls.slice == sl).sum()) for sl in SLICES},
           "all": summary(R, win)}
    for sl in SLICES:
        out[sl] = summary(R[R.slice == sl], win[win.split == sl])
        out[sl]["by_type"] = {t: summary(R[(R.slice == sl) & (R.type == t)],
                                         win[(win.split == sl) & (win.foa_type.map(type_group) == t)])
                              for t in sorted(R[R.slice == sl].type.unique())}
    out["email_framing"] = email_framing(S_pi, win, pi_pos, hp, hist)
    res = Path(__file__).parent / "results"
    res.mkdir(exist_ok=True)
    (res / ("backtest.json" if a.size == "0.6b" else f"backtest_{a.size}.json")).write_text(json.dumps(out, indent=2))
    for k in ("all",) + SLICES:
        o = out[k]
        print(f"\n[{k}] coverage {o['coverage']} ({o['with_history']}/{o['winner_pairs']} winner pairs)")
        for v in ("national", "institution", "email"):
            print(f"  {v:12s} {json.dumps(o[v])}")

    # ---- example emails: winners (new-call slice) whose won call ranked in their top 3
    titles = pd.read_parquet(ROOT / "grants" / "derived" / "foa_text.parquet",
                             columns=["foa_number", "title"]).drop_duplicates("foa_number").set_index("foa_number").title
    a.emails.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261103)
    hits = win[win.has_hist & (win.split == "test")]
    hits = hits[[rank_of(S_pi[:, pi_pos[p]], c) <= 3 for p, c in zip(hits.pi, hits.ci)]]
    chosen = hits.drop_duplicates("pi").sample(min(a.n_emails, hits.pi.nunique()), random_state=rng.integers(1 << 31))
    for n, w in enumerate(chosen.itertuples(), 1):
        j = pi_pos[w.pi]
        rows_j = hp.row.values[starts[j]:(starts[j + 1] if j + 1 < len(starts) else len(hp))]
        top = np.argsort(-S_pi[:, j])[:3]
        lines = ["RETROSPECTIVE DEMO: generated from public NIH data as if sent at the start of FY2022.",
                 "Not sent to anyone.", "", "Dear Dr. [PI],", "",
                 "We found three NIH funding opportunities that closely match your previous work "
                 "and may be a good fit:", ""]
        for k, ci in enumerate(top, 1):
            f = calls.foa_number[ci]
            d = rows_j[np.argmax(S_doc[ci, rows_j])]
            app = hist.application_id[d]
            pr = proj.loc[app]
            lines += [f"{k}. {f}: {titles.get(f, '(title unavailable)')}",
                      f"   Closest match in your past work: {pr.FULL_PROJECT_NUM} (FY{pr.FY}) {pr.PROJECT_TITLE}",
                      f"   https://reporter.nih.gov/project-details/{app}",
                      f"   Similarity {S_doc[ci, d]:.3f}", ""]
        won = calls.foa_number[w.ci]
        lines += ["---", f"Outcome (not part of the email): this PI later won {won}, "
                  f"ranked #{rank_of(S_pi[:, j], w.ci)} of {len(calls)} calls.",
                  f"PI profile ID {w.pi}; {len(rows_j)} history abstracts."]
        (a.emails / f"email_{n}.txt").write_text("\n".join(lines) + "\n")
    print(f"\n{len(chosen)} example emails -> {a.emails}")


if __name__ == "__main__":
    main()
