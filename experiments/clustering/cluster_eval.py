"""Do fine-tuned embeddings cluster grants by the call that funded them? CPU only, cached embeddings.

Docs: the winning abstracts of the test-period calls (FY2022-24), never seen in training, scored
per slice (test = new calls, test_reissue = reissues of training programs; new calls are the clean
test). Both models are read from the embedding cache; nothing is encoded.

Fixed design, decided before looking at any result:
- kNN agreement: for each doc, the share of its 10 nearest neighbours (cosine, within the slice)
  with the same call / administering institute / activity code / funding mechanism family.
- k-means with k = number of calls in the slice (seed fixed, 3 inits): NMI and ARI against calls.
- Figure: t-SNE (seed fixed) of the winners of the 10 largest new calls, one panel per model.

Usage (from the repo root):
  CUDA_VISIBLE_DEVICES= .venv-train/bin/python -I experiments/clustering/cluster_eval.py [--size 4b]
Writes experiments/clustering/results/{cluster_eval.json, tsne_new_calls.svg} (no abstract text);
--size 4b writes cluster_eval_4b.json and tsne_new_calls_4b.svg.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "common"))
sys.path.insert(0, str(ROOT / "experiments" / "pi_backtest"))
from eval import Data, Dense  # noqa: E402
from backtest import cached  # noqa: E402

DATA = ROOT / "experiments" / "common" / "frozen"
MODELS = {"off the shelf": "qwen3-{}-zs", "fine-tuned": "qwen3-{}-ft-B"}
SLICES = ("test", "test_reissue")
K_NN = 10
N_FIG_CALLS = 10
SEED = 20261103
OUT = Path(__file__).parent / "results"


def slice_docs(data, split):
    """Winning docs of a slice in eval_a order, with their labels, plus the eval_a doc texts."""
    s = data.splits
    _, _, doc_ids, d_text, R, _ = data.eval_a(split, "B")
    n = int(R.any(0).sum())
    assert R[:, n:].sum() == 0 and R[:, :n].any(0).all()
    lab = (s[s.split == split].drop_duplicates("application_id").set_index("application_id")
           .loc[doc_ids[:n], ["foa_number", "foa_type", "core_project_num"]])
    core = lab.core_project_num.str
    assert core.match(r"^[A-Z][A-Z0-9]{2}[A-Z]{2}\d").all(), "unexpected core project number format"
    lab["activity"] = core[:3]                      # e.g. R01
    lab["ic"] = core[3:5]                           # administering institute code, e.g. CA
    lab["mech"] = core[0]                           # R, U, P, K, ... family
    return lab.reset_index(), d_text, n


def knn_agreement(E, labels, k=K_NN):
    S = E @ E.T
    np.fill_diagonal(S, -np.inf)
    nn = np.argpartition(-S, k, axis=1)[:, :k]
    out = {}
    for name, y in labels.items():
        y = np.asarray(y)
        agree = (y[nn] == y[:, None]).mean()
        # chance: a random neighbour shares the label with probability sum_c n_c(n_c-1) / (N(N-1))
        _, c = np.unique(y, return_counts=True)
        chance = (c * (c - 1)).sum() / (len(y) * (len(y) - 1))
        out[name] = {"agreement": round(float(agree), 3), "chance": round(float(chance), 3)}
    return out


def svg_scatter(panels, calls, path):
    """Two t-SNE panels side by side, one colour per call; hand-written SVG (no plotting library)."""
    pal = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948", "#b07aa1", "#ff9da7",
           "#9c755f", "#5f6b6d"]
    W, H, pad, top = 440, 440, 24, 70
    sw, sh = 2 * W + 3 * pad, top + H + 40 + 22 * ((len(calls) + 1) // 2)
    # explicit width/height: some markdown previews (JetBrains) render viewBox-only SVGs as broken
    parts = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{sw}' height='{sh}' viewBox='0 0 {sw} {sh}' "
             "font-family='Helvetica, Arial, sans-serif' font-size='13'>",
             "<rect width='100%' height='100%' fill='white'/>",
             f"<text x='{pad}' y='28' font-size='17' font-weight='600'>Fine-tuning pulls each call's winners together</text>",
             f"<text x='{pad}' y='50' fill='#555'>t-SNE of the winning abstracts of the {len(calls)} largest new FY2022–24 calls "
             "(unseen in training); colour = call</text>"]
    for i, (title, xy, y) in enumerate(panels):
        x0 = pad + i * (W + pad)
        lo, hi = xy.min(0), xy.max(0)
        p = (xy - lo) / (hi - lo) * (W - 2 * pad) + pad
        parts.append(f"<rect x='{x0}' y='{top}' width='{W}' height='{H}' fill='none' stroke='#ccc'/>")
        parts.append(f"<text x='{x0 + 10}' y='{top + 20}' font-weight='600'>{title}</text>")
        for (px, py), c in zip(p, y):
            parts.append(f"<circle cx='{x0 + px:.1f}' cy='{top + py:.1f}' r='3.2' fill='{pal[c]}' fill-opacity='0.8'/>")
    ly = top + H + 28
    for j, (f, n) in enumerate(calls):
        lx, lyy = pad + (j % 2) * (W + pad), ly + 22 * (j // 2)
        parts.append(f"<circle cx='{lx + 6}' cy='{lyy - 4}' r='5' fill='{pal[j]}'/>")
        parts.append(f"<text x='{lx + 18}' y='{lyy}'>{f} ({n} awards)</text>")
    parts.append("</svg>")
    path.write_text("\n".join(parts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", choices=["0.6b", "4b"], default="4b")
    a = ap.parse_args()
    sfx = "" if a.size == "0.6b" else f"_{a.size}"
    data = Data(DATA)
    encs = {name: Dense("unused", tag=tag.format(a.size), cache=DATA / "cache") for name, tag in MODELS.items()}
    out = {"design": __doc__.split("Usage")[0].strip(), "models": {n: e.name for n, e in encs.items()}}
    fig = None
    for split in SLICES:
        lab, d_text, n = slice_docs(data, split)
        res = {"docs": n, "calls": int(lab.foa_number.nunique())}
        for name, enc in encs.items():
            E = cached(enc, d_text, False)[:n]
            r = knn_agreement(E, {"call": lab.foa_number, "institute": lab.ic,
                                  "activity_code": lab.activity, "mechanism_family": lab.mech})
            km = KMeans(n_clusters=res["calls"], n_init=3, random_state=SEED).fit_predict(E)
            r["kmeans_vs_call"] = {"NMI": round(float(normalized_mutual_info_score(lab.foa_number, km)), 3),
                                   "ARI": round(float(adjusted_rand_score(lab.foa_number, km)), 3)}
            res[name] = r
            print(split, name, json.dumps(r), flush=True)
        out[split] = res
        if split == "test":
            big = lab.foa_number.value_counts().head(N_FIG_CALLS)
            sel = lab.foa_number.isin(big.index).values
            code = lab.foa_number[sel].map({f: i for i, f in enumerate(big.index)}).values
            panels = []
            for name, enc in encs.items():
                E = cached(enc, d_text, False)[:n][sel]
                Z = PCA(n_components=50, random_state=SEED).fit_transform(E)
                xy = TSNE(n_components=2, perplexity=30, init="pca", random_state=SEED).fit_transform(Z)
                panels.append((name.capitalize(), xy, code))
                r = knn_agreement(E, {"call": lab.foa_number[sel]})
                out.setdefault("figure_subset", {"calls": list(big.index), "docs": int(sel.sum())})[name] = r
            fig = (panels, list(big.items()))
    OUT.mkdir(exist_ok=True)
    (OUT / f"cluster_eval{sfx}.json").write_text(json.dumps(out, indent=2))
    svg_scatter(*fig, OUT / f"tsne_new_calls{sfx}.svg")
    print(json.dumps(out.get("figure_subset"), indent=1))


if __name__ == "__main__":
    main()
