"""Group FOAs into reissue families (Plan 1 step 2).

Usage:
  python -I foa_families.py <pairs.parquet> <out.parquet> <threshold>
  python -I foa_families.py <pairs.parquet> --scan     # threshold scan + samples to eyeball

Titles lose their mechanism suffix ("(R01 Clinical Trial Optional)") and
"Limited Competition:" prefix first, since those make unrelated FOAs look alike.
Edges: TF-IDF cosine on title + purpose >= threshold; title-only cosine >= TITLE_T;
and any two FOAs that share a core project number (a renewal funded under a
reissue). Families are the connected components. The direct edges themselves are also
written to <out dir>/foa_links.parquet (foa_a < foa_b): training masks in-batch documents
whose FOA is directly linked to the anchor's FOA.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from scipy.sparse import coo_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import text  # noqa: E402

TITLE_T = 0.8
MECH_SUFFIX = re.compile(r"\((?:[^()]*\b(?:[A-Z]{1,3}\d{2}[A-Z]?|Clinical Trials?|Parent)\b[^()]*)\)\s*$")
PREFIX = re.compile(r"^(?:Limited Competition|Notice of Special Interest|Reissue)\s*:\s*", re.I)


def topic_title(title):
    t = text.clean_title(title)
    for _ in range(2):
        t = MECH_SUFFIX.sub("", t).strip()
    return PREFIX.sub("", t)


def load(pairs_path):
    pairs = pd.read_parquet(pairs_path)
    foas = pairs.drop_duplicates("foa_number").reset_index(drop=True)
    foas["tt"] = [topic_title(t) for t in foas.foa_title]
    foas["tp"] = [f"{tt}\n{text.clean_foa_field(p)}" for tt, p in zip(foas.tt, foas.foa_purpose)]
    return pairs, foas


def cosine(docs):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words="english")
    x = vec.fit_transform(docs)
    sim = (x @ x.T).toarray()
    np.fill_diagonal(sim, 0)
    return sim


def similarity(foas):
    """(title + purpose cosine, title-only cosine), both FOA x FOA."""
    return cosine(foas.tp), cosine(foas.tt)


def project_edges(pairs, foas):
    pos = dict(zip(foas.foa_number, range(len(foas))))
    edges = set()
    for _, g in pairs.groupby("core_project_num"):
        ids = sorted({pos[f] for f in g.foa_number})
        edges.update((a, b) for i, a in enumerate(ids) for b in ids[i + 1:])
    return edges


def edges(sims, threshold, extra_edges):
    tp, tt = sims
    i, j = np.nonzero(np.triu((tp >= threshold) | (tt >= TITLE_T)))
    if extra_edges:
        a, b = zip(*extra_edges)
        i, j = np.concatenate([i, a]), np.concatenate([j, b])
    return i, j


def components(n, sims, threshold, extra_edges):
    i, j = edges(sims, threshold, extra_edges)
    g = coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n))
    return connected_components(g, directed=False)[1]


def describe(labels, foas, pairs):
    sizes = pd.Series(labels).value_counts()
    fam_pairs = pairs.groupby(pairs.foa_number.map(dict(zip(foas.foa_number, labels)))).size()
    return (f"families {sizes.size}, singletons {int((sizes == 1).sum())}, "
            f"largest {int(sizes.max())} FOAs / {int(fam_pairs.max())} pairs uncapped, "
            f"FOAs in families >=2: {int(sizes[sizes > 1].sum())}")


def show(foas, a, b, tp, tt):
    print(f"{tp[a, b]:.2f}/{tt[a, b]:.2f}  {foas.foa_number[a]:<15} {text.clean_title(foas.foa_title[a])[:90]}")
    print(f"           {foas.foa_number[b]:<15} {text.clean_title(foas.foa_title[b])[:90]}")


def scan(pairs, foas, sims, t0=0.45):
    tp, tt = sims
    pe = project_edges(pairs, foas)
    print(f"FOA pairs sharing a core project: {len(pe)}; title-only edges >= {TITLE_T}: "
          f"{int(np.triu(tt >= TITLE_T).sum())}")
    for t in [0.35, 0.4, 0.45, 0.5, 0.6]:
        print(f"t={t:.2f} text edges: {describe(components(len(foas), sims, t, set()), foas, pairs)}")
        print(f"         + projects: {describe(components(len(foas), sims, t, pe), foas, pairs)}")
    rng = np.random.default_rng(0)
    iu = np.triu_indices(len(foas), 1)
    merged = (tp[iu] >= t0) | (tt[iu] >= TITLE_T)
    near = ~merged & (tp[iu] >= 0.30)
    for mask, label in [(merged, f"MERGES (tp >= {t0} or title >= {TITLE_T})"),
                        (near, f"NEAR MISSES (0.30 <= tp < {t0}, title < {TITLE_T})")]:
        idx = np.nonzero(mask)[0]
        print(f"\n=== {label}: {len(idx)} edges; 20 random (tp/title) ===")
        for k in rng.choice(idx, size=min(20, len(idx)), replace=False):
            show(foas, iu[0][k], iu[1][k], tp, tt)
    miss = [(a, b) for a, b in pe if tp[a, b] < t0 and tt[a, b] < TITLE_T]
    print(f"\n=== shared-project edges the text rules miss: {len(miss)}; 10 random ===")
    for k in rng.choice(len(miss), size=min(10, len(miss)), replace=False):
        show(foas, *miss[k], tp, tt)
    sizes = pd.Series(components(len(foas), sims, t0, pe)).value_counts()
    lab = components(len(foas), sims, t0, pe)
    for fam in sizes.index[:3]:
        members = foas.foa_number[lab == fam]
        print(f"\n=== family of {len(members)} FOAs, sample titles ===")
        for f in members[:12]:
            print("  ", f, foas.tt[foas.foa_number == f].iloc[0][:90])


def main(pairs_path, out=None, threshold=None):
    pairs, foas = load(pairs_path)
    sim = similarity(foas)
    if out == "--scan":
        return scan(pairs, foas, sim, float(threshold or 0.45))
    threshold = float(threshold)
    proj = project_edges(pairs, foas)
    labels = components(len(foas), sim, threshold, proj)
    i, j = edges(sim, threshold, proj)
    nums = foas.foa_number.to_numpy(dtype=object)
    lo = nums[i] < nums[j]
    a, b = np.where(lo, nums[i], nums[j]), np.where(lo, nums[j], nums[i])
    links = pd.DataFrame({"foa_a": a, "foa_b": b}).drop_duplicates().sort_values(["foa_a", "foa_b"])
    links.to_parquet(Path(out).parent / "foa_links.parquet", index=False)
    print(f"direct links: {len(links)}")
    fam = pd.DataFrame({"foa_number": foas.foa_number, "family_id": labels})
    # Name each family by its alphabetically first FOA, so ids are stable and readable.
    first = fam.groupby("family_id").foa_number.min()
    fam["family_id"] = fam.family_id.map(first)
    fam.to_parquet(out, index=False)
    print(f"threshold {threshold}: {describe(labels, foas, pairs)}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
