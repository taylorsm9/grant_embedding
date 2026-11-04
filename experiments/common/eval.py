"""Shared evaluation harness (Plan 1 step 4). Every plan evaluates through this file.

Usage:
  python -I eval.py --data DIR --split val|test|test_reissue --variant A|B --rankers bm25,dense,hybrid [--model PATH] [--tag NAME]
  python -I eval.py --data DIR --split val --variant A --sanity

DIR holds splits.parquet, queries.parquet, docs.parquet and foa_meta.parquet (the
frozen files in experiments/common, or a staging copy); inst_corpus.parquet is optional
(Eval B only).

Eval A (standard retrieval): queries are the split's FOAs that have winners outside
the held-out institutions. Corpus: those winners' abstracts plus an equal number of
distractors (seeded, deduplicated by abstract text) from the same period where possible
(DISTRACTORS): val draws from train; the new-call test slice draws from the reissue
slice; the reissue slice draws from the new-call slice, then the most recent train years.
Metrics: recall@10/50/100 (fraction of the FOA's awards in the top k; the oracle's
maximum is min(k, n_rel) / n_rel), hit@10/50, MRR (full ranking), nDCG@10 (binary).

Eval B (deployment simulation, test-style splits): for each held-out institution and
each FOA with a winner there, the corpus is that institution's abstracts funded before
the FOA's release date, minus every award under the FOA. Labs are contact PIs; a lab
scores the max (and mean of top 3) over its abstracts. Relevant labs are the FOA's
winning contact PIs at that institution; winners with no pre-call abstracts are dropped
and counted. Metrics: lab recall@10/50, MRR.

Rankers implement score(q_texts, d_texts) -> float32 [n_q, n_d] (higher is better).
"""
import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

sys.path.insert(0, str(Path(__file__).resolve().parent))
import text  # noqa: E402

SEED = 20261102
KS = (10, 50, 100)
RRF_K = 60
SIZE_BINS = [(5, 9, "small 5-9"), (10, 29, "medium 10-29"), (30, 200, "large 30-200")]
DISTRACTORS = {"val": ["train"], "test": ["test_reissue"], "test_reissue": ["test", "train"]}


# --------------------------------------------------------------------------- data

class Data:
    def __init__(self, d):
        d = Path(d)
        self.dir = d
        self.splits = pd.read_parquet(d / "splits.parquet")
        self.queries = pd.read_parquet(d / "queries.parquet").set_index("foa_number")
        self.docs = pd.read_parquet(d / "docs.parquet").set_index("application_id")
        self.meta = pd.read_parquet(d / "foa_meta.parquet").set_index("foa_number")
        self.inst = pd.read_parquet(d / "inst_corpus.parquet") if (d / "inst_corpus.parquet").exists() else None

    def eval_a(self, split, variant, seed=SEED, exclude=()):
        """exclude: application_ids removed from the relevant set and corpus (near-duplicate check)."""
        s = self.splits
        rel = s[(s.split == split) & ~s.heldout_inst & ~s.application_id.isin(set(exclude))]
        n = rel.application_id.nunique()
        taken, used, need = [], set(rel.abstract_hash), n
        sources = DISTRACTORS[split]
        for src in sources:
            pool = s[(s.split == src) & ~s.heldout_inst & ~s.abstract_hash.isin(used)].drop_duplicates("abstract_hash")
            pool = pool.sample(frac=1.0, random_state=seed)
            if src == "train" and len(sources) > 1:   # top-up: closest years first
                pool = pool.sort_values("fy", ascending=False, kind="stable")
            take = pool.head(need)
            taken.append(take)
            used |= set(take.abstract_hash)
            need -= len(take)
            if need == 0:
                break
        assert need == 0, f"distractor pools short by {need} for {n} relevant docs"
        distract = pd.concat(taken)
        doc_ids = np.concatenate([rel.application_id.unique(), distract.application_id.values])
        foas = np.array(sorted(rel.foa_number.unique()))
        q_text = self.queries.loc[foas, f"query_{variant}"].tolist()
        d_text = self.docs.loc[doc_ids, "doc"].tolist()
        pos = {a: i for i, a in enumerate(doc_ids)}
        qpos = {f: i for i, f in enumerate(foas)}
        R = np.zeros((len(foas), len(doc_ids)), dtype=bool)
        R[rel.foa_number.map(qpos).values, rel.application_id.map(pos).values] = True
        info = {"split": split, "variant": variant, "n_queries": len(foas), "n_docs": len(doc_ids),
                "n_relevant_docs": int(n), "n_distractors": int(n), "distractor_seed": seed,
                "distractors_by_source": {src: len(t) for src, t in zip(sources, taken)},
                "distractor_fy_range": [int(distract.fy.min()), int(distract.fy.max())],
                "n_excluded_near_dups": len(set(exclude))}
        return foas, q_text, doc_ids, d_text, R, info

    def release_date(self, foa):
        m = self.meta.loc[foa]
        if pd.notna(m.foa_release_date):
            return m.foa_release_date, False
        # Conservative fallback: one year before the start of the first-award fiscal year.
        return pd.Timestamp(f"{int(m.foa_first_award_fy) - 2}-10-01"), True

    def eval_b(self, split, variant):
        """Yields per (institution, FOA) problems over the institution's own corpus."""
        s = self.splits
        wins = s[(s.split == split) & s.heldout_inst]
        problems = []
        for ipf, inst in self.inst.groupby("org_ipf_code"):
            w = wins[wins.org_ipf_code == ipf]
            for foa, wf in w.groupby("foa_number"):
                release, fallback = self.release_date(foa)
                c = inst[(inst.funded_date < release) & (inst.foa_number != foa)]
                labs = set(c.contact_pi_id)
                winners = set(wf.contact_pi_id)
                problems.append({"ipf": ipf, "foa": foa, "release": release, "release_fallback": fallback,
                                 "corpus_idx": c.index.values, "winners": winners,
                                 "winners_kept": winners & labs, "n_labs": len(labs)})
        return problems


# ------------------------------------------------------------------------ rankers

TOKEN = re.compile(r"(?u)\b[a-z0-9][a-z0-9\-]+\b")


class BM25:
    """Okapi BM25 over lowercase alphanumeric tokens (len >= 2), sklearn English stop
    words removed, no stemming. k1 = 1.2, b = 0.75 (Lucene defaults)."""
    name = "bm25"

    def __init__(self, k1=1.2, b=0.75):
        self.k1, self.b = k1, b

    def score(self, q_texts, d_texts):
        vec = CountVectorizer(lowercase=True, token_pattern=TOKEN.pattern, stop_words=list(ENGLISH_STOP_WORDS))
        tf = vec.fit_transform(d_texts).tocsc().astype(np.float32)
        n = tf.shape[0]
        df = np.diff(tf.indptr)
        idf = np.log1p((n - df + 0.5) / (df + 0.5)).astype(np.float32)
        dl = np.asarray(tf.sum(axis=1)).ravel()
        norm = self.k1 * (1 - self.b + self.b * dl / dl.mean())
        tf = tf.tocsr()
        rows = np.repeat(np.arange(n), np.diff(tf.indptr))
        data = tf.data * (self.k1 + 1) / (tf.data + norm[rows])
        w = sp.csr_matrix((data, tf.indices, tf.indptr), shape=tf.shape).multiply(idf).tocsr()
        q = (vec.transform(q_texts) > 0).astype(np.float32)
        return np.asarray((q @ w.T).todense(), dtype=np.float32)


class Dense:
    """Sentence-transformers bi-encoder with the fixed instruction on queries only."""

    def __init__(self, model_path, tag=None, cache=None, q_batch=16, d_batch=16):
        self.model_path, self.name = model_path, tag or "dense"
        self.cache = Path(cache) if cache else None
        self.q_batch, self.d_batch = q_batch, d_batch
        self._model = None

    def model(self):
        if self._model is None:
            import torch
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(
                self.model_path, device="cuda",
                model_kwargs={"attn_implementation": "sdpa", "dtype": torch.bfloat16})
        return self._model

    def encode(self, texts, is_query):
        key = hashlib.sha1(("\x00".join([self.name, str(is_query)] + list(texts))).encode()).hexdigest()[:20]
        f = self.cache / f"{self.name}_{'q' if is_query else 'd'}_{key}.npy" if self.cache else None
        if f is not None and f.exists():
            return np.load(f)
        m = self.model()
        m.max_seq_length = text.QUERY_CAP if is_query else text.DOC_CAP
        t0 = time.time()
        e = m.encode(list(texts), prompt=text.QUERY_PROMPT if is_query else None,
                     batch_size=self.q_batch if is_query else self.d_batch,
                     normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        e = e.astype(np.float32)
        print(f"  [{self.name}] encoded {len(texts)} {'queries' if is_query else 'docs'} "
              f"in {time.time() - t0:.0f}s", flush=True)
        if f is not None:
            f.parent.mkdir(parents=True, exist_ok=True)
            np.save(f, e)
        return e

    def score(self, q_texts, d_texts):
        return self.encode(q_texts, True) @ self.encode(d_texts, False).T


class Random:
    name = "random"

    def __init__(self, seed=SEED):
        self.rng = np.random.default_rng(seed)

    def score(self, q_texts, d_texts):
        return self.rng.random((len(q_texts), len(d_texts)), dtype=np.float32)


def ranks_of(S):
    """1-based rank of every doc for every query (ties broken by position)."""
    order = np.argsort(-S, axis=1, kind="stable")
    r = np.empty_like(order)
    np.put_along_axis(r, order, np.arange(1, S.shape[1] + 1)[None, :].repeat(S.shape[0], 0), axis=1)
    return r


def rrf(*score_mats, k=RRF_K):
    return sum(1.0 / (k + ranks_of(S)) for S in score_mats).astype(np.float32)


# ------------------------------------------------------------------------ metrics

def metrics_a(S, R):
    """Per-query metrics from scores S and boolean relevance R, both [n_q, n_d]."""
    order = np.argsort(-S, axis=1, kind="stable")[:, :max(KS)]
    hits = np.take_along_axis(R, order, axis=1)
    n_rel = R.sum(1)
    out = {"n_rel": n_rel}
    for k in KS:
        out[f"recall@{k}"] = hits[:, :k].sum(1) / n_rel
    for k in (10, 50):
        out[f"hit@{k}"] = hits[:, :k].any(1).astype(float)
    best_rel = np.where(R, S, -np.inf).max(1)
    first = 1 + (S > best_rel[:, None]).sum(1)
    out["mrr"] = 1.0 / first
    disc = 1.0 / np.log2(np.arange(2, 12))
    dcg = (hits[:, :10] * disc).sum(1)
    idcg = np.array([disc[:min(n, 10)].sum() for n in n_rel])
    out["ndcg@10"] = dcg / idcg
    return pd.DataFrame(out)


def type_group(t):
    return "PA/PAS" if t in ("PA", "PAS") else t


def size_group(n):
    for lo, hi, label in SIZE_BINS:
        if lo <= n <= hi:
            return label
    return "other"


def summarise_a(per_q, foas, meta):
    cols = [c for c in per_q.columns if c != "n_rel"]
    per_q = per_q.assign(foa=foas,
                         type=[type_group(meta.loc[f, "foa_type"]) for f in foas],
                         size=[size_group(meta.loc[f, "foa_n_awards"]) for f in foas],
                         year=[int(meta.loc[f, "foa_first_award_fy"]) for f in foas])
    res = {"all": {**per_q[cols].mean().round(4).to_dict(), "n_queries": len(per_q)}}
    # First-award year only for test-style splits (few years); val spans FY2008-21.
    slices = ("type", "size", "year") if per_q.year.nunique() <= 5 else ("type", "size")
    for by in slices:
        for g, d in per_q.groupby(by):
            res[f"{by}={g}"] = {**d[cols].mean().round(4).to_dict(), "n_queries": len(d)}
    return res


def eval_b_scores(ranker, data, problems, foas_q, q_text, cache, seed=SEED):
    """Score each (institution, FOA) problem; one score matrix per institution."""
    rows = []
    for ipf, probs in pd.DataFrame(problems).groupby("ipf"):
        inst = data.inst[data.inst.org_ipf_code == ipf]
        need = sorted(set(probs.foa))
        qi = [foas_q.index(f) for f in need]
        S = (ranker.score([q_text[i] for i in qi], inst.doc.tolist())
             if ranker not in (None, "random") else None)
        local = {idx: j for j, idx in enumerate(inst.index)}
        for _, p in probs.iterrows():
            if not p.winners_kept:
                rows.append({**p.to_dict(), "skip": True})
                continue
            cols = [local[i] for i in p.corpus_idx]
            labs = inst.contact_pi_id.values[cols]
            if ranker is None:   # oracle
                sc = np.isin(labs, list(p.winners_kept)).astype(np.float32) + 0.0
            elif ranker == "random":
                # One random score per lab, so every lab is equally likely (uniform chance);
                # a per-abstract random score would favour labs with many abstracts under max.
                u, inv = np.unique(labs, return_inverse=True)
                sc = np.random.default_rng([seed, len(rows)]).random(len(u))[inv]
            else:
                sc = S[need.index(p.foa), cols]
            rows.append({**p.to_dict(), "skip": False, "lab_scores": lab_rank(labs, sc)})
    return rows


def lab_rank(labs, sc):
    d = pd.DataFrame({"lab": labs, "s": sc})
    mx = d.groupby("lab").s.max()
    top3 = d.groupby("lab").s.apply(lambda x: np.sort(x.values)[-3:].mean())
    return {"max": mx, "top3": top3}


def metrics_b(rows):
    res = {}
    kept = [r for r in rows if not r["skip"]]
    n_win = sum(len(r["winners"]) for r in rows)
    n_kept = sum(len(r["winners_kept"]) for r in rows)
    for agg in ("max", "top3"):
        per = []
        for r in kept:
            s = r["lab_scores"][agg].sort_values(ascending=False, kind="stable")
            rank = pd.Series(np.arange(1, len(s) + 1), index=s.index)
            wr = rank[list(r["winners_kept"])]
            per.append({"recall@10": (wr <= 10).mean(), "recall@50": (wr <= 50).mean(),
                        "mrr": 1.0 / wr.min(), "n_labs": len(s)})
        df = pd.DataFrame(per)
        res[agg] = {**df.drop(columns="n_labs").mean().round(4).to_dict(),
                    "n_queries": len(df), "mean_candidate_labs": round(float(df.n_labs.mean()), 1)}
    res["winners_total"] = n_win
    res["winners_dropped_no_precall_abstracts"] = n_win - n_kept
    res["winners_dropped_frac"] = round((n_win - n_kept) / max(n_win, 1), 4)
    res["problems_skipped_all_winners_dropped"] = len(rows) - len(kept)
    res["problems_release_date_fallback"] = int(sum(r["release_fallback"] for r in rows))
    return res


# --------------------------------------------------------------------------- runs

def top_candidates(S, foas, doc_ids, source, k=100):
    """Top-k documents per query, schema shared with Plan 3: query_id, doc_id, rank, score, source."""
    order = np.argsort(-S, axis=1, kind="stable")[:, :k]
    return pd.DataFrame({"query_id": np.repeat(foas, order.shape[1]),
                         "doc_id": np.asarray(doc_ids)[order].ravel(),
                         "rank": np.tile(np.arange(1, order.shape[1] + 1), len(foas)),
                         "score": np.take_along_axis(S, order, axis=1).ravel().astype(np.float32),
                         "source": source})


def run(data, split, variant, rankers, out_dir, do_eval_b=True, exclude=(), stem_suffix="", candidates=None):
    foas, q_text, doc_ids, d_text, R, info = data.eval_a(split, variant, exclude=exclude)
    out = {"info": info, "results": {}}
    scores, cands = {}, []
    for r in rankers:
        t0 = time.time()
        if r.name == "hybrid":
            S = rrf(scores["bm25"], scores[r.dense])
        else:
            S = r.score(q_text, d_text)
        scores[r.name] = S
        if candidates:
            cands.append(top_candidates(S, foas, doc_ids, r.name))
        res = {"eval_a": summarise_a(metrics_a(S, R), foas, data.meta)}
        if do_eval_b:
            res["eval_b"] = run_eval_b(data, split, variant, r, scores)
        res["seconds"] = round(time.time() - t0, 1)
        out["results"][r.name] = res
        print(f"{r.name}: {json.dumps(res['eval_a']['all'])}", flush=True)
    write(out, out_dir, f"{split}_{variant}{stem_suffix}")
    if candidates:
        c = pd.concat(cands, ignore_index=True).assign(split=split)
        c.to_parquet(candidates, index=False)
        print(f"candidates: {len(c)} rows -> {candidates}")
    return out, scores, (foas, doc_ids, R)


def run_eval_b(data, split, variant, r, scores):
    problems = data.eval_b(split, variant)
    foas_q = sorted({p["foa"] for p in problems})
    q_text = data.queries.loc[foas_q, f"query_{variant}"].tolist()
    if r.name == "hybrid":
        ranker = HybridB(data, r, foas_q, q_text)
    else:
        ranker = r
    return metrics_b(eval_b_scores(ranker, data, problems, foas_q, q_text, None))


class HybridB:
    """RRF of BM25 and a dense ranker over an institution corpus (for Eval B)."""

    def __init__(self, data, r, foas_q, q_text):
        self.parts = [BM25(), r.dense_ranker]

    def score(self, q_texts, d_texts):
        return rrf(*(p.score(q_texts, d_texts) for p in self.parts))


class Hybrid:
    name = "hybrid"

    def __init__(self, dense_ranker):
        self.dense = dense_ranker.name
        self.dense_ranker = dense_ranker


def write(out, out_dir, stem):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{stem}.json").write_text(json.dumps(out, indent=2, default=str))
    lines = [f"# {stem}: {json.dumps(out['info'])}", "",
             "| ranker | slice | n_q | R@10 | R@50 | R@100 | hit@10 | hit@50 | MRR | nDCG@10 |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, res in out["results"].items():
        for sl, m in res["eval_a"].items():
            lines.append(f"| {name} | {sl} | {m['n_queries']} | {m['recall@10']:.3f} | {m['recall@50']:.3f} | "
                         f"{m['recall@100']:.3f} | {m['hit@10']:.3f} | {m['hit@50']:.3f} | {m['mrr']:.3f} | "
                         f"{m['ndcg@10']:.3f} |")
    if any("eval_b" in r for r in out["results"].values()):
        lines += ["", "Eval B (lab level, held-out institutions)", "",
                  "| ranker | lab score | n_q | labs | R@10 | R@50 | MRR | winners dropped |",
                  "|---|---|---:|---:|---:|---:|---:|---:|"]
        for name, res in out["results"].items():
            b = res.get("eval_b")
            if not b:
                continue
            for agg in ("max", "top3"):
                m = b[agg]
                lines.append(f"| {name} | {agg} | {m['n_queries']} | {m['mean_candidate_labs']} | "
                             f"{m['recall@10']:.3f} | {m['recall@50']:.3f} | {m['mrr']:.3f} | "
                             f"{b['winners_dropped_frac']:.1%} |")
    (out_dir / f"{stem}.md").write_text("\n".join(lines) + "\n")


def sanity(data, split, variant, out_dir):
    """Random near chance, oracle at its maximum, shuffled labels collapse (Eval A and B)."""
    foas, q_text, doc_ids, d_text, R, info = data.eval_a(split, variant)
    lines = [f"sanity on {split}/{variant}: {info}"]
    n_d = R.shape[1]
    n_rel = R.sum(1)

    # Random ranker averaged over 20 seeds; must sit within 4 standard errors of k / n_docs.
    per = pd.concat([metrics_a(Random(SEED + i).score(q_text, d_text), R) for i in range(20)])
    rnd = per.drop(columns="n_rel").mean().round(5).to_dict()
    chance = {f"recall@{k}": round(k / n_d, 5) for k in KS}
    lines.append(f"random (mean of 20 seeds): {rnd}")
    lines.append(f"chance recall@k = k/n_docs: {chance}")
    for k in KS:
        se = per[f"recall@{k}"].std() / np.sqrt(len(per))
        assert abs(rnd[f"recall@{k}"] - k / n_d) < 4 * se + 1e-4, (k, rnd[f"recall@{k}"], k / n_d, se)

    orc = summarise_a(metrics_a(R.astype(np.float32), R), foas, data.meta)["all"]
    lines.append(f"oracle: {orc}")
    for k in KS:
        mx = float(np.mean(np.minimum(k, n_rel) / n_rel))
        assert abs(orc[f"recall@{k}"] - mx) < 1e-4, (k, orc[f"recall@{k}"], mx)
    for m in ("hit@10", "hit@50", "mrr", "ndcg@10"):
        assert orc[m] == 1.0, (m, orc[m])

    # Shuffled-label check needs a real ranker's scores; run it on val only so no test
    # metrics of a real ranker are seen before the final evaluation.
    if split == "val":
        S = BM25().score(q_text, d_text)
        real = summarise_a(metrics_a(S, R), foas, data.meta)["all"]
        perm = np.random.default_rng(SEED).permutation(len(foas))
        shuf = summarise_a(metrics_a(S, R[perm]), foas, data.meta)["all"]
        lines.append(f"bm25 real labels: {real}")
        lines.append(f"bm25 shuffled labels: {shuf}")
        assert shuf["recall@100"] < 0.25 * real["recall@100"], "shuffled labels did not collapse"
    lines.append("EVAL A SANITY CHECKS PASSED")

    if data.inst is not None and data.splits.heldout_inst.any():  # Eval B needs held-out winners
        problems = data.eval_b(split, variant)
        foas_q = sorted({p["foa"] for p in problems})
        qt = data.queries.loc[foas_q, f"query_{variant}"].tolist()
        o = metrics_b(eval_b_scores(None, data, problems, foas_q, qt, None))
        rows0 = [r for r in eval_b_scores(None, data, problems, foas_q, qt, None) if not r["skip"]]
        expect = float(np.mean([min(10, len(r["lab_scores"]["max"])) / len(r["lab_scores"]["max"]) for r in rows0]))
        reps = [metrics_b(eval_b_scores("random", data, problems, foas_q, qt, None, seed=SEED + i))["max"]
                for i in range(20)]
        r10 = np.array([x["recall@10"] for x in reps])
        lines.append(f"eval B oracle: {o['max']}")
        lines.append(f"eval B random recall@10 (20 seeds): mean {r10.mean():.5f}, sd {r10.std():.5f}; "
                     f"expected mean(10/labs) = {expect:.5f}")
        assert o["max"]["mrr"] == 1.0 and o["max"]["recall@10"] >= 0.99
        assert abs(r10.mean() - expect) < 4 * r10.std() / np.sqrt(len(r10)) + 1e-3
        lines.append("EVAL B SANITY CHECKS PASSED")
    print("\n".join(lines))
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / f"sanity_{split}_{variant}.txt").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--variant", default="A")
    ap.add_argument("--rankers", default="bm25")
    ap.add_argument("--model", default="models/Qwen3-Embedding-0.6B")
    ap.add_argument("--tag", default="qwen3-0.6b-zs")
    ap.add_argument("--out", default=None)
    ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--no-eval-b", action="store_true")
    ap.add_argument("--candidates", default=None, help="write top-100 per query for every ranker (parquet)")
    ap.add_argument("--near-dup-max", type=float, default=None,
                    help="drop split abstracts whose zero-shot max cosine to train >= this (near_dups.py)")
    a = ap.parse_args()
    data = Data(a.data)
    out = a.out or str(Path(a.data) / "results")
    if a.sanity:
        return sanity(data, a.split, a.variant, out)
    rankers, dense = [], None
    for name in a.rankers.split(","):
        if name == "bm25":
            rankers.append(BM25())
        elif name == "dense":
            dense = Dense(a.model, tag=a.tag, cache=Path(a.data) / "cache")
            rankers.append(dense)
        elif name == "hybrid":
            assert dense is not None and "bm25" in [r.name for r in rankers], "hybrid needs bm25 and dense first"
            rankers.append(Hybrid(dense))
        elif name == "random":
            rankers.append(Random())
    exclude, suffix = (), ""
    if a.near_dup_max is not None:
        nd = pd.read_parquet(Path(a.data) / f"near_dups_{a.split}.parquet")
        exclude = nd.application_id[nd.max_cos >= a.near_dup_max].tolist()
        suffix = f"_nodup{a.near_dup_max}"
    do_b = not a.no_eval_b and data.inst is not None and data.splits.heldout_inst.any()
    run(data, a.split, a.variant, rankers, out, do_eval_b=do_b, exclude=exclude, stem_suffix=suffix,
        candidates=a.candidates)


if __name__ == "__main__":
    main()
