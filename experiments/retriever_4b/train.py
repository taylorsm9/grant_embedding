"""Fine-tune Qwen3-Embedding-4B with LoRA on one or more GPUs (Plan 2 step 3).

Launch (one node, G GPUs):
  torchrun --standalone --nproc_per_node=G experiments/retriever_4b/train.py \\
      --data experiments/common/frozen --out experiments/retriever_4b/run_B [--model PATH] [--resume]
Single process also works (python -I train.py ...), on GPU or, for tests, CPU.

Same objective, data, mask and batching as the project's earlier full fine-tune of
Qwen3-Embedding-0.6B (not included here); the differences are LoRA and data parallelism:
- LoRA (r 16, alpha 32, dropout 0) on all attention and MLP projections; the base is
  frozen in bf16, the adapter is fp32 (autocast bf16 for compute). Dropout must stay 0:
  gradient caching runs each forward twice and needs them identical.
- Distributed gradient caching over the GLOBAL batch: every rank embeds its shard of
  the global batch without grad, all-gathers the embeddings, computes the identical
  global loss (all anchors x all positives and negatives, direct-link mask), keeps the
  gradient rows of its own shard multiplied by world_size (DDP averages gradients),
  then re-embeds its shard in mini-chunks with grad, syncing only on the last chunk.
  The loss and gradients therefore equal a single-GPU run on the same global batch
  (checked by tests/check_dist_equivalence.sh).
- Validation (Eval A on val) runs on rank 0; the others wait. Checkpoints hold the
  adapter weights, optimizer and scheduler, and resume with --resume.
"""
import argparse
import datetime
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/common"))
import text  # noqa: E402
from eval import Data, metrics_a, summarise_a  # noqa: E402

SCALE = 20.0
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


# ----------------------------------------------------------------- distributed

def setup_dist():
    if "RANK" in os.environ:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend, timeout=datetime.timedelta(minutes=60))
        rank, world, local = dist.get_rank(), dist.get_world_size(), int(os.environ.get("LOCAL_RANK", 0))
    else:
        rank, world, local = 0, 1, 0
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
    else:
        device = torch.device("cpu")
    return rank, world, device


def shard_sizes(batch, world):
    """Rows of the global batch held by each rank; uneven when batch % world != 0 (e.g. 512 on 3 GPUs)."""
    return [batch // world + (r < batch % world) for r in range(world)]


def gather(x, world, sizes=None):
    """all_gather along dim 0. With uneven shard sizes, pad to the largest, gather, then strip."""
    if world == 1:
        return x
    sizes = sizes or [len(x)] * world
    m = max(sizes)
    xp = x.contiguous()
    if len(x) < m:
        xp = torch.cat([xp, xp.new_zeros((m - len(x),) + tuple(x.shape[1:]))])
    out = [torch.empty_like(xp) for _ in range(world)]
    dist.all_gather(out, xp)
    return torch.cat([o[:n] for o, n in zip(out, sizes)])


def barrier(world):
    if world > 1:
        dist.barrier()


def broadcast_flag(flag, world, device):
    if world == 1:
        return flag
    t = torch.tensor([int(flag)], device=device)
    dist.broadcast(t, 0)
    return bool(t.item())


# ------------------------------------------------- data helpers (as in 0.6B trainer)

class RandomBatchSampler:
    """Seeded random global batches of exactly batch_size; the leftover tail differs each epoch."""

    def __init__(self, n, batch_size, seed):
        self.n, self.B, self.seed = n, batch_size, seed
        self.n_batches = n // batch_size

    def epoch(self, e):
        perm = np.random.default_rng([self.seed, e]).permutation(self.n)
        return [perm[i * self.B:(i + 1) * self.B] for i in range(self.n_batches)]


def link_matrix(links, foa_index, device):
    n = len(foa_index)
    A = torch.zeros(n, n, dtype=torch.bool, device=device)
    l = links[links.foa_a.isin(foa_index) & links.foa_b.isin(foa_index)]
    i = torch.as_tensor(l.foa_a.map(foa_index).to_numpy(dtype=np.int64), device=device)
    j = torch.as_tensor(l.foa_b.map(foa_index).to_numpy(dtype=np.int64), device=device)
    A[i, j] = True
    A[j, i] = True
    A.fill_diagonal_(True)
    return A


def hash_ids(hashes):
    return np.array([int(h[:15], 16) for h in hashes], dtype=np.int64)


def chunks(n, size):
    return [slice(i, min(i + size, n)) for i in range(0, n, size)]


# ----------------------------------------------------------------------- model

def load_model(path, device, r, grad_ckpt):
    from peft import LoraConfig, get_peft_model
    from sentence_transformers import SentenceTransformer
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    m = SentenceTransformer(path, device=str(device),
                            model_kwargs={"attn_implementation": "sdpa", "dtype": dtype})
    base = m[0].model
    for p in base.parameters():
        p.requires_grad_(False)
    cfg = LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.0, target_modules=LORA_TARGETS, bias="none")
    # Assign .model: in sentence-transformers 6, .auto_model is a read-only alias of it, and
    # assigning auto_model would register a second module instead of replacing this one.
    m[0].model = get_peft_model(base, cfg)
    for n, p in m.named_parameters():
        if p.requires_grad:
            p.data = p.data.float()
    if grad_ckpt:
        m[0].model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        m[0].model.enable_input_require_grads()
    return m


def trainable_state(model):
    return {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}


def tokenize(model, texts, cap, device):
    f = model.tokenizer(list(texts), padding=True, truncation=True, max_length=cap, return_tensors="pt")
    return {k: v.to(device, non_blocking=True) for k, v in f.items()}


def embed(module, feats, device):
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        out = module(feats)["sentence_embedding"]
    return F.normalize(out.float(), dim=-1)


def dist_gradcache_step(st, ddp, groups, mask_fn, mini, rank, world, device, sizes=None):
    """groups: list of (local_texts, cap) for [anchors, positives, (negatives)], each a
    rank's shard of the global batch (sizes[r] rows on rank r; equal if sizes is None).
    Returns (global loss, local tokens)."""
    feats, local_embs, orders, tokens = [], [], [], 0
    with torch.no_grad():
        for texts, cap in groups:
            order = np.argsort([len(t) for t in texts], kind="stable")
            fs = [tokenize(st, texts[order][c], cap, device) for c in chunks(len(texts), mini)]
            tokens += sum(int(f["attention_mask"].sum()) for f in fs)
            e = torch.cat([embed(st, f, device) for f in fs])
            inv = torch.as_tensor(np.argsort(order), device=device)
            feats.append(fs)
            orders.append(torch.as_tensor(order, device=device))
            local_embs.append(e[inv])
    b = len(groups[0][0])
    sizes = sizes or [b] * world
    assert sizes[rank] == b
    reps = [gather(e, world, sizes).detach().requires_grad_() for e in local_embs]
    q, docs = reps[0], torch.cat(reps[1:])
    logits = (SCALE * q @ docs.T).masked_fill(mask_fn(), float("-inf"))
    loss = F.cross_entropy(logits, torch.arange(len(q), device=device))
    loss.backward()
    start = sum(sizes[:rank])
    own = slice(start, start + b)
    work = []
    for fs, r, order in zip(feats, reps, orders):
        g = r.grad[own][order] * world      # this rank's rows, sorted order to match chunks
        work += [(f, g[c]) for c, f in zip(chunks(len(g), mini), fs)]
    for k, (f, gc) in enumerate(work):
        last = k == len(work) - 1
        ctx = ddp.no_sync() if (world > 1 and not last) else _null()
        with ctx:
            embed(ddp, f, device).backward(gc)
    return loss.item(), tokens


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


# ------------------------------------------------------------------------ eval

@torch.no_grad()
def val_eval(st, data, variant, device, batch=16):
    st.eval()
    foas, q_text, doc_ids, d_text, R, info = data.eval_a("val", variant)
    enc = lambda texts, prompt: st.encode(list(texts), prompt=prompt, batch_size=batch, normalize_embeddings=True,
                                          convert_to_numpy=True, show_progress_bar=False)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        st.max_seq_length = text.QUERY_CAP
        Q = enc(q_text, text.QUERY_PROMPT)
        st.max_seq_length = text.DOC_CAP
        D = enc(d_text, None)
    st.train()
    return summarise_a(metrics_a(Q.astype(np.float32) @ D.astype(np.float32).T, R), foas, data.meta)["all"]


# ------------------------------------------------------------------------ main

def file_hash(p):
    return hashlib.sha1(Path(p).read_bytes()).hexdigest()[:16]


def git_hash():
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=str(ROOT / "models/Qwen3-Embedding-4B"))
    ap.add_argument("--variant", default="B")
    ap.add_argument("--batch", type=int, default=512, help="GLOBAL batch (all GPUs together)")
    ap.add_argument("--mini", type=int, default=8, help="per-GPU mini-batch for the grad pass")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=float, default=0.05)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=None, help="stop early (smoke/throughput tests)")
    ap.add_argument("--limit-pairs", type=int, default=None, help="use only the first N pairs (smoke tests)")
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--ckpt-every", type=int, default=25)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--no-negatives", action="store_true")
    ap.add_argument("--no-grad-ckpt", action="store_true", help="faster if memory allows (80 GB cards)")
    ap.add_argument("--seed", type=int, default=20261102)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--query-cap", type=int, default=text.QUERY_CAP, help="TESTS ONLY: change the frozen cap")
    ap.add_argument("--doc-cap", type=int, default=text.DOC_CAP, help="TESTS ONLY: change the frozen cap")
    a = ap.parse_args()
    rank, world, device = setup_dist()
    out = Path(a.out)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    log = (lambda *x: print(*x, flush=True)) if rank == 0 else (lambda *x: None)

    data = Data(a.data)
    s = data.splits
    pairs = s[(s.split == "train") & s.in_train_cap].sort_values("application_id").reset_index(drop=True)
    neg_path = Path(a.data) / f"hard_negatives_{a.variant}.parquet"
    use_neg = not a.no_negatives
    n_no_neg = 0
    if use_neg:
        neg = pd.read_parquet(neg_path).set_index("application_id").neg_application_id
        has = pairs.application_id.isin(neg.index)
        n_no_neg = int((~has).sum())
        pairs = pairs[has].reset_index(drop=True)
    if a.limit_pairs:
        pairs = pairs.iloc[:a.limit_pairs].reset_index(drop=True)
    fam_id = {f: i for i, f in enumerate(sorted(s.family_id.unique()))}
    fam_of_doc = dict(zip(s.application_id, s.family_id.map(fam_id)))
    foa_index = pd.Series(np.arange(s.foa_number.nunique()), index=sorted(s.foa_number.unique()))
    foa_of_doc = dict(zip(s.application_id, s.foa_number.map(foa_index)))
    hash_of_doc = dict(zip(s.application_id, hash_ids(s.abstract_hash)))
    links = pd.read_parquet(Path(a.data) / "foa_links.parquet")
    A = link_matrix(links, foa_index, device)

    q_all = data.queries[f"query_{a.variant}"]
    P_q = (text.QUERY_PROMPT + q_all.loc[pairs.foa_number]).values
    P_d = data.docs.loc[pairs.application_id, "doc"].values
    P_fam = pairs.family_id.map(fam_id).values
    P_foa = pairs.foa_number.map(foa_index).to_numpy(dtype=np.int64)
    P_hash = hash_ids(pairs.abstract_hash)
    if use_neg:
        neg_ids = neg.loc[pairs.application_id].values
        P_n = data.docs.loc[neg_ids, "doc"].values
        N_fam = np.array([fam_of_doc[x] for x in neg_ids])
        N_foa = np.array([foa_of_doc[x] for x in neg_ids], dtype=np.int64)
        N_hash = np.array([hash_of_doc[x] for x in neg_ids])
        assert (N_fam != P_fam).all(), "a hard negative shares its anchor's family"
        assert not A[torch.as_tensor(P_foa, device=device), torch.as_tensor(N_foa, device=device)].any()

    sampler = RandomBatchSampler(len(pairs), a.batch, a.seed)
    steps_per_epoch = sampler.n_batches
    total = steps_per_epoch * a.epochs
    if a.max_steps:
        total = min(total, a.max_steps)
    st = load_model(a.model, device, a.lora_r, not a.no_grad_ckpt)
    st.train()
    if world > 1:
        ddp = torch.nn.parallel.DistributedDataParallel(
            st, device_ids=[device.index] if device.type == "cuda" else None)
    else:
        ddp = st
    params = [p for p in st.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.wd, fused=device.type == "cuda")
    warm = max(1, int(a.warmup * steps_per_epoch * a.epochs))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda t: min(1.0, (t + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, t / max(1, steps_per_epoch * a.epochs)))))

    n_train = sum(p.numel() for p in params)
    cfg = {**vars(a), "world_size": world, "git": git_hash(), "pairs": len(pairs),
           "families": int(pairs.family_id.nunique()), "foa_links": len(links),
           "pairs_without_negative_dropped": n_no_neg, "trainable_params": n_train,
           "steps_per_epoch": steps_per_epoch, "total_steps": total, "warmup_steps": warm, "scale": SCALE,
           "hard_negatives": use_neg,
           "data_hashes": {f: file_hash(Path(a.data) / f) for f in
                           ["splits.parquet", "queries.parquet", "docs.parquet", "foa_links.parquet"] +
                           ([neg_path.name] if use_neg else [])},
           "torch": torch.__version__,
           "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"}
    if rank == 0:
        (out / "config.json").write_text(json.dumps(cfg, indent=2))
    log(json.dumps({k: cfg[k] for k in ["world_size", "pairs", "trainable_params", "steps_per_epoch",
                                        "total_steps", "git"]}))

    step, best, bad = 0, -1.0, 0
    ck = out / "ckpt_last.pt"
    if a.resume and ck.exists():
        stt = torch.load(ck, map_location="cpu", weights_only=False)
        missing = st.load_state_dict(stt["adapter"], strict=False)
        assert not missing.unexpected_keys, missing.unexpected_keys
        opt.load_state_dict(stt["opt"])
        sched.load_state_dict(stt["sched"])
        step, best, bad = stt["step"], stt["best"], stt["bad"]
        log(f"resumed at step {step}")
    logf = open(out / "train_log.jsonl", "a") if rank == 0 else None
    sizes = shard_sizes(a.batch, world)   # uneven if batch % world != 0; the loss is still global
    own = slice(sum(sizes[:rank]), sum(sizes[:rank + 1]))
    t_start = time.time()
    done = False
    for epoch in range(a.epochs):
        batches = sampler.epoch(epoch)
        for bi, gb in enumerate(batches):
            gstep = epoch * steps_per_epoch + bi
            if gstep < step:
                continue
            t0 = time.time()
            lb = gb[own]
            groups = [(P_q[lb], a.query_cap), (P_d[lb], a.doc_cap)]
            doc_foa, doc_hash = [P_foa[gb]], [P_hash[gb]]
            if use_neg:
                groups.append((P_n[lb], a.doc_cap))
                doc_foa.append(N_foa[gb])
                doc_hash.append(N_hash[gb])

            def mask_fn():   # built from GLOBAL batch indices, identical on every rank
                dfoa = torch.as_tensor(np.concatenate(doc_foa), device=device)
                dhash = torch.as_tensor(np.concatenate(doc_hash), device=device)
                qfoa = torch.as_tensor(P_foa[gb], device=device)
                qhash = torch.as_tensor(P_hash[gb], device=device)
                m = A[qfoa][:, dfoa] | (qhash[:, None] == dhash[None, :])
                m[torch.arange(len(gb)), torch.arange(len(gb))] = False
                mask_fn.frac = float(m.float().mean())
                return m

            loss, tokens = dist_gradcache_step(st, ddp, groups, mask_fn, a.mini, rank, world, device, sizes)
            gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step = gstep + 1
            dt = time.time() - t0
            if world > 1:   # shards differ in size and length, so sum the real counts
                t = torch.tensor([tokens], dtype=torch.long, device=device)
                dist.all_reduce(t)
                tok_all = int(t.item())
            else:
                tok_all = tokens
            rec = {"step": step, "epoch": epoch, "loss": round(loss, 4), "lr": sched.get_last_lr()[0],
                   "grad_norm": round(gn, 4), "sec": round(dt, 2), "tokens_global": tok_all,
                   "tok_per_s_global": round(tok_all / dt), "tok_per_s_per_gpu": round(tokens / dt),
                   "masked_frac": round(mask_fn.frac, 5),
                   "mem_gb": round(torch.cuda.max_memory_allocated(device) / 2**30, 2) if device.type == "cuda" else 0}
            if rank == 0:
                logf.write(json.dumps(rec) + "\n")
                logf.flush()
            if step % 10 == 0 or step <= 3:
                log(f"{json.dumps(rec)} eta_h={dt * (total - step) / 3600:.2f}")
            stop = False
            # Final-step validation only for real runs (tests pass a huge --eval-every to skip it).
            if step % a.eval_every == 0 or (step == total and a.eval_every <= total):
                if rank == 0:
                    m = val_eval(st, data, a.variant, device)
                    logf.write(json.dumps({"step": step, "val": m}) + "\n")
                    log(f"step {step} val: {json.dumps(m)}")
                    if m["ndcg@10"] > best:
                        best, bad = m["ndcg@10"], 0
                        st[0].model.save_pretrained(str(out / "best_adapter"))
                        (out / "best_adapter" / "best.json").write_text(json.dumps({"step": step, "val": m}))
                    else:
                        bad += 1
                    stop = bad >= a.patience
                    if stop:
                        log("early stop")
                barrier(world)
                stop = broadcast_flag(stop, world, device)
            if rank == 0 and (step % a.ckpt_every == 0 or stop or step == total):
                torch.save({"adapter": trainable_state(st), "opt": opt.state_dict(), "sched": sched.state_dict(),
                            "step": step, "best": best, "bad": bad}, ck)
            barrier(world)
            if stop or step >= total:
                done = True
                break
        if done:
            break
    log(f"finished: step {step}, best val ndcg@10 {best:.4f}, {(time.time() - t_start) / 60:.1f} min")
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
