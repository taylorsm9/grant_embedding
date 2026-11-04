# Reproducing the results

Code to reproduce the results in the [write-up](README.md). It fine-tunes Qwen3-Embedding-4B to match NIH funding calls with the abstracts of grants awarded under them, then backtests an email that suggests calls to researchers.

All data comes from public NIH sources, NIH ExPORTER and the NIH Guide. The abstracts belong to their authors, so this repository holds no abstract text. It holds the code that downloads and builds the dataset, plus a manifest of IDs and hashes (`release/manifest/`) that the rebuild is checked against.

## Layout

| Path | Contents |
|---|---|
| `grants/scripts/` | Download ExPORTER files and NIH Guide pages; build projects, calls and call–award pairs |
| `experiments/common/` | Text cleaning, call families, time-based splits, near-duplicate scores, hard negatives, evaluation harness |
| `experiments/retriever_4b/` | LoRA training, adapter merge, test evaluation, embedding exports |
| `experiments/pi_backtest/` | The email backtest |
| `experiments/clustering/` | Clustering by call, and new co-PI pairs |
| `release/` | Dataset rebuild script, manifest and verifier |
| `experiments/*/results/` | The results reported in the write-up (numbers only) |

## Requirements

- Linux with a CUDA GPU. Training used one H100 80 GB. Everything else ran on one RTX 4090 (24 GB).
- Two Python environments:
  - `grants/.venv` (Python 3.14) for downloading and parsing: `pip install -r experiments/env/data_requirements.txt`
  - `.venv-train` (Python 3.12) for everything else: torch 2.14.1 from the CUDA index, then `pip install -r experiments/env/train_requirements_h100.txt`
- About 40 GB of disk, mostly for models and embeddings.

## Reproduce

Run everything from the repository root.

**1. Rebuild the dataset** (about 1.5 hours, mostly polite downloading at one request per second):

```bash
GRANT_RETRIEVAL_CONTACT=you@example.org release/reproduce.sh experiments/common/frozen
```

Every request sends your contact address in its User-Agent, and the scripts stop if a server answers 403 or 429. The script downloads Qwen3-Embedding-0.6B, whose tokenizer sets the text caps and which mines the hard negatives. At the end it compares the rebuild with the manifest. If NIH has revised a record since 2026-11-02, use the `pinned/` copy it writes, which takes the split labels and other GPU-dependent outputs from the manifest.

**2. Download the base model:** `Qwen/Qwen3-Embedding-4B`, revision `5cf2132abc99cad020ac570b19d031efec650f2b`, into `models/Qwen3-Embedding-4B`.

**3. Zero-shot validation** (optional; gives the off-the-shelf val row):

```bash
.venv-train/bin/python -I experiments/common/eval.py --data experiments/common/frozen --split val --variant B \
    --rankers bm25,dense,hybrid --model models/Qwen3-Embedding-4B --tag qwen3-4b-zs --out experiments/retriever_4b/results/zs
```

**4. Train** (about 6 hours on one H100 80 GB; with less memory, drop `--no-grad-ckpt --mini 4`):

```bash
.venv-train/bin/python -I -m torch.distributed.run --standalone --nproc_per_node=1 experiments/retriever_4b/train.py \
    --data experiments/common/frozen --out experiments/retriever_4b/run_B --no-grad-ckpt --mini 4
```

The run's configuration and per-step log are in `experiments/retriever_4b/run_B/`. The best adapter was the last step (val nDCG@10 0.681).

**5. Merge, test, embed** (about 3 hours on an RTX 4090):

```bash
bash experiments/retriever_4b/after_training.sh
```

Each test command runs once; rerunning the script skips any whose results file exists.

**6. Analyses** (CPU only, from the cached embeddings):

```bash
CUDA_VISIBLE_DEVICES= .venv-train/bin/python -I experiments/pi_backtest/backtest.py
CUDA_VISIBLE_DEVICES= .venv-train/bin/python -I experiments/clustering/cluster_eval.py
CUDA_VISIBLE_DEVICES= .venv-train/bin/python -I experiments/clustering/unusual_teams.py
rsvg-convert -z 2 experiments/clustering/results/tsne_new_calls_4b.svg -o figures/tsne_v2_4b_top10_calls.png
```

The backtest and the collaboration analysis also write example emails and teams, with project titles and PI IDs, to `experiments/pi_backtest/emails/` and `experiments/clustering/examples/`. Both folders are git-ignored.

## Where each number comes from

| Write-up section | File |
|---|---|
| Matching model table | `experiments/retriever_4b/results/{zs,ft}/{test,test_reissue}_B.json` |
| Near-duplicate check | the same files with `_nodup0.95` |
| From model to email | `experiments/pi_backtest/results/backtest_4b.json` (`email_framing`) |
| Clustering table and NMI | `experiments/clustering/results/cluster_eval_4b.json` |
| New co-PI pairs with little topical overlap | `experiments/clustering/results/unusual_teams_4b.json` (`pairs_by_kind`) |

## License

The code is released under the Apache License 2.0 (`LICENSE`), the same license as the Qwen3 models it fine-tunes. NIH data and the abstracts it contains are not covered by this license; they remain with NIH and the grants' authors.
