#!/usr/bin/env bash
# Everything after training, on one GPU (an RTX 4090 with 24 GB is enough):
#   1. merge the best LoRA adapter into the base model in fp32 and check rankings match;
#   2. test evaluation, each command ONCE (a command whose results file exists is skipped);
#   3. full-corpus embeddings for the analysis scripts (PI backtest, clustering, collaborations);
#   4. near-duplicate checks, val candidates and ID-keyed embedding exports.
# Run from the repo root:  bash experiments/retriever_4b/after_training.sh
# Override the encode batch size with BATCH=8 if the GPU runs out of memory.
set -euo pipefail
cd "$(dirname "$0")/../.."
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
PY="${PY:-.venv-train/bin/python} -I"
R=experiments/retriever_4b
M=models/Qwen3-Embedding-4B
FT=$R/run_B/merged
BATCH=${BATCH:-16}

[ -f $M/config.json ] || { echo "!! $M missing"; exit 1; }
[ -f $R/run_B/best_adapter/adapter_model.safetensors ] || { echo "!! run_B/best_adapter missing (train first)"; exit 1; }

echo "== merge $(date -Is) =="
if [ ! -f $FT/MERGE_OK ]; then
    rm -rf $FT
    $PY $R/merge_adapter.py --model $M --adapter $R/run_B/best_adapter --out $FT \
        --data experiments/common/frozen --fp32 2>&1 | tee $R/run_B/merge.out
    grep -q "rankings identical on sample: True" $R/run_B/merge.out || { echo "!! merge check failed; not running test"; exit 1; }
    touch $FT/MERGE_OK
fi

ev() {  # ev <results json that this command writes> <eval.py args...>
    local res=$1; shift
    if [ -f "$res" ]; then echo "-- $res exists; NOT re-running test"; return; fi
    echo "-- $(date -Is) $res"
    $PY experiments/common/eval.py --data experiments/common/frozen --variant B --rankers bm25,dense,hybrid "$@"
    [ -f "$res" ] || { echo "!! expected $res was not written"; exit 1; }
}

echo "== test evaluation $(date -Is) =="
for SPLIT in test test_reissue; do
    ev $R/results/zs/${SPLIT}_B.json --split $SPLIT --model $M --tag qwen3-4b-zs --out $R/results/zs
    ev $R/results/ft/${SPLIT}_B.json --split $SPLIT --model $FT --tag qwen3-4b-ft-B --out $R/results/ft \
        --candidates experiments/common/candidates_4b_$SPLIT.parquet
done

echo "== full-corpus embeddings $(date -Is) =="
$PY $R/encode_all.py --data experiments/common/frozen --model $FT --tag qwen3-4b-ft-B --batch $BATCH

echo "== near-duplicate checks $(date -Is) =="
for SPLIT in test test_reissue; do
    ev $R/results/zs/${SPLIT}_B_nodup0.95.json --split $SPLIT --model $M --tag qwen3-4b-zs --out $R/results/zs --near-dup-max 0.95
    ev $R/results/ft/${SPLIT}_B_nodup0.95.json --split $SPLIT --model $FT --tag qwen3-4b-ft-B --out $R/results/ft --near-dup-max 0.95
done

echo "== val candidates, exports, zero-shot full corpus $(date -Is) =="
ev $R/results/ft/val_B.json --split val --model $FT --tag qwen3-4b-ft-B --out $R/results/ft \
    --candidates experiments/common/candidates_4b_val.parquet
$PY $R/export_embeddings.py --data experiments/common/frozen --tags qwen3-4b-zs,qwen3-4b-ft-B
$PY $R/encode_all.py --data experiments/common/frozen --model $M --tag qwen3-4b-zs --batch $BATCH
echo "== $(date -Is) all done =="
