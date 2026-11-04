#!/usr/bin/env bash
# Rebuild everything from the downloaded data into a staging directory (not frozen).
# Usage: experiments/common/build_all.sh <out_dir> [family_threshold]
# Freezing = copying the staged splits/families/texts into experiments/common/ after review.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
mkdir -p "$1"; OUT="$(cd "$1" && pwd)"; T="${2:-0.45}"
DATA_PY="$ROOT/grants/.venv/bin/python -I"
PY="$ROOT/.venv-train/bin/python -I"
TOK="$ROOT/models/Qwen3-Embedding-0.6B/tokenizer.json"
C="$ROOT/experiments/common"
mkdir -p "$OUT"
cd "$ROOT/grants"
$DATA_PY scripts/parse_foas.py .
$DATA_PY scripts/build_pairs.py . | tee "$OUT/pairs_stats.json"
cd "$ROOT"
$PY $C/build_text.py grants/derived/pairs.parquet "$TOK" "$OUT"
$PY $C/foa_families.py grants/derived/pairs.parquet --scan "$T" > "$OUT/family_scan.txt"
$PY $C/foa_families.py grants/derived/pairs.parquet "$OUT/foa_families.parquet" "$T"
$PY $C/make_splits.py grants/derived/pairs.parquet "$OUT/foa_families.parquet" "$OUT" | tee "$OUT/make_splits.out"
# Eval B corpus (held-out institutions) is deferred: no institution is held out.
for SPLIT in val test test_reissue; do
  $PY $C/eval.py --data "$OUT" --split $SPLIT --variant B --sanity
done
