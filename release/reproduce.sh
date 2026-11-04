#!/usr/bin/env bash
# Rebuild the frozen grant-retrieval dataset from public sources and check it against the manifest.
#
# Usage: GRANT_RETRIEVAL_CONTACT=you@example.org release/reproduce.sh [out_dir]
#
# Needs the two environments from REPRODUCE.md (grants/.venv for downloading and parsing,
# .venv-train for tokenising, embedding and evaluation) and a CUDA GPU for the zero-shot steps.
# Downloads ~4 GB of ExPORTER zips and ~4,950 NIH Guide pages at about one request per second
# (roughly 1.5 hours). Every step resumes if re-run. Stop if a server answers 403 or 429.
set -euo pipefail
: "${GRANT_RETRIEVAL_CONTACT:?set GRANT_RETRIEVAL_CONTACT to your email (sent in the User-Agent header)}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$ROOT/experiments/common/rebuild}"
MAN="$ROOT/release/manifest"
DATA_PY="$ROOT/grants/.venv/bin/python -I"
PY="$ROOT/.venv-train/bin/python -I"
G="$ROOT/grants"
mkdir -p "$G/raw" "$G/derived" "$G/logs"

# 1. ExPORTER project and abstract files, FY2008-2025 (compare SHA-256 with $MAN/exporter_files.csv).
$DATA_PY "$G/scripts/download_exporter.py" "$G" 2008 2025
$DATA_PY "$G/scripts/build_projects.py" "$G"
$DATA_PY "$G/scripts/select_foas.py" "$G" 2008 2025

# 2. FOA text: the exact NIH Guide list used, then the Grants.gov fallback for test-period FOAs
#    with no Guide page.
$DATA_PY "$G/scripts/fetch_foas.py" "$G" "$MAN/foa_fetch_list.txt"
$DATA_PY "$G/scripts/fetch_grantsgov.py" "$G" "$MAN/grantsgov_list.txt"

# 3. Model (Apache-2.0), downloaded once.
if [ ! -f "$ROOT/models/Qwen3-Embedding-0.6B/config.json" ]; then
  $PY -c "import os; from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-Embedding-0.6B', local_dir='$ROOT/models/Qwen3-Embedding-0.6B', max_workers=1, user_agent='grant-retrieval-research/0.1 (contact: ' + os.environ['GRANT_RETRIEVAL_CONTACT'] + ')')"
fi

# 4. Pairs, cleaned text, FOA families and links, splits, harness sanity checks.
"$ROOT/experiments/common/build_all.sh" "$OUT" "$(python3 -c "import json; print(json.load(open('$MAN/build_info.json'))['family_threshold'])")"

# 5. Zero-shot steps: near-duplicate scores and hard negatives (GPU; borderline cases can differ
#    across hardware, so the manifest's copies are canonical).
for SPLIT in val test test_reissue; do
  $PY "$ROOT/experiments/common/near_dups.py" --data "$OUT" --split $SPLIT
done
$PY "$ROOT/experiments/common/mine_negatives.py" --data "$OUT" --variant B

# 6. Compare with the manifest; write a pinned copy that uses the canonical split labels.
$PY "$ROOT/release/verify.py" "$OUT" "$MAN" --pin "$OUT/pinned" || {
  echo "Rebuild differs from the manifest (see $OUT/verify_report.json). Use $OUT/pinned for exact splits."; exit 1; }
echo "Exact match. Data directory for training and evaluation: $OUT (or $OUT/pinned)."
