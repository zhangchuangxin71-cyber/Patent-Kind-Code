#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/p520/anaconda3/envs/zpp1/bin/python}"
GPU="${1:-0}"
DATASET=amazon_beauty
DATA_ROOT="$ROOT/data_strict/processed/$DATASET/v1_strict"
VALIDATION="experiments/reports/${DATASET}_v4_strong_latefusion_validation.json"
cd "$ROOT"
mkdir -p logs experiments/reports

# The corrected-v3 pure-LightGCN checkpoints already contain 200 complete BPR
# passes and use the same optimizer/data path. Re-run only the stricter v4
# validation gate before introducing the new adaptive configuration.
"$PYTHON_BIN" scripts/validate_late_fusion.py --dataset "$DATASET" --data_root "$DATA_ROOT" \
  --source_records experiments/records/${DATASET}_lightgcn_strict_ood_seed*_v3.json \
  --alphas 0,0.025,0.05,0.075,0.1,0.15,0.2,0.25,0.5,0.75,1,1.25,1.5,2 \
  --protocol_version corrected_v4_strong_latefusion_validation_only --out "$VALIDATION" \
  > "logs/${DATASET}_v4_strong_latefusion_validation.log" 2>&1

bash scripts/run_v4_activity_adaptive_dataset.sh "$DATASET" \
  "experiments/records/${DATASET}_lightgcn_strict_ood_seed*_v3.json" "$VALIDATION" "$GPU"
echo "AMAZON_BEAUTY_V4_ADAPTIVE_COMPLETE"
