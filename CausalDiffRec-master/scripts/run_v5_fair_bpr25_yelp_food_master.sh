#!/usr/bin/env bash
set -euo pipefail

# End-to-end corrected-v5 fair comparisons for the two priority datasets.
# Each dataset finishes its validation-only training and audit before its OOD
# labels are loaded. Datasets run serially to keep GPU memory headroom stable.
GPU="${1:-0}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATASETS=(yelp2018 food)

cd "$ROOT"
mkdir -p logs
for dataset in "${DATASETS[@]}"; do
  echo "[$(date '+%F %T')] start corrected-v5 fair BPR25 dataset=$dataset"
  bash scripts/run_v5_fair_bpr25_training.sh "$dataset" "$GPU"
  bash scripts/run_v5_fair_bpr25_ood.sh "$dataset" "$GPU"
  echo "[$(date '+%F %T')] complete corrected-v5 fair BPR25 dataset=$dataset"
done
echo "V5_FAIR_BPR25_YELP_FOOD_COMPLETE"
