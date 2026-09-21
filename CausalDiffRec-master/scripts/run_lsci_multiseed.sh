#!/usr/bin/env bash
# Usage: bash scripts/run_lsci_multiseed.sh food [gpu_id]
set -euo pipefail
DATASET="${1:-food}"
GPU="${2:-0}"
SEEDS=(1024 2048 3072 4096 5120)
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs experiments/records checkpoints

echo "[$(date '+%F %T')] LSCI Stage I multi-seed start: dataset=$DATASET gpu=$GPU"
for SEED in "${SEEDS[@]}"; do
  LOG="logs/${DATASET}_lsci_seed${SEED}.log"
  echo "[$(date '+%F %T')] >>> $DATASET seed=$SEED"
  CUDA_VISIBLE_DEVICES="$GPU" python -u train_lsci.py \
    -d "$DATASET" -e 25 --seed "$SEED" \
    2>&1 | tee "$LOG"
  echo "[$(date '+%F %T')] <<< done $DATASET seed=$SEED"
done
echo "[$(date '+%F %T')] All seeds finished for $DATASET"
