#!/usr/bin/env bash
# DeepGraph Douban method-validation: CausalDiffRec E0 on v1_strict (popularity OOD).
# Usage: bash scripts/run_douban_deepgraph_e0_multiseed.sh [gpu_id]
set -euo pipefail
GPU="${1:-0}"
SEEDS=(1024 2048 3072 4096 5120)
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
DATA_ROOT="data_strict/processed/douban/v1_strict"
mkdir -p logs experiments/records checkpoints
MASTER="logs/douban_deepgraph_e0_multiseed_master.log"

{
  echo "[$(date '+%F %T')] Douban DeepGraph E0 multi-seed start gpu=$GPU"
  echo "data_root=$DATA_ROOT (NOT CausalDiffRec Table-1; method validation only)"
  for SEED in "${SEEDS[@]}"; do
    LOG="logs/douban_deepgraph_e0_seed${SEED}.log"
    echo "[$(date '+%F %T')] >>> douban E0 seed=$SEED"
    CUDA_VISIBLE_DEVICES="$GPU" python -u train.py \
      -d douban -e 25 --seed "$SEED" \
      --data_root "$DATA_ROOT" \
      --eval_split ood \
      --run_id "_e0_deepgraph" \
      2>&1 | tee "$LOG"
    echo "[$(date '+%F %T')] <<< done douban E0 seed=$SEED"
  done
  echo "[$(date '+%F %T')] All Douban DeepGraph E0 seeds finished"
} 2>&1 | tee -a "$MASTER"
