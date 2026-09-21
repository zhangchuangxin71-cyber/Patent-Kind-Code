#!/usr/bin/env bash
set -euo pipefail

# Strict OOD E3 semantic-prior runner.
# Usage: bash scripts/run_strict_e3_semantic_multiseed.sh DATASET [GPU] [EPOCHS]
DATASET="${1:?dataset required}"
GPU="${2:-0}"
EPOCHS="${3:-25}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEEDS=(1024 2048 3072 4096 5120)
DATA_ROOT="$ROOT/data_strict/processed/$DATASET/v1_strict"
PRIOR="$DATA_ROOT/features/semantic_prior.pt"

cd "$ROOT"
mkdir -p logs experiments/records checkpoints
for required in "$DATA_ROOT/READY" "$PRIOR"; do
  [[ -f "$required" ]] || { echo "missing $required" >&2; exit 2; }
done

for seed in "${SEEDS[@]}"; do
  run_id="strict_e3_semantic_seed${seed}"
  log="logs/${DATASET}_${run_id}.log"
  echo "[$(date '+%F %T')] dataset=$DATASET seed=$seed"
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -u train_lsci.py \
    --dataset "$DATASET" --data_root "$DATA_ROOT" --eval_split ood \
    --epochs "$EPOCHS" --seed "$seed" \
    --edge_gate_mode soft_pair_environment \
    --lambda_rank_upstream 0 \
    --use_semantic_prior --semantic_prior_path "$PRIOR" \
    --lambda_sem 0.1 --run_id "$run_id" \
    2>&1 | tee "$log"
done
echo "[$(date '+%F %T')] completed E3 dataset=$DATASET"
