#!/usr/bin/env bash
set -euo pipefail

# Strict OOD ablation runner.
# Usage: bash scripts/run_strict_ablation_multiseed.sh DATASET [GPU] [EPOCHS]
DATASET="${1:?dataset required}"
GPU="${2:-0}"
EPOCHS="${3:-25}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEEDS=(1024 2048 3072 4096 5120)
MODES=(e0 none soft_pair_environment random_pair_environment)
DATA_ROOT="$ROOT/data_strict/processed/$DATASET/v1_strict"

cd "$ROOT"
mkdir -p logs experiments/records checkpoints
if [[ ! -f "$DATA_ROOT/READY" ]]; then
  echo "missing strict READY marker: $DATA_ROOT/READY" >&2
  exit 2
fi

for mode in "${MODES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    run_id="strict_${mode}_seed${seed}"
    log="logs/${DATASET}_${run_id}.log"
    echo "[$(date '+%F %T')] dataset=$DATASET mode=$mode seed=$seed"
    if [[ "$mode" == "e0" ]]; then
      CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -u train.py \
        --dataset "$DATASET" --data_root "$DATA_ROOT" --eval_split ood \
        --epochs "$EPOCHS" --seed "$seed" --run_id "$run_id" \
        2>&1 | tee "$log"
    else
      CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -u train_lsci.py \
        --dataset "$DATASET" --data_root "$DATA_ROOT" --eval_split ood \
        --epochs "$EPOCHS" --seed "$seed" --edge_gate_mode "$mode" \
        --lambda_rank_upstream 0 \
        --run_id "$run_id" 2>&1 | tee "$log"
    fi
  done
done

echo "[$(date '+%F %T')] completed dataset=$DATASET"
