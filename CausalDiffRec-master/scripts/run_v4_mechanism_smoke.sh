#!/usr/bin/env bash
set -euo pipefail

# Validation-only single-seed gate for corrected-v4.  This script deliberately
# does not read OOD test ground truth and does not expand to five seeds.
DATASET="${1:?dataset required}"
GPU="${2:-0}"
EPOCHS="${3:-5}"
SEM_ALPHA="${4:-0.0}"
SEED="${SEED:-1024}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/p520/anaconda3/envs/zpp1/bin/python}"
DATA_ROOT="$ROOT/data_strict/processed/$DATASET/v1_strict"
PRIOR="$DATA_ROOT/features/semantic_prior.pt"
SHUFFLED_PRIOR="$DATA_ROOT/features/semantic_prior_shuffled.pt"

cd "$ROOT"
mkdir -p logs experiments/records checkpoints
[[ -f "$DATA_ROOT/READY" ]] || { echo "missing $DATA_ROOT/READY" >&2; exit 2; }

run_arm() {
  local arm="$1"
  shift
  local log="logs/${DATASET}_v4_smoke_${arm}_seed${SEED}.log"
  env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -u "$@" \
    --dataset "$DATASET" --data_root "$DATA_ROOT" --eval_split ood \
    --epochs "$EPOCHS" --seed "$SEED" --validation_only \
    --rec_refresh 0.25 --rec_lr 0.001 --rec_batch_size 1024 \
    --run_id "v4_smoke_${arm}" > "$log" 2>&1
}

run_arm e0 train.py
run_arm none train_lsci.py --edge_gate_mode none --lambda_rank_upstream 0
run_arm random train_lsci.py --edge_gate_mode random_pair_environment --lambda_rank_upstream 0
run_arm learned train_lsci.py --edge_gate_mode soft_pair_environment --lambda_rank_upstream 0

if [[ -f "$PRIOR" ]]; then
  run_arm semantic_real train_lsci.py \
    --edge_gate_mode soft_pair_environment --lambda_rank_upstream 0 \
    --use_semantic_prior --semantic_prior_path "$PRIOR" --lambda_sem 0.1 \
    --semantic_score_alpha "$SEM_ALPHA"
fi
if [[ -f "$SHUFFLED_PRIOR" ]]; then
  run_arm semantic_shuffled train_lsci.py \
    --edge_gate_mode soft_pair_environment --lambda_rank_upstream 0 \
    --use_semantic_prior --semantic_prior_path "$SHUFFLED_PRIOR" --lambda_sem 0.1 \
    --semantic_score_alpha "$SEM_ALPHA"
fi

echo "corrected-v4 validation smoke complete: $DATASET seed=$SEED"
