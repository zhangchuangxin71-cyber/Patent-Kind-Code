#!/usr/bin/env bash
set -euo pipefail

# Five-seed validation-only confirmation.  OOD test GT is never loaded here.
DATASET="${1:?dataset required}"
ALPHA="${2:?validation-selected alpha required}"
GPU="${3:-0}"
EPOCHS="${4:-25}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/p520/anaconda3/envs/zpp1/bin/python}"
DATA_ROOT="$ROOT/data_strict/processed/$DATASET/v1_strict"
PRIOR="$DATA_ROOT/features/semantic_prior.pt"
SHUFFLED_PRIOR="$DATA_ROOT/features/semantic_prior_shuffled.pt"
SEEDS=(1024 2048 3072 4096 5120)

cd "$ROOT"
mkdir -p logs experiments/records checkpoints
for required in "$DATA_ROOT/READY" "$PRIOR" "$SHUFFLED_PRIOR"; do
  [[ -f "$required" ]] || { echo "missing $required" >&2; exit 2; }
done

run_arm() {
  local arm="$1"
  local seed="$2"
  shift 2
  local log="logs/${DATASET}_v4_confirm_${arm}_seed${seed}.log"
  env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="$GPU" \
    "$PYTHON_BIN" -u train_lsci.py "$@" \
    --dataset "$DATASET" --data_root "$DATA_ROOT" --eval_split ood \
    --epochs "$EPOCHS" --seed "$seed" --validation_only \
    --edge_gate_mode none --lambda_rank_upstream 0 \
    --rec_refresh 0.25 --rec_lr 0.001 --rec_batch_size 1024 \
    --run_id "v4_confirm_${arm}_seed${seed}" > "$log" 2>&1
}

for seed in "${SEEDS[@]}"; do
  run_arm none "$seed"
  run_arm semantic_real "$seed" \
    --use_semantic_prior --semantic_prior_path "$PRIOR" --lambda_sem 0.1 \
    --semantic_score_alpha "$ALPHA"
  run_arm semantic_shuffled "$seed" \
    --use_semantic_prior --semantic_prior_path "$SHUFFLED_PRIOR" --lambda_sem 0.1 \
    --semantic_score_alpha "$ALPHA"
done

echo "corrected-v4 semantic confirmation complete: $DATASET"
