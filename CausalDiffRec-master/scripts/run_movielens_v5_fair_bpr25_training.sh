#!/usr/bin/env bash
set -euo pipefail

# Validation-only preparation for the fair three-method comparison.
# Every arm receives 25 complete BPR passes.  OOD test labels are not loaded.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/p520/anaconda3/envs/zpp1/bin/python}"
GPU="${1:-0}"
MAX_E0_JOBS="${MAX_E0_JOBS:-5}"
DATASET=movielens1m
DATA_ROOT="$ROOT/data_strict/processed/$DATASET/v1_strict"
SEEDS=(1024 2048 3072 4096 5120)

cd "$ROOT"
mkdir -p logs experiments/records experiments/checkpoints checkpoints
[[ -f "$DATA_ROOT/READY" ]] || { echo "missing $DATA_ROOT/READY" >&2; exit 2; }

run_e0() {
  local seed="$1"
  local record="experiments/records/${DATASET}_strict_ood_v5_fair_bpr25_e0_seed${seed}_strict_ood_seed${seed}_v4.json"
  local log="logs/${DATASET}_v5_fair_bpr25_e0_seed${seed}.log"
  [[ -f "$record" ]] && return 0
  env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -u train.py \
    --dataset "$DATASET" --data_root "$DATA_ROOT" --eval_split ood \
    --epochs 25 --seed "$seed" --validation_only \
    --rec_refresh 0.25 --rec_lr 0.001 --rec_batch_size 1024 \
    --rec_epochs_per_outer 1 --run_id "v5_fair_bpr25_e0_seed${seed}" \
    > "$log" 2>&1
}

active=0
for seed in "${SEEDS[@]}"; do
  run_e0 "$seed" &
  active=$((active + 1))
  if (( active >= MAX_E0_JOBS )); then
    wait -n
    active=$((active - 1))
  fi
done
wait
echo "[$(date '+%F %T')] CausalDiffRec E0 validation-only arm complete"

# LightGCN is inexpensive in memory, so its five seeds can run together after
# E0 releases the dense CausalDiffRec allocations.
for seed in "${SEEDS[@]}"; do
  record="experiments/records/${DATASET}_v5_fair_bpr25_lightgcn_seed${seed}.json"
  checkpoint="experiments/checkpoints/${DATASET}_v5_fair_bpr25_lightgcn_seed${seed}.pt"
  log="logs/${DATASET}_v5_fair_bpr25_lightgcn_seed${seed}.log"
  [[ -f "$record" ]] && continue
  env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -u \
    scripts/train_lightgcn_strict.py --dataset "$DATASET" --data_root "$DATA_ROOT" \
    --seed "$seed" --epochs 25 --min_epochs 25 --patience 0 \
    --embedding_dim 8 --layers 3 --batch_size 1024 --lr 0.001 --l2 0.001 \
    --protocol_version corrected_v5_fair_bpr25_lightgcn_validation_only \
    --out "$record" --checkpoint "$checkpoint" > "$log" 2>&1 &
done
wait
echo "[$(date '+%F %T')] LightGCN validation-only arm complete"
echo "MOVIELENS_V5_FAIR_BPR25_VALIDATION_TRAINING_COMPLETE"
