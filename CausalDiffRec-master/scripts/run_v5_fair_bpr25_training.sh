#!/usr/bin/env bash
set -euo pipefail

# Create clean validation-only checkpoints for the corrected-v5 fair comparison.
# Usage: bash scripts/run_v5_fair_bpr25_training.sh DATASET [GPU]
# The three arms receive exactly 25 complete BPR passes.  OOD labels are never
# loaded in this script; the separate OOD runner checks the audit first.
DATASET="${1:?dataset required}"
GPU="${2:-0}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/p520/anaconda3/envs/zpp1/bin/python}"
# Yelp's dense reconstruction decoder peaks near 12 GiB per causal-diffusion
# process and has an additional 1.9 GiB allocation spike. On the 24 GiB GPU
# the causal-diffusion arms must run one seed at a time.
MAX_E0_JOBS="${MAX_E0_JOBS:-1}"
MAX_FULL_JOBS="${MAX_FULL_JOBS:-1}"
DATA_ROOT="$ROOT/data_strict/processed/$DATASET/v1_strict"
PRIOR="${PRIOR:-$DATA_ROOT/features/semantic_prior.pt}"
SEEDS=(1024 2048 3072 4096 5120)

cd "$ROOT"
mkdir -p logs experiments/records experiments/checkpoints checkpoints
[[ -f "$DATA_ROOT/READY" ]] || { echo "missing $DATA_ROOT/READY" >&2; exit 2; }
[[ -f "$PRIOR" ]] || { echo "missing $PRIOR" >&2; exit 2; }

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

run_full() {
  local seed="$1"
  local record="experiments/records/${DATASET}_lsci_strict_ood_e3_v5_fair_bpr25_full_seed${seed}_strict_ood_e3_seed${seed}_v4.json"
  local log="logs/${DATASET}_v5_fair_bpr25_full_seed${seed}.log"
  [[ -f "$record" ]] && return 0
  env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -u train_lsci.py \
    --dataset "$DATASET" --data_root "$DATA_ROOT" --eval_split ood \
    --epochs 25 --seed "$seed" --validation_only \
    --edge_gate_mode none --lambda_rank_upstream 0 \
    --rec_refresh 0.25 --rec_lr 0.001 --rec_batch_size 1024 \
    --rec_epochs_per_outer 1 --use_semantic_prior --semantic_prior_path "$PRIOR" \
    --lambda_sem 0.1 --semantic_score_alpha 0.75 \
    --run_id "v5_fair_bpr25_full_seed${seed}" \
    > "$log" 2>&1
}

run_parallel() {
  local limit="$1"
  local fn="$2"
  local active=0
  local seed
  for seed in "${SEEDS[@]}"; do
    "$fn" "$seed" &
    active=$((active + 1))
    if (( active >= limit )); then
      wait -n
      active=$((active - 1))
    fi
  done
  wait
}

run_parallel "$MAX_E0_JOBS" run_e0
echo "[$(date '+%F %T')] $DATASET CausalDiffRec E0 validation-only arm complete"

run_parallel "$MAX_FULL_JOBS" run_full
echo "[$(date '+%F %T')] $DATASET full semantic method validation-only arm complete"

# LightGCN occupies little VRAM; all five seeds can run after the dense arms.
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
echo "[$(date '+%F %T')] $DATASET LightGCN validation-only arm complete"
echo "${DATASET}_V5_FAIR_BPR25_VALIDATION_TRAINING_COMPLETE"
