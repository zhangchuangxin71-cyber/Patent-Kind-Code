#!/usr/bin/env bash
set -euo pipefail

# Validation-only Stage-I refresh mechanism ablation. Existing pure and
# refresh=0.25 arms are reused; only missing arms are trained.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/p520/anaconda3/envs/zpp1/bin/python}"
GPU="${1:-0}"
DATASET=movielens1m
DATA_ROOT="$ROOT/data_strict/processed/$DATASET/v1_strict"
SEEDS=(1024 2048 3072 4096 5120)

cd "$ROOT"
mkdir -p logs experiments/records experiments/reports checkpoints
for arm in stage1_once matched_random_refresh; do
  for seed in "${SEEDS[@]}"; do
    refresh=0
    source=upstream
    if [[ "$arm" == matched_random_refresh ]]; then
      refresh=0.25
      source=matched_random
    fi
    record="experiments/records/${DATASET}_lsci_strict_ood_v4_${arm}_bpr200_seed${seed}_strict_ood_seed${seed}_v4.json"
    [[ -f "$record" ]] && continue
    env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -u train_lsci.py \
      --dataset "$DATASET" --data_root "$DATA_ROOT" --eval_split ood --epochs 25 \
      --seed "$seed" --validation_only --edge_gate_mode none --lambda_rank_upstream 0 \
      --rec_refresh "$refresh" --rec_refresh_source "$source" --rec_lr 0.001 \
      --rec_batch_size 1024 --rec_epochs_per_outer 8 \
      --run_id "v4_${arm}_bpr200_seed${seed}" \
      > "logs/${DATASET}_v4_${arm}_bpr200_seed${seed}.log" 2>&1
  done
done

"$PYTHON_BIN" scripts/summarize_refresh_budget_curve.py \
  --pure_records experiments/records/${DATASET}_pure_lightgcn_v4_bpr200_seed*.json \
  --stage1_once_records experiments/records/${DATASET}_lsci_strict_ood_v4_stage1_once_bpr200_seed*_strict_ood_seed*_v4.json \
  --stage1_refresh_records experiments/records/${DATASET}_lsci_strict_ood_lsci_none_v4_bpr200_seed*_strict_ood_seed*_v4.json \
  --random_refresh_records experiments/records/${DATASET}_lsci_strict_ood_v4_matched_random_refresh_bpr200_seed*_strict_ood_seed*_v4.json \
  --out experiments/reports/${DATASET}_v4_refresh_budget_curve.json \
  > logs/${DATASET}_v4_refresh_budget_curve.log 2>&1
echo "MOVIELENS_V4_REFRESH_ABLATION_COMPLETE"
