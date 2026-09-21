#!/usr/bin/env bash
set -euo pipefail

# Five-seed, validation-only mechanism test.  Both arms receive exactly 200
# complete BPR passes; no command in this script loads OOD test ground truth.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/p520/anaconda3/envs/zpp1/bin/python}"
GPU="${1:-0}"
DATASET=movielens1m
DATA_ROOT="$ROOT/data_strict/processed/$DATASET/v1_strict"
SEEDS=(1024 2048 3072 4096 5120)

cd "$ROOT"
mkdir -p logs experiments/records experiments/reports checkpoints
[[ -f "$DATA_ROOT/READY" ]] || { echo "missing $DATA_ROOT/READY" >&2; exit 2; }

for seed in "${SEEDS[@]}"; do
  record="experiments/records/${DATASET}_pure_lightgcn_v4_bpr200_seed${seed}.json"
  checkpoint="checkpoints/${DATASET}_pure_lightgcn_v4_bpr200_seed${seed}.pt"
  [[ -f "$record" ]] || env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -u \
    scripts/train_lightgcn_strict.py --dataset "$DATASET" --data_root "$DATA_ROOT" --seed "$seed" \
    --epochs 200 --protocol_version corrected_v4_pure_lightgcn_bpr200_validation_only \
    --out "$record" --checkpoint "$checkpoint" > "logs/${DATASET}_pure_lightgcn_v4_bpr200_seed${seed}.log" 2>&1
done

for seed in "${SEEDS[@]}"; do
  record="experiments/records/${DATASET}_lsci_strict_ood_lsci_none_v4_bpr200_seed${seed}_strict_ood_seed${seed}_v4.json"
  [[ -f "$record" ]] || env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -u train_lsci.py \
    --dataset "$DATASET" --data_root "$DATA_ROOT" --eval_split ood --epochs 25 --seed "$seed" --validation_only \
    --edge_gate_mode none --lambda_rank_upstream 0 --rec_refresh 0.25 --rec_lr 0.001 --rec_batch_size 1024 \
    --rec_epochs_per_outer 8 --run_id "lsci_none_v4_bpr200_seed${seed}" \
    > "logs/${DATASET}_lsci_none_v4_bpr200_seed${seed}.log" 2>&1
done

for arm in pure_lightgcn lsci_none; do
  if [[ "$arm" == pure_lightgcn ]]; then
    pattern="experiments/records/${DATASET}_pure_lightgcn_v4_bpr200_seed*.json"
  else
    pattern="experiments/records/${DATASET}_lsci_strict_ood_lsci_none_v4_bpr200_seed*_strict_ood_seed*_v4.json"
  fi
  "$PYTHON_BIN" scripts/validate_late_fusion.py --dataset "$DATASET" --data_root "$DATA_ROOT" \
    --source_records $pattern --out "experiments/reports/${DATASET}_${arm}_v4_bpr200_latefusion_validation.json" \
    --protocol_version "corrected_v4_${arm}_bpr200_latefusion_validation_only" \
    > "logs/${DATASET}_${arm}_v4_bpr200_latefusion_validation.log" 2>&1
done

for arm in pure_lightgcn lsci_none; do
  report="experiments/reports/${DATASET}_${arm}_v4_bpr200_latefusion_validation.json"
  alpha=$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["summary"]["selected_alpha"])' "$report")
  if [[ "$arm" == pure_lightgcn ]]; then
    pattern="experiments/records/${DATASET}_pure_lightgcn_v4_bpr200_seed*.json"
  else
    pattern="experiments/records/${DATASET}_lsci_strict_ood_lsci_none_v4_bpr200_seed*_strict_ood_seed*_v4.json"
  fi
  for prior in none real shuffle; do
    alpha_for_prior="$alpha"; [[ "$prior" == none ]] && alpha_for_prior=0
    "$PYTHON_BIN" scripts/evaluate_latefusion_groups.py --dataset "$DATASET" --data_root "$DATA_ROOT" \
      --source_records $pattern --alpha "$alpha_for_prior" --semantic_prior "$prior" \
      --out "experiments/reports/${DATASET}_${arm}_v4_bpr200_groups_${prior}.json" \
      --protocol_version "corrected_v4_${arm}_bpr200_group_validation_only" \
      > "logs/${DATASET}_${arm}_v4_bpr200_groups_${prior}.log" 2>&1
  done
done

echo "MOVIELENS_V4_BUDGETED_MECHANISM_VALIDATION_COMPLETE"
