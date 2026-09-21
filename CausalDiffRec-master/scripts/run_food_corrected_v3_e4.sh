#!/usr/bin/env bash
set -euo pipefail

# Pre-registered Food strict-OOD E4 control.
# This matches final E3 except for lambda_sem=0: semantic fusion remains on,
# while the InfoNCE alignment term is disabled.  Checkpoints are selected by
# validation NDCG@20; OOD test is read once by train_lsci.py.
#
# Usage: bash scripts/run_food_corrected_v3_e4.sh [GPU]

GPU="${1:-0}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/p520/anaconda3/envs/zpp1/bin/python}"
DATASET="food"
DATA_ROOT="$ROOT/data_strict/processed/$DATASET/v1_strict"
PRIOR="$DATA_ROOT/features/semantic_prior.pt"
SEEDS=(1024 2048 3072 4096 5120)
MASTER_LOG="logs/food_corrected_v3_e4_master.log"

cd "$ROOT"
mkdir -p logs experiments/records checkpoints
for required in "$DATA_ROOT/READY" "$PRIOR"; do
  [[ -f "$required" ]] || { echo "missing required file: $required" >&2; exit 2; }
done

record_for() {
  local seed="$1"
  printf 'experiments/records/%s_lsci_strict_ood_e4_lamsem0_corrected_v3_food_e4_lamsem0_formal_seed%s_strict_ood_e4_lamsem0_seed%s_v3.json' \
    "$DATASET" "$seed" "$seed"
}

run_one() {
  local seed="$1"
  local record log run_id
  record="$(record_for "$seed")"
  log="logs/food_corrected_v3_food_e4_lamsem0_formal_seed${seed}.log"
  run_id="corrected_v3_food_e4_lamsem0_formal_seed${seed}"

  if [[ -f "$record" ]]; then
    echo "[$(date '+%F %T')] skip existing formal record: $record"
    return 0
  fi

  echo "[$(date '+%F %T')] start Food E4 seed=$seed"
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -u train_lsci.py \
    --dataset "$DATASET" --data_root "$DATA_ROOT" --eval_split ood \
    --epochs 25 --seed "$seed" --edge_gate_mode none \
    --use_semantic_prior --semantic_prior_path "$PRIOR" --lambda_sem 0 \
    --run_id "$run_id" 2>&1 | tee "$log"

  [[ -f "$record" ]] || {
    echo "[$(date '+%F %T')] error: expected record was not produced: $record" >&2
    return 1
  }
  echo "[$(date '+%F %T')] completed Food E4 seed=$seed"
}

{
  echo "[$(date '+%F %T')] FOOD_CORRECTED_V3_E4_START gpu=$GPU"
  for seed in "${SEEDS[@]}"; do
    run_one "$seed"
  done
  echo "[$(date '+%F %T')] FOOD_CORRECTED_V3_E4_COMPLETE"
} 2>&1 | tee -a "$MASTER_LOG"
