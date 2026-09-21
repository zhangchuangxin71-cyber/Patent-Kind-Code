#!/usr/bin/env bash
set -euo pipefail

# Pre-registered corrected-v3 semantic controls for Yelp strict OOD.
#
# The only intentional difference from formal E3 is one factor per arm:
#   1. e4_lamsem0: semantic fusion remains enabled, InfoNCE is disabled.
#   2. e3_shuffle: full E3, but item semantic rows are permuted.
#
# Workflow: two seed-1024 five-epoch smoke runs, then the formal 5-seed runs.
# Formal runs are entered only when both smoke commands exit successfully.
# No OOD result is used to choose a new setting after this script starts.
#
# Usage: bash scripts/run_yelp_corrected_v3_semantic_controls.sh [GPU]

GPU="${1:-0}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/p520/anaconda3/envs/zpp1/bin/python}"
DATASET="yelp2018"
DATA_ROOT="$ROOT/data_strict/processed/$DATASET/v1_strict"
PRIOR="$DATA_ROOT/features/semantic_prior.pt"
SHUFFLED_PRIOR="$DATA_ROOT/features/semantic_prior_shuffled.pt"

cd "$ROOT"
mkdir -p logs experiments/records checkpoints
for required in "$DATA_ROOT/READY" "$PRIOR" "$SHUFFLED_PRIOR"; do
  [[ -f "$required" ]] || { echo "missing required file: $required" >&2; exit 2; }
done

run_arm() {
  local arm="$1"
  local seed="$2"
  local epochs="$3"
  local stage="$4"
  local run_id log

  case "$arm" in
    e4_lamsem0)
      run_id="corrected_v3_yelp_e4_lamsem0_${stage}_seed${seed}"
      log="logs/yelp2018_${run_id}.log"
      CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -u train_lsci.py \
        --dataset "$DATASET" --data_root "$DATA_ROOT" --eval_split ood \
        --epochs "$epochs" --seed "$seed" --edge_gate_mode none \
        --use_semantic_prior --semantic_prior_path "$PRIOR" --lambda_sem 0 \
        --run_id "$run_id" 2>&1 | tee "$log"
      ;;
    e3_shuffle)
      run_id="corrected_v3_yelp_e3_shuffle_${stage}_seed${seed}"
      log="logs/yelp2018_${run_id}.log"
      CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -u train_lsci.py \
        --dataset "$DATASET" --data_root "$DATA_ROOT" --eval_split ood \
        --epochs "$epochs" --seed "$seed" --edge_gate_mode none \
        --use_semantic_prior --semantic_prior_path "$SHUFFLED_PRIOR" --lambda_sem 0.1 \
        --run_id "$run_id" 2>&1 | tee "$log"
      ;;
    *)
      echo "unknown arm: $arm" >&2
      return 2
      ;;
  esac
}

echo "[$(date '+%F %T')] Yelp corrected-v3 semantic controls: smoke start"
for arm in e4_lamsem0 e3_shuffle; do
  # Checkpoint selection is intentionally unavailable before the causal-score
  # warmup; five epochs is the smallest valid smoke length for this protocol.
  run_arm "$arm" 1024 5 smoke
  echo "[$(date '+%F %T')] smoke completed: arm=$arm seed=1024"
done

echo "[$(date '+%F %T')] both smoke runs passed; formal 5-seed controls start"
for arm in e4_lamsem0 e3_shuffle; do
  for seed in 1024 2048 3072 4096 5120; do
    run_arm "$arm" "$seed" 25 formal
    echo "[$(date '+%F %T')] formal completed: arm=$arm seed=$seed"
  done
done

echo "[$(date '+%F %T')] YELP_CORRECTED_V3_SEMANTIC_CONTROLS_COMPLETE"
