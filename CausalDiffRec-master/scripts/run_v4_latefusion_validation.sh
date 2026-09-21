#!/usr/bin/env bash
set -euo pipefail

# Train collaborative v4 checkpoints, then select/confirm semantic alpha only
# on validation.  This script never loads OOD test ground truth.
DATASET="${1:?dataset required}"
GPU="${2:-0}"
EPOCHS="${3:-25}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/p520/anaconda3/envs/zpp1/bin/python}"
DATA_ROOT="$ROOT/data_strict/processed/$DATASET/v1_strict"
SEEDS=(1024 2048 3072 4096 5120)

cd "$ROOT"
mkdir -p logs experiments/records experiments/reports checkpoints
[[ -f "$DATA_ROOT/READY" ]] || { echo "missing $DATA_ROOT/READY" >&2; exit 2; }

for seed in "${SEEDS[@]}"; do
  record="experiments/records/${DATASET}_lsci_strict_ood_v4_lfval_none_seed${seed}_strict_ood_seed${seed}_v4.json"
  log="logs/${DATASET}_v4_lfval_none_seed${seed}.log"
  [[ -f "$record" ]] && continue
  env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -u train_lsci.py \
    --dataset "$DATASET" --data_root "$DATA_ROOT" --eval_split ood \
    --epochs "$EPOCHS" --seed "$seed" --validation_only \
    --edge_gate_mode none --lambda_rank_upstream 0 \
    --rec_refresh 0.25 --rec_lr 0.001 --rec_batch_size 1024 \
    --run_id "v4_lfval_none_seed${seed}" > "$log" 2>&1
done

"$PYTHON_BIN" scripts/validate_late_fusion.py \
  --dataset "$DATASET" --data_root "$DATA_ROOT" \
  --source_records experiments/records/${DATASET}_lsci_strict_ood_v4_lfval_none_seed*_strict_ood_seed*_v4.json \
  --out "experiments/reports/${DATASET}_v4_latefusion_validation.json" \
  --protocol_version corrected_v4_latefusion_validation_only \
  > "logs/${DATASET}_v4_latefusion_validation_sweep.log" 2>&1
echo "corrected-v4 late-fusion validation complete: $DATASET"
