#!/usr/bin/env bash
set -euo pipefail

# Frozen OOD evaluation: this script refuses to run until validation-only
# confirmation has authorized it.  No hyperparameter is selected here.
DATASET="${1:?dataset required}"
ALPHA="${2:?frozen validation-selected alpha required}"
GPU="${3:-0}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/p520/anaconda3/envs/zpp1/bin/python}"
DATA_ROOT="$ROOT/data_strict/processed/$DATASET/v1_strict"
CONFIRMATION="$ROOT/experiments/summaries/${DATASET}_v4_confirmation.json"
PRIOR="$DATA_ROOT/features/semantic_prior.pt"
SHUFFLED_PRIOR="$DATA_ROOT/features/semantic_prior_shuffled.pt"
SEEDS=(1024 2048 3072 4096 5120)

cd "$ROOT"
mkdir -p logs experiments/records experiments/reports
[[ -f "$CONFIRMATION" ]] || { echo "missing confirmation: $CONFIRMATION" >&2; exit 2; }
[[ "$("$PYTHON_BIN" -c 'import json,sys; print(str(bool(json.load(open(sys.argv[1], encoding="utf-8")).get("ood_test_allowed", False))).lower())' "$CONFIRMATION")" == "true" ]] || {
  echo "OOD gate closed" >&2; exit 3;
}

run_one() {
  local arm="$1"
  local seed="$2"
  local source="$3"
  local prior="$4"
  local alpha="$5"
  local out="experiments/records/${DATASET}_v4_ood_${arm}_seed${seed}_v4.json"
  local log="logs/${DATASET}_v4_ood_${arm}_seed${seed}.log"
  [[ -f "$out" ]] && return 0
  env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" \
    scripts/evaluate_late_fusion.py --source_record "$source" --data_root "$DATA_ROOT" \
    --semantic_prior_path "$prior" --eval_split ood --alpha "$alpha" --out "$out" \
    --protocol_version corrected_v4_frozen_ood --comparison_label "$arm" > "$log" 2>&1
}

for seed in "${SEEDS[@]}"; do
  run_one none "$seed" \
    "experiments/records/${DATASET}_lsci_strict_ood_v4_confirm_none_seed${seed}_strict_ood_seed${seed}_v4.json" \
    "$PRIOR" 0
  run_one semantic_real "$seed" \
    "experiments/records/${DATASET}_lsci_strict_ood_e3_v4_confirm_semantic_real_seed${seed}_strict_ood_e3_seed${seed}_v4.json" \
    "$PRIOR" "$ALPHA"
  run_one semantic_shuffled "$seed" \
    "experiments/records/${DATASET}_lsci_strict_ood_e4_shuffle_v4_confirm_semantic_shuffled_seed${seed}_strict_ood_e4_shuffle_seed${seed}_v4.json" \
    "$SHUFFLED_PRIOR" "$ALPHA"
done

"$PYTHON_BIN" scripts/summarize_v4_ood.py --dataset "$DATASET" \
  --confirmation "$CONFIRMATION" --out "experiments/reports/${DATASET}_v4_semantic_ood.json" \
  > "logs/${DATASET}_v4_ood_summary.log" 2>&1
echo "corrected-v4 frozen OOD evaluation complete: $DATASET"
