#!/usr/bin/env bash
set -euo pipefail

# Frozen OOD scoring from v4 validation-selected source checkpoints.
DATASET="${1:?dataset required}"
GPU="${2:-0}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/p520/anaconda3/envs/zpp1/bin/python}"
DATA_ROOT="$ROOT/data_strict/processed/$DATASET/v1_strict"
VALIDATION="$ROOT/experiments/reports/${DATASET}_v4_latefusion_validation.json"
PRIOR="$DATA_ROOT/features/semantic_prior.pt"
SHUFFLED_PRIOR="$DATA_ROOT/features/semantic_prior_shuffled.pt"
SEEDS=(1024 2048 3072 4096 5120)

cd "$ROOT"
mkdir -p logs experiments/records experiments/reports
[[ -f "$VALIDATION" ]] || { echo "missing validation report: $VALIDATION" >&2; exit 2; }
ALPHA=$("$PYTHON_BIN" -c 'import json,sys; s=json.load(open(sys.argv[1], encoding="utf-8"))["summary"]; assert s["ood_test_allowed"]; print(s["selected_alpha"])' "$VALIDATION")

run_one() {
  local arm="$1"
  local seed="$2"
  local prior="$3"
  local out="experiments/records/${DATASET}_v4_lf_ood_${arm}_seed${seed}_v4.json"
  local log="logs/${DATASET}_v4_lf_ood_${arm}_seed${seed}.log"
  local source="experiments/records/${DATASET}_lsci_strict_ood_v4_lfval_none_seed${seed}_strict_ood_seed${seed}_v4.json"
  [[ -f "$out" ]] && return 0
  env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" \
    scripts/evaluate_late_fusion.py --source_record "$source" --data_root "$DATA_ROOT" \
    --semantic_prior_path "$prior" --eval_split ood --alpha "$ALPHA" --out "$out" \
    --protocol_version corrected_v4_frozen_ood --comparison_label "$arm" > "$log" 2>&1
}

for seed in "${SEEDS[@]}"; do
  run_one semantic_real "$seed" "$PRIOR"
  run_one semantic_shuffled "$seed" "$SHUFFLED_PRIOR"
done

"$PYTHON_BIN" scripts/summarize_v4_latefusion_ood.py --dataset "$DATASET" \
  --validation_report "$VALIDATION" \
  --real_records experiments/records/${DATASET}_v4_lf_ood_semantic_real_seed*_v4.json \
  --shuffle_records experiments/records/${DATASET}_v4_lf_ood_semantic_shuffled_seed*_v4.json \
  --out "experiments/reports/${DATASET}_v4_latefusion_ood.json" \
  > "logs/${DATASET}_v4_lf_ood_summary.log" 2>&1
echo "corrected-v4 frozen late-fusion OOD complete: $DATASET alpha=$ALPHA"
