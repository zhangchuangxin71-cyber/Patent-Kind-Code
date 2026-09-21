#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:?dataset required}"
VALIDATION="${2:?validation report required}"
GPU="${3:-0}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/p520/anaconda3/envs/zpp1/bin/python}"
DATA_ROOT="$ROOT/data_strict/processed/$DATASET/v1_strict"
SEEDS=(1024 2048 3072 4096 5120)
cd "$ROOT"

read -r allowed alpha < <("$PYTHON_BIN" -c \
  'import json,sys; s=json.load(open(sys.argv[1]))["summary"]; print(str(bool(s["ood_test_allowed"])).lower(),s["selected_alpha"])' \
  "$VALIDATION")
if [[ "$allowed" != true ]]; then
  echo "${DATASET}_UNIFORM_VALIDATION_FAILED_GATE"
  exit 0
fi
for seed in "${SEEDS[@]}"; do
  source=$("$PYTHON_BIN" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["per_seed"][sys.argv[2]]["source_record"])' \
    "$VALIDATION" "$seed")
  for prior_name in semantic_real semantic_shuffled; do
    prior="$DATA_ROOT/features/semantic_prior.pt"
    [[ "$prior_name" == semantic_shuffled ]] && prior="$DATA_ROOT/features/semantic_prior_shuffled.pt"
    out="experiments/records/${DATASET}_v4_uniform_ood_${prior_name}_seed${seed}.json"
    [[ -f "$out" ]] && continue
    env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" \
      scripts/evaluate_late_fusion.py --source_record "$source" --data_root "$DATA_ROOT" \
      --semantic_prior_path "$prior" --eval_split ood --alpha "$alpha" --out "$out" \
      --protocol_version corrected_v4_frozen_ood --comparison_label "$prior_name" \
      > "logs/${DATASET}_v4_uniform_ood_${prior_name}_seed${seed}.log" 2>&1
  done
done
"$PYTHON_BIN" scripts/summarize_v4_latefusion_ood.py --dataset "$DATASET" \
  --validation_report "$VALIDATION" \
  --real_records experiments/records/${DATASET}_v4_uniform_ood_semantic_real_seed*.json \
  --shuffle_records experiments/records/${DATASET}_v4_uniform_ood_semantic_shuffled_seed*.json \
  --out "experiments/reports/${DATASET}_v4_uniform_ood.json" \
  > "logs/${DATASET}_v4_uniform_ood_summary.log" 2>&1
echo "${DATASET}_V4_UNIFORM_OOD_COMPLETE"
