#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/p520/anaconda3/envs/zpp1/bin/python}"
GPU="${1:-0}"
DATASET=movielens1m
DATA_ROOT="$ROOT/data_strict/processed/$DATASET/v1_strict"
VALIDATION="experiments/reports/${DATASET}_v4_activity_adaptive_validation.json"
SEEDS=(1024 2048 3072 4096 5120)

cd "$ROOT"
mkdir -p logs experiments/records experiments/reports
"$PYTHON_BIN" scripts/validate_activity_adaptive_fusion.py \
  --dataset "$DATASET" --data_root "$DATA_ROOT" \
  --source_records experiments/records/${DATASET}_pure_lightgcn_v4_bpr200_seed*.json \
  --uniform_validation_report experiments/reports/${DATASET}_pure_lightgcn_v4_bpr200_latefusion_validation.json \
  --out "$VALIDATION" > logs/${DATASET}_v4_activity_adaptive_validation.log 2>&1

allowed=$("$PYTHON_BIN" -c \
  'import json,sys; print(str(bool(json.load(open(sys.argv[1]))["summary"]["ood_test_allowed"])).lower())' \
  "$VALIDATION")
if [[ "$allowed" != true ]]; then
  echo "ACTIVITY_ADAPTIVE_VALIDATION_FAILED_GATE"
  exit 0
fi

for seed in "${SEEDS[@]}"; do
  source="experiments/records/${DATASET}_pure_lightgcn_v4_bpr200_seed${seed}.json"
  for prior in real shuffle; do
    out="experiments/records/${DATASET}_v4_activity_adaptive_ood_${prior}_seed${seed}.json"
    [[ -f "$out" ]] && continue
    env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" \
      scripts/evaluate_activity_adaptive_fusion.py --validation_report "$VALIDATION" \
      --source_record "$source" --data_root "$DATA_ROOT" --semantic_prior "$prior" \
      --out "$out" > "logs/${DATASET}_v4_activity_adaptive_ood_${prior}_seed${seed}.log" 2>&1
  done
done

"$PYTHON_BIN" scripts/summarize_activity_adaptive_ood.py --dataset "$DATASET" --validation_report "$VALIDATION" \
  --real_records experiments/records/${DATASET}_v4_activity_adaptive_ood_real_seed*.json \
  --shuffle_records experiments/records/${DATASET}_v4_activity_adaptive_ood_shuffle_seed*.json \
  --out experiments/reports/${DATASET}_v4_activity_adaptive_ood.json \
  > logs/${DATASET}_v4_activity_adaptive_ood_summary.log 2>&1
echo "MOVIELENS_V4_ACTIVITY_ADAPTIVE_COMPLETE"
