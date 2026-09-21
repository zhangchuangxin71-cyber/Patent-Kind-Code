#!/usr/bin/env bash
set -euo pipefail

# One-shot OOD evaluation of the two equal-BPR-budget arms.  Alpha and the
# real/shuffled comparison set are frozen in their validation reports.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/p520/anaconda3/envs/zpp1/bin/python}"
GPU="${1:-0}"
DATASET=movielens1m
DATA_ROOT="$ROOT/data_strict/processed/$DATASET/v1_strict"
REAL_PRIOR="$DATA_ROOT/features/semantic_prior.pt"
SHUFFLE_PRIOR="$DATA_ROOT/features/semantic_prior_shuffled.pt"
SEEDS=(1024 2048 3072 4096 5120)

cd "$ROOT"
mkdir -p logs experiments/records experiments/reports

run_arm() {
  local arm="$1" validation="$2" source_pattern="$3"
  local alpha
  alpha=$("$PYTHON_BIN" -c \
    'import json,sys; s=json.load(open(sys.argv[1], encoding="utf-8"))["summary"]; assert s["ood_test_allowed"]; print(s["selected_alpha"])' \
    "$validation")
  for seed in "${SEEDS[@]}"; do
    local source
    source=$(compgen -G "${source_pattern//SEED/$seed}" | head -n 1)
    [[ -n "$source" && -f "$source" ]] || { echo "missing source for $arm seed=$seed" >&2; exit 2; }
    for prior_name in semantic_real semantic_shuffled; do
      local prior="$REAL_PRIOR"
      [[ "$prior_name" == semantic_shuffled ]] && prior="$SHUFFLE_PRIOR"
      local out="experiments/records/${DATASET}_${arm}_v4_bpr200_ood_${prior_name}_seed${seed}.json"
      local log="logs/${DATASET}_${arm}_v4_bpr200_ood_${prior_name}_seed${seed}.log"
      [[ -f "$out" ]] && continue
      env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" \
        scripts/evaluate_late_fusion.py --source_record "$source" --data_root "$DATA_ROOT" \
        --semantic_prior_path "$prior" --eval_split ood --alpha "$alpha" --out "$out" \
        --protocol_version corrected_v4_equal_bpr200_frozen_ood \
        --comparison_label "${arm}_${prior_name}" > "$log" 2>&1
    done
  done
  "$PYTHON_BIN" scripts/summarize_v4_latefusion_ood.py --dataset "$DATASET" \
    --validation_report "$validation" \
    --real_records experiments/records/${DATASET}_${arm}_v4_bpr200_ood_semantic_real_seed*.json \
    --shuffle_records experiments/records/${DATASET}_${arm}_v4_bpr200_ood_semantic_shuffled_seed*.json \
    --out "experiments/reports/${DATASET}_${arm}_v4_bpr200_ood.json" \
    > "logs/${DATASET}_${arm}_v4_bpr200_ood_summary.log" 2>&1
}

run_arm pure_lightgcn \
  experiments/reports/${DATASET}_pure_lightgcn_v4_bpr200_latefusion_validation.json \
  "experiments/records/${DATASET}_pure_lightgcn_v4_bpr200_seedSEED.json"
run_arm lsci_none \
  experiments/reports/${DATASET}_lsci_none_v4_bpr200_latefusion_validation.json \
  "experiments/records/${DATASET}_lsci_strict_ood_lsci_none_v4_bpr200_seedSEED_strict_ood_seedSEED_v4.json"

"$PYTHON_BIN" scripts/summarize_budgeted_ood.py \
  --pure_report experiments/reports/${DATASET}_pure_lightgcn_v4_bpr200_ood.json \
  --lsci_report experiments/reports/${DATASET}_lsci_none_v4_bpr200_ood.json \
  --out experiments/reports/${DATASET}_v4_bpr200_budgeted_ood_comparison.json \
  > logs/${DATASET}_v4_bpr200_budgeted_ood_comparison.log 2>&1
echo "MOVIELENS_V4_BPR200_BUDGETED_OOD_COMPLETE"
