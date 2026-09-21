#!/usr/bin/env bash
set -euo pipefail

# Validation alpha selection followed by frozen one-shot OOD A0/A1/A2/A3
# evaluation and five row-permutation controls. A3 always reuses A2 checkpoint.
DATASET="${1:?dataset required}"
GPU="${2:-0}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/p520/anaconda3/envs/zpp1/bin/python}"
DATA_ROOT="$ROOT/data_strict/processed/$DATASET/v1_strict"
PRIOR="$DATA_ROOT/features/semantic_prior.pt"
SHUFFLE_DIR="$DATA_ROOT/features/shuffle_controls_v6"
ALPHA_REPORT="experiments/reports/${DATASET}_v6_module_ablation_alpha_validation.json"
SEEDS=(1024 2048 3072 4096 5120)
SHUFFLE_SEEDS=(1101 1102 1103 1104 1105)

cd "$ROOT"
mkdir -p logs experiments/records experiments/reports "$SHUFFLE_DIR"

record_for() {
  local arm="$1" seed="$2"
  case "$arm" in
    a0) printf 'experiments/records/%s_lsci_strict_ood_v6_module_ablation_a0_seed%s_strict_ood_seed%s_v4.json' "$DATASET" "$seed" "$seed" ;;
    a1) printf 'experiments/records/%s_lsci_strict_ood_e4_lamsem0_v6_module_ablation_a1_seed%s_strict_ood_e4_lamsem0_seed%s_v4.json' "$DATASET" "$seed" "$seed" ;;
    a2) printf 'experiments/records/%s_lsci_strict_ood_e3_v6_module_ablation_a2_seed%s_strict_ood_e3_seed%s_v4.json' "$DATASET" "$seed" "$seed" ;;
    *) return 2 ;;
  esac
}

a2_records=()
a0_records=()
a1_records=()
for seed in "${SEEDS[@]}"; do
  for arm in a0 a1 a2; do
    required="$(record_for "$arm" "$seed")"
    [[ -f "$required" ]] || { echo "missing training record $required" >&2; exit 2; }
  done
  a0_records+=("$(record_for a0 "$seed")")
  a1_records+=("$(record_for a1 "$seed")")
  a2_records+=("$(record_for a2 "$seed")")
done

"$PYTHON_BIN" scripts/audit_module_ablation_training.py --dataset "$DATASET" \
  --a0 "${a0_records[@]}" --a1 "${a1_records[@]}" --a2 "${a2_records[@]}" \
  --out "experiments/reports/${DATASET}_v6_module_ablation_training_audit.json" \
  > "logs/${DATASET}_v6_module_ablation_training_audit.log" 2>&1

env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" scripts/validate_late_fusion.py \
  --dataset "$DATASET" --data_root "$DATA_ROOT" \
  --source_records "${a2_records[@]}" --semantic_prior_path "$PRIOR" \
  --alphas '0,0.025,0.05,0.075,0.1,0.15,0.2,0.25,0.5,0.75,1,1.25,1.5,2' \
  --selection_seeds '1024,2048,3072' --confirmation_seeds '4096,5120' \
  --protocol_version corrected_v6_module_ablation_alpha_validation_only \
  --out "$ALPHA_REPORT" > "logs/${DATASET}_v6_module_ablation_alpha_validation.log" 2>&1

alpha="$($PYTHON_BIN -c 'import json,sys; print(json.load(open(sys.argv[1]))["summary"]["selected_alpha"])' "$ALPHA_REPORT")"
"$PYTHON_BIN" scripts/build_semantic_shuffle_controls.py \
  --source "$PRIOR" --out_dir "$SHUFFLE_DIR" \
  --seeds '1101,1102,1103,1104,1105' \
  > "logs/${DATASET}_v6_build_shuffle_controls.log" 2>&1

evaluate() {
  local source="$1" prior="$2" weight="$3" label="$4" out="$5"
  [[ -f "$out" ]] && return 0
  env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" scripts/evaluate_late_fusion.py \
    --source_record "$source" --data_root "$DATA_ROOT" \
    --semantic_prior_path "$prior" --eval_split ood --alpha "$weight" \
    --protocol_version corrected_v6_module_ablation_frozen_ood \
    --comparison_label "$label" --out "$out" \
    > "logs/$(basename "${out%.json}").log" 2>&1
}

for seed in "${SEEDS[@]}"; do
  a0="$(record_for a0 "$seed")"
  a1="$(record_for a1 "$seed")"
  a2="$(record_for a2 "$seed")"
  evaluate "$a0" "$PRIOR" 0 A0 "experiments/records/${DATASET}_v6_module_ablation_ood_a0_seed${seed}.json"
  evaluate "$a1" "$PRIOR" 0 A1 "experiments/records/${DATASET}_v6_module_ablation_ood_a1_seed${seed}.json"
  # This single record contains A2 in baseline_metrics and A3 in best_metrics.
  evaluate "$a2" "$PRIOR" "$alpha" A3_reuses_A2 \
    "experiments/records/${DATASET}_v6_module_ablation_ood_a3_real_seed${seed}.json"
  for shuffle_seed in "${SHUFFLE_SEEDS[@]}"; do
    shuffled="$SHUFFLE_DIR/semantic_prior_shuffled_seed${shuffle_seed}.pt"
    evaluate "$a2" "$shuffled" "$alpha" "A3_shuffle_${shuffle_seed}" \
      "experiments/records/${DATASET}_v6_module_ablation_ood_a3_shuffle${shuffle_seed}_seed${seed}.json"
  done
done

"$PYTHON_BIN" scripts/summarize_module_ablation.py --dataset "$DATASET" \
  --root "$ROOT" --out "experiments/reports/${DATASET}_v6_module_ablation_summary.json" \
  --markdown "experiments/reports/${DATASET}_v6_module_ablation_summary.md" \
  > "logs/${DATASET}_v6_module_ablation_summary.log" 2>&1
echo "${DATASET}_V6_MODULE_ABLATION_EVALUATION_COMPLETE alpha=${alpha}"
