#!/usr/bin/env bash
set -euo pipefail

# Frozen one-shot OOD evaluation for the corrected-v5 fair comparison.
# Usage: bash scripts/run_v5_fair_bpr25_ood.sh DATASET [GPU]
DATASET="${1:?dataset required}"
GPU="${2:-0}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/p520/anaconda3/envs/zpp1/bin/python}"
DATA_ROOT="$ROOT/data_strict/processed/$DATASET/v1_strict"
PRIOR="$DATA_ROOT/features/semantic_prior.pt"
AUDIT="experiments/reports/${DATASET}_v5_fair_bpr25_validation_audit.json"
SEEDS=(1024 2048 3072 4096 5120)

cd "$ROOT"
mkdir -p logs experiments/records experiments/reports

"$PYTHON_BIN" scripts/audit_fair_method_validation.py --dataset "$DATASET" \
  --lightgcn_records experiments/records/${DATASET}_v5_fair_bpr25_lightgcn_seed*.json \
  --causaldiffrec_records experiments/records/${DATASET}_strict_ood_v5_fair_bpr25_e0_seed*_v4.json \
  --full_records experiments/records/${DATASET}_lsci_strict_ood_e3_v5_fair_bpr25_full_seed*_v4.json \
  --semantic_confirmation "" --out "$AUDIT" \
  > "logs/${DATASET}_v5_fair_bpr25_validation_audit.log" 2>&1

allowed=$("$PYTHON_BIN" -c \
  'import json,sys; print(str(bool(json.load(open(sys.argv[1]))["ood_test_allowed"])).lower())' \
  "$AUDIT")
[[ "$allowed" == true ]] || { echo "fairness audit rejected OOD test" >&2; exit 3; }

for seed in "${SEEDS[@]}"; do
  light="experiments/records/${DATASET}_v5_fair_bpr25_lightgcn_seed${seed}.json"
  e0="experiments/records/${DATASET}_strict_ood_v5_fair_bpr25_e0_seed${seed}_strict_ood_seed${seed}_v4.json"
  full="experiments/records/${DATASET}_lsci_strict_ood_e3_v5_fair_bpr25_full_seed${seed}_strict_ood_e3_seed${seed}_v4.json"
  for arm in lightgcn causaldiffrec full_method; do
    source="$light"; alpha=0
    [[ "$arm" == causaldiffrec ]] && source="$e0"
    if [[ "$arm" == full_method ]]; then source="$full"; alpha=0.75; fi
    out="experiments/records/${DATASET}_v5_fair_bpr25_ood_${arm}_seed${seed}.json"
    log="logs/${DATASET}_v5_fair_bpr25_ood_${arm}_seed${seed}.log"
    [[ -f "$out" ]] && continue
    env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" \
      scripts/evaluate_late_fusion.py --source_record "$source" --data_root "$DATA_ROOT" \
      --semantic_prior_path "$PRIOR" --eval_split ood --alpha "$alpha" --out "$out" \
      --protocol_version corrected_v5_fair_bpr25_frozen_ood \
      --comparison_label "$arm" > "$log" 2>&1
  done
done

"$PYTHON_BIN" scripts/summarize_fair_method_ood.py --dataset "$DATASET" --audit "$AUDIT" \
  --lightgcn_records experiments/records/${DATASET}_v5_fair_bpr25_ood_lightgcn_seed*.json \
  --causaldiffrec_records experiments/records/${DATASET}_v5_fair_bpr25_ood_causaldiffrec_seed*.json \
  --full_records experiments/records/${DATASET}_v5_fair_bpr25_ood_full_method_seed*.json \
  --out experiments/reports/${DATASET}_v5_fair_bpr25_method_comparison.json \
  > "logs/${DATASET}_v5_fair_bpr25_method_comparison.log" 2>&1
echo "${DATASET}_V5_FAIR_BPR25_OOD_COMPLETE"
