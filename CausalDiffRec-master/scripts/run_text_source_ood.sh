#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:?dataset required}"
DATA_ROOT="${2:?data root required}"
VALIDATION="${3:?validation report required}"
GPU="${4:-0}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/p520/anaconda3/envs/zpp1/bin/python}"
SEEDS=(1024 2048 3072 4096 5120)

cd "$ROOT"
mkdir -p logs experiments/records experiments/reports

read -r allowed alpha raw_alpha llm_prior raw_prior < <("$PYTHON_BIN" -c '
import json,sys
s=json.load(open(sys.argv[1],encoding="utf-8"))["summary"]
print(str(bool(s["ood_test_allowed"])).lower(),
      s["selected_alpha"]["shared_llm_selected"],
      s["selected_alpha"]["raw_independently_selected"],
      s["llm_prior_path"],s["raw_prior_path"])
' "$VALIDATION")
if [[ "$allowed" != true ]]; then
  echo "${DATASET}: validation gate rejected OOD test; no test ground truth loaded"
  exit 0
fi

for seed in "${SEEDS[@]}"; do
  source=$("$PYTHON_BIN" -c '
import json,sys
print(json.load(open(sys.argv[1],encoding="utf-8"))["per_seed"][sys.argv[2]]["source_record"])
' "$VALIDATION" "$seed")
  for arm in llm raw; do
    prior="$llm_prior"
    [[ "$arm" == raw ]] && prior="$raw_prior"
    out="experiments/records/${DATASET}_v5_raw_vs_deepseek_ood_${arm}_seed${seed}.json"
    log="logs/${DATASET}_v5_raw_vs_deepseek_ood_${arm}_seed${seed}.log"
    [[ -f "$out" ]] && continue
    env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" \
      scripts/evaluate_late_fusion.py \
      --source_record "$source" --data_root "$DATA_ROOT" \
      --semantic_prior_path "$prior" --eval_split ood --alpha "$alpha" \
      --out "$out" \
      --protocol_version corrected_v5_raw_vs_deepseek_frozen_ood \
      --comparison_label "${arm}_text_sbert" > "$log" 2>&1
  done
done

# Secondary fairness view: equal alpha search spaces, but each text source uses
# its own frozen validation-selected alpha.  The LLM-own value already equals
# the shared alpha, so only the raw arm needs an additional evaluation pass.
for seed in "${SEEDS[@]}"; do
  source=$("$PYTHON_BIN" -c '
import json,sys
print(json.load(open(sys.argv[1],encoding="utf-8"))["per_seed"][sys.argv[2]]["source_record"])
' "$VALIDATION" "$seed")
  out="experiments/records/${DATASET}_v5_raw_vs_deepseek_ood_raw_own_seed${seed}.json"
  log="logs/${DATASET}_v5_raw_vs_deepseek_ood_raw_own_seed${seed}.log"
  [[ -f "$out" ]] && continue
  env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" \
    scripts/evaluate_late_fusion.py \
    --source_record "$source" --data_root "$DATA_ROOT" \
    --semantic_prior_path "$raw_prior" --eval_split ood --alpha "$raw_alpha" \
    --out "$out" \
    --protocol_version corrected_v5_raw_vs_deepseek_frozen_ood \
    --comparison_label "raw_text_sbert_own_validation_alpha" > "$log" 2>&1
done

"$PYTHON_BIN" scripts/summarize_text_source_ood.py \
  --dataset "$DATASET" --validation_report "$VALIDATION" \
  --llm_records experiments/records/${DATASET}_v5_raw_vs_deepseek_ood_llm_seed*.json \
  --raw_records experiments/records/${DATASET}_v5_raw_vs_deepseek_ood_raw_seed*.json \
  --raw_own_records experiments/records/${DATASET}_v5_raw_vs_deepseek_ood_raw_own_seed*.json \
  --out "experiments/reports/${DATASET}_v5_raw_vs_deepseek_ood.json" \
  > "logs/${DATASET}_v5_raw_vs_deepseek_ood_summary.log" 2>&1
echo "${DATASET}: corrected-v5 raw-vs-DeepSeek OOD complete, alpha=${alpha}"
