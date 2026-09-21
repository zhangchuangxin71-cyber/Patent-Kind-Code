#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# Recommendation checkpoints were trained with this DGL-compatible environment.
# The zcx3 environment is used only to build SBERT priors; its DGL/GraphBolt
# package does not match its installed PyTorch build.
PYTHON_BIN="${PYTHON_BIN:-/home/p520/anaconda3/envs/zpp1/bin/python}"
ALPHAS="0,0.025,0.05,0.075,0.1,0.15,0.2,0.25,0.5,0.75,1,1.25,1.5,2"

cd "$ROOT"

food_records=(experiments/records/food_lightgcn_strict_ood_seed*_v3.json)
yelp_records=(experiments/records/yelp2018_lsci_strict_ood_v4_lfval_none_seed*_strict_ood_seed*_v4.json)
kuairec_records=(experiments/records/kuairec_v2_lightgcn_seed*.jsonl)

for records_name in food_records yelp_records kuairec_records; do
  declare -n records_ref="$records_name"
  if [[ "${#records_ref[@]}" -ne 5 ]]; then
    echo "Expected 5 records in $records_name, found ${#records_ref[@]}" >&2
    exit 1
  fi
done

"$PYTHON_BIN" -u scripts/validate_text_source_ablation.py \
  --dataset food \
  --data_root data_strict/processed/food/v1_strict \
  --source_records "${food_records[@]}" \
  --llm_prior_path data_strict/processed/food/v1_strict/features/semantic_prior.pt \
  --raw_prior_path data_strict/processed/food/v1_strict/features/semantic_prior_raw_sbert.pt \
  --alphas "$ALPHAS" \
  --out experiments/reports/food_v5_raw_vs_deepseek_validation.json

"$PYTHON_BIN" -u scripts/validate_text_source_ablation.py \
  --dataset yelp2018 \
  --data_root data_strict/processed/yelp2018/v1_strict \
  --source_records "${yelp_records[@]}" \
  --llm_prior_path data_strict/processed/yelp2018/v1_strict/features/semantic_prior.pt \
  --raw_prior_path data_strict/processed/yelp2018/v1_strict/features/semantic_prior_raw_sbert.pt \
  --alphas "$ALPHAS" \
  --out experiments/reports/yelp2018_v5_raw_vs_deepseek_validation.json

"$PYTHON_BIN" -u scripts/validate_text_source_ablation.py \
  --dataset kuairec \
  --data_root data_strict/processed/kuairec/v2_exposure \
  --source_records "${kuairec_records[@]}" \
  --llm_prior_path data_strict/processed/kuairec/v2_exposure/features/semantic_prior.pt \
  --raw_prior_path data_strict/processed/kuairec/v1_strict/features/semantic_prior_raw_sbert.pt \
  --alphas "$ALPHAS" \
  --out experiments/reports/kuairec_v5_raw_vs_deepseek_validation.json
