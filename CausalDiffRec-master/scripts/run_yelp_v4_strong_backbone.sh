#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/p520/anaconda3/envs/zpp1/bin/python}"
GPU="${1:-0}"
DATASET=yelp2018
DATA_ROOT="$ROOT/data_strict/processed/$DATASET/v1_strict"
SEEDS=(1024 2048 3072 4096 5120)
cd "$ROOT"
mkdir -p logs experiments/records experiments/reports checkpoints

for seed in "${SEEDS[@]}"; do
  prior_record="experiments/records/${DATASET}_lightgcn_longrun_strict_ood_seed${seed}_v3.json"
  prior_checkpoint="checkpoints/${DATASET}_lightgcn_longrun_strict_ood_seed${seed}_v3.pt"
  record="experiments/records/${DATASET}_pure_lightgcn_v4_bpr400_seed${seed}.json"
  checkpoint="checkpoints/${DATASET}_pure_lightgcn_v4_bpr400_seed${seed}.pt"
  [[ -f "$record" ]] && continue
  env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" \
    scripts/train_lightgcn_strict.py --dataset "$DATASET" --data_root "$DATA_ROOT" \
    --seed "$seed" --epochs 400 --min_epochs 400 --patience 0 \
    --resume_checkpoint "$prior_checkpoint" --resume_record "$prior_record" \
    --protocol_version corrected_v4_pure_lightgcn_bpr400_validation_only \
    --out "$record" --checkpoint "$checkpoint" \
    > "logs/${DATASET}_pure_lightgcn_v4_bpr400_seed${seed}.log" 2>&1
done

"$PYTHON_BIN" scripts/summarize_lightgcn_budget_curve.py --dataset "$DATASET" \
  --records experiments/records/${DATASET}_pure_lightgcn_v4_bpr400_seed*.json \
  --out "experiments/reports/${DATASET}_v4_lightgcn_budget_curve.json" \
  > "logs/${DATASET}_v4_lightgcn_budget_curve.log" 2>&1

validation="experiments/reports/${DATASET}_v4_strong_latefusion_validation.json"
"$PYTHON_BIN" scripts/validate_late_fusion.py --dataset "$DATASET" --data_root "$DATA_ROOT" \
  --source_records experiments/records/${DATASET}_pure_lightgcn_v4_bpr400_seed*.json \
  --alphas 0,0.025,0.05,0.075,0.1,0.15,0.2,0.25,0.5 \
  --protocol_version corrected_v4_strong_latefusion_validation_only --out "$validation" \
  > "logs/${DATASET}_v4_strong_latefusion_validation.log" 2>&1

bash scripts/run_v4_uniform_ood_dataset.sh "$DATASET" "$validation" "$GPU"
bash scripts/run_v4_activity_adaptive_dataset.sh "$DATASET" \
  "experiments/records/${DATASET}_pure_lightgcn_v4_bpr400_seed*.json" "$validation" "$GPU"
echo "YELP_V4_STRONG_BACKBONE_COMPLETE"
