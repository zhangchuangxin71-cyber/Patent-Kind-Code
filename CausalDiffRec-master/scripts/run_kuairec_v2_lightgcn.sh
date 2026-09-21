#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/p520/anaconda3/envs/zpp1/bin/python}"
GPU="${1:-0}"
DATASET=kuairec
DATA_ROOT="$ROOT/data_strict/processed/kuairec/v2_exposure"
SEEDS=(1024 2048 3072 4096 5120)

cd "$ROOT"
mkdir -p logs experiments/records experiments/checkpoints
for seed in "${SEEDS[@]}"; do
  record="experiments/records/kuairec_v2_lightgcn_seed${seed}.jsonl"
  [[ -f "$record" ]] && continue
  env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -u scripts/train_lightgcn_strict.py \
    --dataset "$DATASET" --data_root "$DATA_ROOT" --seed "$seed" \
    --epochs 25 --min_epochs 25 --batch_size 4096 --embedding_dim 8 --layers 3 \
    --protocol_version corrected_v4_kuairec_exposure_lightgcn \
    --out "$record" --checkpoint "experiments/checkpoints/kuairec_v2_lightgcn_seed${seed}.pt" \
    > "logs/kuairec_v2_lightgcn_seed${seed}.log" 2>&1
done
echo "KUAIREC_V2_LIGHTGCN_COMPLETE"
