#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/p520/anaconda3/envs/zcx3/bin/python}"

cd "$ROOT"

for dataset in food yelp2018 kuairec; do
  "$PYTHON_BIN" -u scripts/build_semantic_prior.py \
    -d "$dataset" \
    --mode raw \
    --backend sbert \
    --sbert_model sentence-transformers/all-MiniLM-L6-v2 \
    --batch_size 64 \
    --out "data_strict/processed/$dataset/v1_strict/features/semantic_prior_raw_sbert.pt"
done
