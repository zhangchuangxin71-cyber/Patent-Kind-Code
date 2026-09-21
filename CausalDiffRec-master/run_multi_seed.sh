#!/usr/bin/env bash
# Multi-seed repeated experiments for CausalDiffRec.
# Usage: bash run_multi_seed.sh [datasets...]
set -euo pipefail
cd "$(dirname "$0")"
PYTHON=/home/p520/anaconda3/envs/zpp1/bin/python
export PYTHONUNBUFFERED=1

DATASETS=("$@")
if [ ${#DATASETS[@]} -eq 0 ]; then
  DATASETS=(yelp2018 douban food kuairec)
fi

# 5 independent seeds for mean ± std
SEEDS=(1024 2048 3072 4096 5120)
EPOCHS=25

mkdir -p logs experiments/records experiments/summaries

echo "======= Multi-seed protocol ======="
echo "datasets: ${DATASETS[*]}"
echo "seeds: ${SEEDS[*]}"
echo "epochs: ${EPOCHS}"
echo "==================================="

for ds in "${DATASETS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    echo "========== START ${ds} seed=${seed} =========="
    ${PYTHON} -u train.py --dataset "${ds}" --seed "${seed}" --epochs "${EPOCHS}" \
      2>&1 | tee "logs/${ds}_seed${seed}_train.log"
    echo "========== DONE ${ds} seed=${seed} =========="
  done
  ${PYTHON} -u scripts/aggregate_results.py --dataset "${ds}"
done

${PYTHON} -u scripts/aggregate_results.py --all
echo "All multi-seed experiments finished."
