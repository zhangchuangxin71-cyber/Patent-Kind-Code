#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON=/home/p520/anaconda3/envs/zpp1/bin/python
export PYTHONUNBUFFERED=1
mkdir -p logs checkpoints

for ds in yelp2018 douban food kuairec; do
  echo "========== START ${ds} $(date '+%F %T') =========="
  ${PYTHON} -u train.py --dataset "${ds}" --epochs 25 2>&1 | tee "logs/${ds}_train.log"
  echo "========== DONE ${ds} $(date '+%F %T') =========="
done

echo "All datasets finished at $(date '+%F %T')"
