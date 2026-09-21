#!/usr/bin/env bash
# After Douban E0 frees GPU: MovieLens-1M E0 then LSCI E3 (5 seeds each).
# Usage: bash scripts/run_movielens1m_e0_e3_multiseed.sh [gpu_id] [wait_pid]
set -euo pipefail
GPU="${1:-0}"
WAIT_PID="${2:-}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
DATA_ROOT="data_strict/processed/movielens1m/v1_strict"
PRIOR="$DATA_ROOT/features/semantic_prior.pt"
SEEDS=(1024 2048 3072 4096 5120)
mkdir -p logs experiments/records checkpoints
MASTER="logs/movielens1m_e0_e3_master.log"

{
  if [[ -n "$WAIT_PID" ]]; then
    echo "[$(date '+%F %T')] wait for PID=$WAIT_PID before MovieLens jobs"
    while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
    echo "[$(date '+%F %T')] wait done"
  fi

  echo "[$(date '+%F %T')] MovieLens-1M E0 start gpu=$GPU"
  for SEED in "${SEEDS[@]}"; do
    OUT="experiments/records/movielens1m_strict_ood_seed${SEED}.json"
    if [[ -f "$OUT" ]]; then
      echo "skip E0 seed=$SEED"
      continue
    fi
    echo "[$(date '+%F %T')] >>> E0 seed=$SEED"
    CUDA_VISIBLE_DEVICES="$GPU" python -u train.py \
      -d movielens1m -e 25 --seed "$SEED" \
      --data_root "$DATA_ROOT" --eval_split ood \
      --run_id "_e0" \
      2>&1 | tee "logs/movielens1m_e0_seed${SEED}.log"
  done

  echo "[$(date '+%F %T')] MovieLens-1M LSCI E3 start"
  for SEED in "${SEEDS[@]}"; do
    OUT="experiments/records/movielens1m_lsci_strict_ood_seed${SEED}.json"
    # train_lsci may use different tag; also check _e3 suffix variants
    if ls experiments/records/movielens1m_lsci*seed${SEED}*.json >/dev/null 2>&1; then
      echo "skip E3 seed=$SEED"
      continue
    fi
    echo "[$(date '+%F %T')] >>> E3 seed=$SEED"
    CUDA_VISIBLE_DEVICES="$GPU" python -u train_lsci.py \
      -d movielens1m -e 25 --seed "$SEED" \
      --data_root "$DATA_ROOT" --eval_split ood \
      --use_semantic_prior --semantic_prior_path "$PRIOR" \
      --lambda_sem 0.1 \
      --run_id "_e3" \
      2>&1 | tee "logs/movielens1m_e3_seed${SEED}.log"
  done
  echo "[$(date '+%F %T')] MovieLens-1M E0+E3 all seeds finished"
} 2>&1 | tee -a "$MASTER"
