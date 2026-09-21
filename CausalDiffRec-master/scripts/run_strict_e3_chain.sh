#!/usr/bin/env bash
set -euo pipefail

# Continue an in-progress E3 chain. The caller may provide an active tmux
# session to wait for, followed by the next dataset to run.
WAIT_SESSION="${1:?tmux session to wait for}"
NEXT_DATASET="${2:?next dataset required}"
GPU="${3:-0}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
MASTER_LOG="logs/strict_e3_chain.log"

{
  echo "[$(date '+%F %T')] waiting for session=$WAIT_SESSION"
  while tmux has-session -t "$WAIT_SESSION" 2>/dev/null; do
    sleep 30
  done
  echo "[$(date '+%F %T')] session=$WAIT_SESSION finished; starting dataset=$NEXT_DATASET"
  PYTHON_BIN="$PYTHON_BIN" bash scripts/run_strict_e3_semantic_multiseed.sh \
    "$NEXT_DATASET" "$GPU" 25
  echo "[$(date '+%F %T')] dataset=$NEXT_DATASET finished"
} 2>&1 | tee -a "$MASTER_LOG"
