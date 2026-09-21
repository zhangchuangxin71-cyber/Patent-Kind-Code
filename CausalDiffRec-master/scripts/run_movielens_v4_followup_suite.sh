#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
while [[ ! -f experiments/reports/movielens1m_v4_refresh_budget_curve.json ]]; do
  sleep 30
done
bash scripts/run_movielens_v4_activity_adaptive.sh "${1:-0}"
