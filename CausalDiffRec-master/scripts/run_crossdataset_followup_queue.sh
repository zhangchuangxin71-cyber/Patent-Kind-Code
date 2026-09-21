#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
while pgrep -f '[r]un_yelp_v4_strong_backbone.sh' >/dev/null; do
  sleep 30
done
bash scripts/run_amazon_beauty_v4_adaptive.sh "${1:-0}"
