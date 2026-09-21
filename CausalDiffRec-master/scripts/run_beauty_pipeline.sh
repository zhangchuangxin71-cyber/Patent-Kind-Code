#!/usr/bin/env bash
# Amazon Beauty: prepare (k-core 5/5) -> validate -> SBERT prior -> E0+E3(+shuffle)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate zpp1
export http_proxy="${http_proxy:-http://127.0.0.1:7897}"
export https_proxy="${https_proxy:-http://127.0.0.1:7897}"

echo "[$(date -Iseconds)] re-prepare amazon_beauty (registry k-core 5/5)"
python -u scripts/data/prepare_amazon_beauty.py
python -u scripts/data/validate_dataset.py --dataset amazon_beauty
echo "[$(date -Iseconds)] build prior"
python -u scripts/build_semantic_prior.py -d amazon_beauty --mode raw --backend sbert
echo "[$(date -Iseconds)] train E0+E3 multiseed"
bash scripts/run_amazon_beauty_e0_e3_multiseed.sh
echo "[$(date -Iseconds)] DONE beauty pipeline"
