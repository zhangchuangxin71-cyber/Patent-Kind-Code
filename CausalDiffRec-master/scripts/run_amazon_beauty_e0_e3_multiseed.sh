#!/usr/bin/env bash
# Amazon Beauty: E0 then LSCI E3 (5 seeds), then shuffle diag seed1024.
set -euo pipefail
GPU="${1:-0}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
DATA_ROOT="data_strict/processed/amazon_beauty/v1_strict"
PRIOR="$DATA_ROOT/features/semantic_prior.pt"
SEEDS=(1024 2048 3072 4096 5120)
mkdir -p logs experiments/records checkpoints
MASTER="logs/amazon_beauty_e0_e3_master.log"
{
  echo "[$(date '+%F %T')] Amazon Beauty E0 start"
  for SEED in "${SEEDS[@]}"; do
    OUT="experiments/records/amazon_beauty_strict_ood_seed${SEED}.json"
    [[ -f "$OUT" ]] && { echo "skip E0 $SEED"; continue; }
    echo "[$(date '+%F %T')] >>> E0 seed=$SEED"
    CUDA_VISIBLE_DEVICES="$GPU" python -u train.py \
      -d amazon_beauty -e 25 --seed "$SEED" \
      --data_root "$DATA_ROOT" --eval_split ood --run_id "_e0" \
      2>&1 | tee "logs/amazon_beauty_e0_seed${SEED}.log"
  done
  echo "[$(date '+%F %T')] Amazon Beauty E3 start"
  for SEED in "${SEEDS[@]}"; do
    if ls experiments/records/amazon_beauty_lsci*e3*seed${SEED}*.json >/dev/null 2>&1; then
      echo "skip E3 $SEED"; continue
    fi
    echo "[$(date '+%F %T')] >>> E3 seed=$SEED"
    CUDA_VISIBLE_DEVICES="$GPU" python -u train_lsci.py \
      -d amazon_beauty -e 25 --seed "$SEED" \
      --data_root "$DATA_ROOT" --eval_split ood \
      --use_semantic_prior --semantic_prior_path "$PRIOR" \
      --lambda_sem 0.1 --run_id "_e3" \
      2>&1 | tee "logs/amazon_beauty_e3_seed${SEED}.log"
  done
  # shuffle prior + one seed
  echo "[$(date '+%F %T')] shuffle diag"
  python - <<'PY'
import torch
from pathlib import Path
from copy import deepcopy
root=Path("data_strict/processed/amazon_beauty/v1_strict")
src=root/"features"/"semantic_prior.pt"
dst=root/"features"/"semantic_prior_shuffled.pt"
obj=torch.load(src,map_location="cpu")
emb=obj["item_emb"].clone(); mask=obj["semantic_mask"].clone()
meta=deepcopy(obj.get("meta") or {})
g=torch.Generator().manual_seed(20260731)
perm=torch.randperm(emb.size(0), generator=g)
emb,mask=emb[perm],mask[perm]
meta.update({"diagnostic":"row_shuffle","shuffle_seed":20260731,"source_prior":str(src),"backend_original":meta.get("backend","sbert")})
torch.save({"item_emb":emb,"semantic_mask":mask,"meta":meta}, dst)
print("wrote", dst)
PY
  CUDA_VISIBLE_DEVICES="$GPU" python -u train_lsci.py \
    -d amazon_beauty -e 25 --seed 1024 \
    --data_root "$DATA_ROOT" --eval_split ood \
    --use_semantic_prior \
    --semantic_prior_path "$DATA_ROOT/features/semantic_prior_shuffled.pt" \
    --lambda_sem 0.1 --run_id "_e4_shuffle" \
    2>&1 | tee "logs/amazon_beauty_e4_shuffle_seed1024.log"
  echo "[$(date '+%F %T')] Amazon Beauty E0+E3+shuffle finished"
} 2>&1 | tee -a "$MASTER"
