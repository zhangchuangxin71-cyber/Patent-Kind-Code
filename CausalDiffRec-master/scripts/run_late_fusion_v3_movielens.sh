#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=/home/p520/anaconda3/envs/zpp1/bin/python
DATA_ROOT=data_strict/processed/movielens1m/v1_strict
ALPHA=0.75
SEEDS=(1024 2048 3072 4096 5120)

mkdir -p logs experiments/records
for seed in "${SEEDS[@]}"; do
    source="experiments/records/movielens1m_lsci_strict_ood_corrected_v3_formal_stage1_strict_ood_seed${seed}_v3.json"
    out="experiments/records/movielens1m_latefusion_alpha075_strict_ood_seed${seed}_v3.json"
    log="logs/movielens1m_latefusion_alpha075_seed${seed}.log"
    if [[ -f "$out" ]]; then
        echo "[skip] $out"
        continue
    fi
    echo "[start] $out"
    env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" \
        scripts/evaluate_late_fusion.py \
        --source_record "$source" --data_root "$DATA_ROOT" \
        --eval_split ood --alpha "$ALPHA" --out "$out" > "$log" 2>&1
    echo "[done] $out"
done

echo "MOVIELENS_LATE_FUSION_V3_COMPLETE"
