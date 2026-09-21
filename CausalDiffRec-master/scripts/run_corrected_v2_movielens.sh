#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=/home/p520/anaconda3/envs/zpp1/bin/python
DATASET=movielens1m
DATA_ROOT=data_strict/processed/movielens1m/v1_strict
EVAL_SPLIT=ood
SEEDS=(1024 2048 3072 4096 5120)

mkdir -p logs experiments/records checkpoints

run_one() {
    local record=$1
    local log=$2
    shift 2
    if [[ -f "$record" ]]; then
        echo "[skip] $record"
        return
    fi
    echo "[start] $record"
    env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" "$@" > "$log" 2>&1
    if [[ ! -f "$record" ]]; then
        echo "[error] command exited without producing $record" >&2
        return 1
    fi
    echo "[done] $record"
}

for seed in "${SEEDS[@]}"; do
    run_one \
        "experiments/records/${DATASET}_strict_${EVAL_SPLIT}_corrected_v2_formal_e0_strict_${EVAL_SPLIT}_seed${seed}_v2.json" \
        "logs/${DATASET}_e0_corrected_v2_seed${seed}.log" \
        train.py --dataset "$DATASET" --data_root "$DATA_ROOT" \
        --eval_split "$EVAL_SPLIT" --epochs 25 --seed "$seed" \
        --run_id corrected_v2_formal_e0
done

for seed in "${SEEDS[@]}"; do
    run_one \
        "experiments/records/${DATASET}_lsci_strict_${EVAL_SPLIT}_corrected_v2_formal_stage1_strict_${EVAL_SPLIT}_seed${seed}_v2.json" \
        "logs/${DATASET}_lsci_stage1_corrected_v2_seed${seed}.log" \
        train_lsci.py --dataset "$DATASET" --data_root "$DATA_ROOT" \
        --eval_split "$EVAL_SPLIT" --epochs 25 --seed "$seed" \
        --run_id corrected_v2_formal_stage1
done

for seed in "${SEEDS[@]}"; do
    run_one \
        "experiments/records/${DATASET}_lsci_strict_${EVAL_SPLIT}_e3_corrected_v2_formal_e3_strict_${EVAL_SPLIT}_e3_seed${seed}_v2.json" \
        "logs/${DATASET}_lsci_e3_corrected_v2_seed${seed}.log" \
        train_lsci.py --dataset "$DATASET" --data_root "$DATA_ROOT" \
        --eval_split "$EVAL_SPLIT" --epochs 25 --seed "$seed" \
        --run_id corrected_v2_formal_e3 --use_semantic_prior --lambda_sem 0.1
done

echo "MOVIELENS_CORRECTED_V2_COMPLETE"
