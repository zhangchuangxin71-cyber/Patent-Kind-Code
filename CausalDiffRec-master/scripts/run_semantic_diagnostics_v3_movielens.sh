#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=/home/p520/anaconda3/envs/zpp1/bin/python
DATASET=movielens1m
DATA_ROOT=data_strict/processed/movielens1m/v1_strict
SHUFFLED_PRIOR=${DATA_ROOT}/features/semantic_prior_shuffled.pt
SEEDS=(1024 2048 3072)

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
    env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" train_lsci.py \
        --dataset "$DATASET" --data_root "$DATA_ROOT" --eval_split ood \
        --epochs 25 --validation_only "$@" > "$log" 2>&1
    if [[ ! -f "$record" ]]; then
        echo "[error] command exited without producing $record" >&2
        return 1
    fi
    echo "[done] $record"
}

for seed in "${SEEDS[@]}"; do
    run_one \
        "experiments/records/${DATASET}_lsci_strict_ood_e4_lamsem0_corrected_v3_diag_condition_only_strict_ood_e4_lamsem0_seed${seed}_v3.json" \
        "logs/${DATASET}_diag_condition_only_seed${seed}.log" \
        --seed "$seed" --run_id corrected_v3_diag_condition_only \
        --use_semantic_prior --lambda_sem 0
done

for seed in "${SEEDS[@]}"; do
    run_one \
        "experiments/records/${DATASET}_lsci_strict_ood_e4_shuffle_corrected_v3_diag_shuffle_strict_ood_e4_shuffle_seed${seed}_v3.json" \
        "logs/${DATASET}_diag_shuffle_seed${seed}.log" \
        --seed "$seed" --run_id corrected_v3_diag_shuffle \
        --use_semantic_prior --semantic_prior_path "$SHUFFLED_PRIOR" --lambda_sem 0.1
done

for seed in "${SEEDS[@]}"; do
    run_one \
        "experiments/records/${DATASET}_lsci_strict_ood_e4_forcegate100_lamsem0p1_corrected_v3_diag_gate1_strict_ood_e4_forcegate1_lamsem0.1_seed${seed}_v3.json" \
        "logs/${DATASET}_diag_gate100_seed${seed}.log" \
        --seed "$seed" --run_id corrected_v3_diag_gate1 \
        --use_semantic_prior --lambda_sem 0.1 --force_sem_gate 1.0
done

for gate in 0.9 0.7 0.5; do
    case "$gate" in
        0.9) gtag=090 ;;
        0.7) gtag=070 ;;
        0.5) gtag=050 ;;
    esac
    for seed in "${SEEDS[@]}"; do
        run_one \
            "experiments/records/${DATASET}_lsci_strict_ood_e4_forcegate${gtag}_lamsem0p1_corrected_v3_diag_gate${gtag}_strict_ood_e4_forcegate${gate}_lamsem0.1_seed${seed}_v3.json" \
            "logs/${DATASET}_diag_gate${gtag}_seed${seed}.log" \
            --seed "$seed" --run_id "corrected_v3_diag_gate${gtag}" \
            --use_semantic_prior --lambda_sem 0.1 --force_sem_gate "$gate"
    done
done

echo "MOVIELENS_SEMANTIC_DIAGNOSTICS_V3_COMPLETE"
