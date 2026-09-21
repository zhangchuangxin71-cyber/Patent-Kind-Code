#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=/home/p520/anaconda3/envs/zpp1/bin/python
DATASET=movielens1m
DATA_ROOT=data_strict/processed/movielens1m/v1_strict
SHUFFLED_PRIOR=${DATA_ROOT}/features/semantic_prior_shuffled.pt
SEEDS=(1024 2048 3072)
MIN_FREE_MIB=${MIN_FREE_MIB:-10000}
POLL_SECONDS=${POLL_SECONDS:-60}
MASTER_LOG=logs/movielens_semantic_missing_controls_v3_master.log

mkdir -p logs experiments/records checkpoints

log_status() {
    printf '[%s] %s\n' "$(date '+%F %T')" "$*" >> "$MASTER_LOG"
}

wait_for_gpu() {
    local consecutive=0
    local free_mib
    while true; do
        free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
            | head -n 1 | tr -d ' ')
        if [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= MIN_FREE_MIB )); then
            consecutive=$((consecutive + 1))
            log_status "GPU free=${free_mib} MiB, readiness=${consecutive}/2"
            if (( consecutive >= 2 )); then
                return
            fi
        else
            consecutive=0
            log_status "waiting for GPU: free=${free_mib:-unknown} MiB, required=${MIN_FREE_MIB} MiB"
        fi
        sleep "$POLL_SECONDS"
    done
}

run_one() {
    local record=$1
    local log=$2
    shift 2
    if [[ -f "$record" ]]; then
        log_status "skip existing record: $record"
        return
    fi
    wait_for_gpu
    log_status "start: $record"
    env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" train_lsci.py \
        --dataset "$DATASET" --data_root "$DATA_ROOT" --eval_split ood \
        --epochs 25 --validation_only "$@" > "$log" 2>&1
    if [[ ! -f "$record" ]]; then
        log_status "error: command exited without producing $record"
        return 1
    fi
    log_status "done: $record"
}

log_status "monitor started: min_free=${MIN_FREE_MIB} MiB, poll=${POLL_SECONDS}s"

# Matched control A: fixed 50% semantic conditioning without InfoNCE.
for seed in "${SEEDS[@]}"; do
    run_one \
        "experiments/records/${DATASET}_lsci_strict_ood_e4_forcegate050_lamsem0_corrected_v3_diag_gate050_nonse_strict_ood_e4_forcegate0.5_lamsem0_seed${seed}_v3.json" \
        "logs/${DATASET}_diag_gate050_nonse_seed${seed}.log" \
        --seed "$seed" --run_id corrected_v3_diag_gate050_nonse \
        --use_semantic_prior --lambda_sem 0 --force_sem_gate 0.5
done

# Matched control B: shuffled semantics without InfoNCE, using the learned gate.
for seed in "${SEEDS[@]}"; do
    run_one \
        "experiments/records/${DATASET}_lsci_strict_ood_e4_shuffle_corrected_v3_diag_shuffle_nonse_strict_ood_e4_shuffle_seed${seed}_v3.json" \
        "logs/${DATASET}_diag_shuffle_nonse_seed${seed}.log" \
        --seed "$seed" --run_id corrected_v3_diag_shuffle_nonse \
        --use_semantic_prior --semantic_prior_path "$SHUFFLED_PRIOR" --lambda_sem 0
done

log_status "MOVIELENS_SEMANTIC_MISSING_CONTROLS_V3_COMPLETE"
