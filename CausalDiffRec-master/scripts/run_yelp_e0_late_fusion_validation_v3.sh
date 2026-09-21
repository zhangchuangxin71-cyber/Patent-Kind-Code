#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=/home/p520/anaconda3/envs/zpp1/bin/python
DATASET=yelp2018
DATA_ROOT=data_strict/processed/yelp2018/v1_strict
SEEDS=(1024 2048 3072 4096 5120)
MIN_FREE_MIB=${MIN_FREE_MIB:-20000}
POLL_SECONDS=${POLL_SECONDS:-60}
MASTER_LOG=logs/yelp_e0_late_fusion_validation_v3_master.log

mkdir -p logs experiments/records experiments/reports checkpoints

status() {
    printf '[%s] %s\n' "$(date '+%F %T')" "$*" >> "$MASTER_LOG"
}

wait_for_gpu() {
    local consecutive=0 free_mib
    while true; do
        free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
            | head -n 1 | tr -d ' ')
        if [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= MIN_FREE_MIB )); then
            consecutive=$((consecutive + 1))
            status "GPU free=${free_mib} MiB, readiness=${consecutive}/2"
            if (( consecutive >= 2 )); then return; fi
        else
            consecutive=0
            status "waiting for GPU: free=${free_mib:-unknown} MiB, required=${MIN_FREE_MIB} MiB"
        fi
        sleep "$POLL_SECONDS"
    done
}

status "Yelp corrected-v3 E0 attribution validation monitor started"
for seed in "${SEEDS[@]}"; do
    record="experiments/records/yelp2018_strict_ood_corrected_v3_yelp_e0_valonly_strict_ood_seed${seed}_v3.json"
    log="logs/yelp2018_e0_valonly_v3_seed${seed}.log"
    if [[ -f "$record" ]]; then
        status "skip existing record: $record"
        continue
    fi
    wait_for_gpu
    status "E0 validation-only start: seed=$seed"
    env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" train.py \
        --dataset "$DATASET" --data_root "$DATA_ROOT" --eval_split ood \
        --epochs 25 --seed "$seed" --validation_only \
        --run_id corrected_v3_yelp_e0_valonly > "$log" 2>&1
    if [[ ! -f "$record" ]]; then
        status "error: no E0 record produced for seed=$seed"
        exit 1
    fi
    status "E0 validation-only done: seed=$seed"
done
status "YELP_E0_VALIDATION_V3_COMPLETE"

wait_for_gpu
status "E0 real/shuffle late-fusion validation sweep start"
env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" \
    scripts/validate_late_fusion.py \
    --dataset "$DATASET" --data_root "$DATA_ROOT" \
    --source_records experiments/records/yelp2018_strict_ood_corrected_v3_yelp_e0_valonly_strict_ood_seed*_v3.json \
    --out experiments/reports/yelp_e0_late_fusion_validation_v3.json \
    > logs/yelp_e0_late_fusion_validation_sweep_v3.log 2>&1
status "YELP_E0_LATE_FUSION_VALIDATION_V3_COMPLETE"
