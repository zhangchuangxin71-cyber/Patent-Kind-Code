#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=/home/p520/anaconda3/envs/zpp1/bin/python
DATA_ROOT=data_strict/processed/movielens1m/v1_strict
SEEDS=(1024 2048 3072 4096 5120)
MIN_FREE_MIB=${MIN_FREE_MIB:-20000}
POLL_SECONDS=${POLL_SECONDS:-60}
MASTER_LOG=logs/movielens_stage1init_lightgcn200_validation_v3_master.log
PREREQUISITE=experiments/reports/movielens_lightgcn_converged_retry_late_fusion_validation_v3.json

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

status "MovieLens Stage1-init continuous-LightGCN fairness monitor started"
while [[ ! -f "$PREREQUISITE" ]]; do
    status "waiting for seed4096 correction report before fairness experiment"
    sleep "$POLL_SECONDS"
done

for seed in "${SEEDS[@]}"; do
    source_record="experiments/records/movielens1m_lsci_strict_ood_corrected_v3_formal_stage1_strict_ood_seed${seed}_v3.json"
    source_checkpoint=$("$PYTHON_BIN" -c \
        'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["checkpoint"])' \
        "$source_record")
    record="experiments/records/movielens1m_stage1init_lightgcn200_strict_ood_seed${seed}_v3.json"
    checkpoint="checkpoints/movielens1m_stage1init_lightgcn200_strict_ood_seed${seed}_v3.pt"
    log="logs/movielens1m_stage1init_lightgcn200_v3_seed${seed}.log"
    if [[ -f "$record" ]]; then
        status "skip existing record: $record"
        continue
    fi
    wait_for_gpu
    status "Stage1-init continuous LightGCN start: seed=$seed"
    env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" \
        scripts/train_lightgcn_strict.py \
        --dataset movielens1m --data_root "$DATA_ROOT" --seed "$seed" \
        --epochs 200 --patience 0 \
        --init_checkpoint "$source_checkpoint" --init_record "$source_record" \
        --out "$record" --checkpoint "$checkpoint" > "$log" 2>&1
    status "Stage1-init continuous LightGCN done: seed=$seed"
done
status "MOVIELENS_STAGE1INIT_LIGHTGCN200_VALIDATION_V3_COMPLETE"

wait_for_gpu
status "Stage1-init LightGCN semantic fine-grid validation sweep start"
env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" \
    scripts/validate_late_fusion.py \
    --dataset movielens1m --data_root "$DATA_ROOT" \
    --source_records experiments/records/movielens1m_stage1init_lightgcn200_strict_ood_seed*_v3.json \
    --alphas 0,0.025,0.05,0.075,0.1,0.15,0.2,0.25,0.5,0.75,1,1.25,1.5,2 \
    --out experiments/reports/movielens_stage1init_lightgcn200_late_fusion_validation_v3.json \
    > logs/movielens_stage1init_lightgcn200_late_fusion_validation_sweep_v3.log 2>&1
status "MOVIELENS_STAGE1INIT_LIGHTGCN200_LATE_FUSION_VALIDATION_V3_COMPLETE"
