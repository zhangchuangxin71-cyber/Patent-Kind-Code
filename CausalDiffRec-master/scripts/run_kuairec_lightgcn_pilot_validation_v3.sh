#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=/home/p520/anaconda3/envs/zpp1/bin/python
DATASET=kuairec
DATA_ROOT=data_strict/processed/kuairec/v1_strict
SEED=1024
MIN_FREE_MIB=${MIN_FREE_MIB:-20000}
POLL_SECONDS=${POLL_SECONDS:-15}
MASTER_LOG=logs/kuairec_lightgcn_pilot_validation_v3_master.log
RECORD=experiments/records/kuairec_lightgcn_pilot_strict_ood_seed1024_v3.json
CHECKPOINT=checkpoints/kuairec_lightgcn_pilot_strict_ood_seed1024_v3.pt
RUN_LOG=logs/kuairec_lightgcn_pilot_v3_seed1024.log

mkdir -p logs experiments/records checkpoints

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

if [[ -f "$RECORD" ]]; then
    status "skip existing completed record: $RECORD"
    exit 0
fi

status "KuaiRec pure-LightGCN validation-only pilot monitor started"
wait_for_gpu
status "KuaiRec LightGCN start: seed=$SEED epochs=200 patience=disabled"
env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" \
    scripts/train_lightgcn_strict.py \
    --dataset "$DATASET" --data_root "$DATA_ROOT" --seed "$SEED" \
    --epochs 200 --min_epochs 200 --patience 0 \
    --embedding_dim 8 --layers 3 --batch_size 1024 --lr 0.001 --l2 0.001 \
    --out "$RECORD" --checkpoint "$CHECKPOINT" > "$RUN_LOG" 2>&1
status "KUAIREC_LIGHTGCN_PILOT_VALIDATION_V3_COMPLETE"
