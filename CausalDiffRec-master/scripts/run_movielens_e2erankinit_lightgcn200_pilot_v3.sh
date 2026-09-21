#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=/home/p520/anaconda3/envs/zpp1/bin/python
DATA_ROOT=data_strict/processed/movielens1m/v1_strict
SOURCE_RECORD=experiments/records/movielens1m_lsci_strict_ood_corrected_v3_e2erank_lam01_pilot_strict_ood_seed1024_v3.json
OUT_RECORD=experiments/records/movielens1m_e2erankinit_lightgcn200_strict_ood_seed1024_v3.json
OUT_CHECKPOINT=checkpoints/movielens1m_e2erankinit_lightgcn200_strict_ood_seed1024_v3.pt
MASTER_LOG=logs/movielens_e2erankinit_lightgcn200_pilot_v3_master.log
MIN_FREE_MIB=${MIN_FREE_MIB:-20000}
POLL_SECONDS=${POLL_SECONDS:-60}

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

status "MovieLens end2end-rank initialization downstream pilot started"
if [[ ! -f "$OUT_RECORD" ]]; then
    wait_for_gpu
    source_checkpoint=$("$PYTHON_BIN" -c \
        'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["checkpoint"])' \
        "$SOURCE_RECORD")
    status "continuous LightGCN 200 epochs start from end2end-rank checkpoint"
    env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" \
        scripts/train_lightgcn_strict.py \
        --dataset movielens1m --data_root "$DATA_ROOT" --seed 1024 \
        --epochs 200 --patience 0 \
        --init_checkpoint "$source_checkpoint" --init_record "$SOURCE_RECORD" \
        --out "$OUT_RECORD" --checkpoint "$OUT_CHECKPOINT" \
        > logs/movielens1m_e2erankinit_lightgcn200_v3_seed1024.log 2>&1
fi
status "MOVIELENS_E2ERANKINIT_LIGHTGCN200_PILOT_V3_COMPLETE"
