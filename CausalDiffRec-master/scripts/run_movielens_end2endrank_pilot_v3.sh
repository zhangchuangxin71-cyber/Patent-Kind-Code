#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=/home/p520/anaconda3/envs/zpp1/bin/python
DATA_ROOT=data_strict/processed/movielens1m/v1_strict
MIN_FREE_MIB=${MIN_FREE_MIB:-20000}
POLL_SECONDS=${POLL_SECONDS:-60}
MASTER_LOG=logs/movielens_end2endrank_pilot_v3_master.log
SMOKE_RECORD=experiments/records/movielens1m_lsci_strict_ood_corrected_v3_e2erank_smoke_strict_ood_seed1024_v3.json
PILOT_RECORD=experiments/records/movielens1m_lsci_strict_ood_corrected_v3_e2erank_lam01_pilot_strict_ood_seed1024_v3.json

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

status "MovieLens upstream-ranking pilot monitor started"
wait_for_gpu
if [[ ! -f "$SMOKE_RECORD" ]]; then
    status "one-epoch gradient smoke start"
    env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" train_lsci.py \
        --dataset movielens1m --data_root "$DATA_ROOT" --eval_split ood \
        --epochs 1 --seed 1024 --validation_only \
        --lambda_rank_upstream 0.1 --rank_batch_size 2048 --rank_layers 3 \
        --run_id corrected_v3_e2erank_smoke \
        > logs/movielens1m_end2endrank_smoke_v3_seed1024.log 2>&1
fi

"$PYTHON_BIN" -c '
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
diag = d["epoch_logs"][0]["upstream_rank_grad_diagnostics"]
if not diag or not all(float(value) > 0 for value in diag.values()):
    raise SystemExit("gradient smoke failed: " + repr(diag))
print("gradient smoke passed", diag)
' "$SMOKE_RECORD" >> "$MASTER_LOG" 2>&1
status "gradient smoke passed"

if [[ ! -f "$PILOT_RECORD" ]]; then
    status "25-epoch lambda_rank_upstream=0.1 pilot start"
    env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" train_lsci.py \
        --dataset movielens1m --data_root "$DATA_ROOT" --eval_split ood \
        --epochs 25 --seed 1024 --validation_only \
        --lambda_rank_upstream 0.1 --rank_batch_size 2048 --rank_layers 3 \
        --run_id corrected_v3_e2erank_lam01_pilot \
        > logs/movielens1m_end2endrank_lam01_pilot_v3_seed1024.log 2>&1
fi
status "MOVIELENS_END2ENDRANK_LAM01_PILOT_V3_COMPLETE"
