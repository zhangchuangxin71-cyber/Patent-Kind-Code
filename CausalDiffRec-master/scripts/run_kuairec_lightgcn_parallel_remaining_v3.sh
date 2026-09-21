#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=/home/p520/anaconda3/envs/zpp1/bin/python
DATASET=kuairec
DATA_ROOT=data_strict/processed/kuairec/v1_strict
SEEDS=(3072 4096 5120)
MIN_FREE_MIB=${MIN_FREE_MIB:-14000}
SEQUENTIAL_CONTROLLER_PID=${SEQUENTIAL_CONTROLLER_PID:-}
MASTER_LOG=logs/kuairec_lightgcn_parallel_remaining_v3_master.log

mkdir -p logs experiments/records checkpoints

status() {
    printf '[%s] %s\n' "$(date '+%F %T')" "$*" >> "$MASTER_LOG"
}

resume_sequential_controller() {
    if [[ "$SEQUENTIAL_CONTROLLER_PID" =~ ^[0-9]+$ ]] \
        && kill -0 "$SEQUENTIAL_CONTROLLER_PID" 2>/dev/null; then
        kill -CONT "$SEQUENTIAL_CONTROLLER_PID"
        status "resumed sequential follow-up controller pid=$SEQUENTIAL_CONTROLLER_PID"
    fi
}
trap resume_sequential_controller EXIT

free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
    | head -n 1 | tr -d ' ')
if [[ ! "$free_mib" =~ ^[0-9]+$ ]] || (( free_mib < MIN_FREE_MIB )); then
    status "parallel launch aborted: free=${free_mib:-unknown} MiB required=$MIN_FREE_MIB MiB"
    exit 1
fi

status "parallel group start: free=${free_mib} MiB seeds=${SEEDS[*]}"
pids=()
for seed in "${SEEDS[@]}"; do
    record="experiments/records/kuairec_lightgcn_pilot_strict_ood_seed${seed}_v3.json"
    checkpoint="checkpoints/kuairec_lightgcn_pilot_strict_ood_seed${seed}_v3.pt"
    log="logs/kuairec_lightgcn_pilot_v3_seed${seed}.log"
    if [[ -f "$record" ]]; then
        status "skip existing completed record: $record"
        continue
    fi
    status "parallel validation-only start: seed=$seed"
    env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" \
        scripts/train_lightgcn_strict.py \
        --dataset "$DATASET" --data_root "$DATA_ROOT" --seed "$seed" \
        --epochs 200 --min_epochs 200 --patience 0 \
        --embedding_dim 8 --layers 3 --batch_size 1024 --lr 0.001 --l2 0.001 \
        --out "$record" --checkpoint "$checkpoint" > "$log" 2>&1 &
    pids+=("$!")
done

failures=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        status "parallel child failed: pid=$pid"
        failures=$((failures + 1))
    fi
done
if (( failures > 0 )); then
    status "parallel group finished with failures=$failures; sequential controller will retry missing records"
    exit 1
fi
status "KUAIREC_PARALLEL_REMAINING_V3_COMPLETE"
