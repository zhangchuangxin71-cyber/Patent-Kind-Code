#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=/home/p520/anaconda3/envs/zpp1/bin/python
DATASET=amazon_beauty
DATA_ROOT=data_strict/processed/amazon_beauty/v1_strict
SEEDS=(1024 2048 3072 4096 5120)
MIN_FREE_MIB=${MIN_FREE_MIB:-16000}
MASTER_LOG=logs/amazon_beauty_lightgcn_validation_v3_master.log
VALIDATION_REPORT=experiments/reports/amazon_beauty_lightgcn_late_fusion_validation_v3.json

mkdir -p logs experiments/records experiments/reports checkpoints

status() {
    printf '[%s] %s\n' "$(date '+%F %T')" "$*" >> "$MASTER_LOG"
}

free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
    | head -n 1 | tr -d ' ')
if [[ ! "$free_mib" =~ ^[0-9]+$ ]] || (( free_mib < MIN_FREE_MIB )); then
    status "parallel launch aborted: free=${free_mib:-unknown} MiB required=$MIN_FREE_MIB MiB"
    exit 1
fi

status "Amazon Beauty five-seed parallel LightGCN start: free=${free_mib} MiB"
pids=()
for seed in "${SEEDS[@]}"; do
    record="experiments/records/amazon_beauty_lightgcn_strict_ood_seed${seed}_v3.json"
    checkpoint="checkpoints/amazon_beauty_lightgcn_strict_ood_seed${seed}_v3.pt"
    log="logs/amazon_beauty_lightgcn_v3_seed${seed}.log"
    if [[ -f "$record" ]]; then
        status "skip existing completed record: $record"
        continue
    fi
    status "validation-only start: seed=$seed"
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
        status "parallel training child failed: pid=$pid"
        failures=$((failures + 1))
    fi
done
if (( failures > 0 )); then
    status "training stopped with failures=$failures; semantic validation not run"
    exit 1
fi
status "five-seed LightGCN complete; starting split-seed semantic validation"

env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" \
    scripts/validate_late_fusion.py \
    --dataset "$DATASET" --data_root "$DATA_ROOT" \
    --source_records \
        experiments/records/amazon_beauty_lightgcn_strict_ood_seed1024_v3.json \
        experiments/records/amazon_beauty_lightgcn_strict_ood_seed2048_v3.json \
        experiments/records/amazon_beauty_lightgcn_strict_ood_seed3072_v3.json \
        experiments/records/amazon_beauty_lightgcn_strict_ood_seed4096_v3.json \
        experiments/records/amazon_beauty_lightgcn_strict_ood_seed5120_v3.json \
    --selection_seeds 1024,2048,3072 --confirmation_seeds 4096,5120 \
    --alphas 0,0.025,0.05,0.075,0.1,0.15,0.2,0.25,0.5,0.75,1,1.25,1.5,2 \
    --out "$VALIDATION_REPORT" \
    > logs/amazon_beauty_lightgcn_late_fusion_validation_v3.log 2>&1
status "semantic validation complete; applying frozen OOD-test gate"

if ! ALPHA=$("$PYTHON_BIN" - "$VALIDATION_REPORT" \
    2> logs/amazon_beauty_lightgcn_ood_gate_v3.log <<'PY'
import json
import sys

d = json.load(open(sys.argv[1], encoding="utf-8"))
s = d["summary"]
a = str(s["selected_alpha"])
base = float(s["confirmation_baseline_ndcg20"])
real = float(s["confirmation_real_ndcg20"])
relative_gain = real / base - 1.0
checks = {
    "positive_alpha": s["selected_alpha"] > 0,
    "selection_beats_baseline": s["selection_real_ndcg20"] > s["selection_baseline_ndcg20"],
    "confirmation_beats_baseline": real > base,
    "confirmation_relative_gain_at_least_1pct": relative_gain >= 0.01,
    "confirmation_beats_shuffle": real > s["confirmation_shuffle_ndcg20"],
    "each_confirmation_seed_improves": all(
        d["per_seed"][str(seed)]["real"][a]
        > d["per_seed"][str(seed)]["real"]["0.0"]
        for seed in s["confirmation_seeds"]
    ),
}
if not all(checks.values()):
    raise SystemExit(json.dumps({"checks": checks, "relative_gain": relative_gain}, sort_keys=True))
print(s["selected_alpha"])
PY
); then
    status "semantic confirmation gate failed; OOD test remains unopened"
    exit 0
fi

status "semantic gate passed; frozen alpha=$ALPHA; starting one-shot OOD evaluation"
for seed in "${SEEDS[@]}"; do
    source="experiments/records/amazon_beauty_lightgcn_strict_ood_seed${seed}_v3.json"
    out="experiments/records/amazon_beauty_lightgcn_latefusion_strict_ood_seed${seed}_v3.json"
    log="logs/amazon_beauty_lightgcn_latefusion_test_v3_seed${seed}.log"
    if [[ -f "$out" ]]; then
        status "skip existing one-shot OOD record: $out"
        continue
    fi
    env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" \
        scripts/evaluate_late_fusion.py \
        --source_record "$source" --data_root "$DATA_ROOT" \
        --eval_split ood --alpha "$ALPHA" --out "$out" > "$log" 2>&1
    status "one-shot OOD evaluation done: seed=$seed"
done

"$PYTHON_BIN" scripts/summarize_late_fusion_test.py \
    --records experiments/records/amazon_beauty_lightgcn_latefusion_strict_ood_seed*_v3.json \
    --validation_report "$VALIDATION_REPORT" \
    --out experiments/reports/amazon_beauty_lightgcn_late_fusion_test_v3.json \
    > logs/amazon_beauty_lightgcn_late_fusion_test_summary_v3.log 2>&1
status "AMAZON_BEAUTY_LIGHTGCN_LATE_FUSION_PIPELINE_V3_COMPLETE"
