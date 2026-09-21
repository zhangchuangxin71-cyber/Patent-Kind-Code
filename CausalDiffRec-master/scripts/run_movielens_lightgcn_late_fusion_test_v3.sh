#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=/home/p520/anaconda3/envs/zpp1/bin/python
DATA_ROOT=data_strict/processed/movielens1m/v1_strict
VALIDATION_REPORT=experiments/reports/movielens_lightgcn_converged_retry_late_fusion_validation_v3.json
MASTER_LOG=logs/movielens_lightgcn_late_fusion_test_v3_master.log
SEEDS=(1024 2048 3072 4096 5120)

mkdir -p logs experiments/records experiments/reports

status() {
    printf '[%s] %s\n' "$(date '+%F %T')" "$*" >> "$MASTER_LOG"
}

if ! ALPHA=$("$PYTHON_BIN" - "$VALIDATION_REPORT" \
    2> logs/movielens_lightgcn_ood_gate_v3.log <<'PY'
import json
import sys

d = json.load(open(sys.argv[1], encoding="utf-8"))
s = d["summary"]
a = str(s["selected_alpha"])
base = float(s["confirmation_baseline_ndcg20"])
real = float(s["confirmation_real_ndcg20"])
relative_gain = real / base - 1.0
checks = {
    "validation_only": s.get("test_gt_loaded") is False,
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
    status "validation gate failed; strong-LightGCN OOD comparison not run"
    exit 1
fi

status "validation gate passed; frozen alpha=$ALPHA; starting paired OOD comparison"
for seed in "${SEEDS[@]}"; do
    if [[ "$seed" == 4096 ]]; then
        source="experiments/records/movielens1m_lightgcn_converged_retry_strict_ood_seed4096_v3.json"
    else
        source="experiments/records/movielens1m_lightgcn_converged_strict_ood_seed${seed}_v3.json"
    fi
    out="experiments/records/movielens1m_lightgcn_latefusion_strict_ood_seed${seed}_v3.json"
    log="logs/movielens1m_lightgcn_latefusion_test_v3_seed${seed}.log"
    if [[ -f "$out" ]]; then
        status "skip existing paired OOD record: $out"
        continue
    fi
    env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" \
        scripts/evaluate_late_fusion.py \
        --source_record "$source" --data_root "$DATA_ROOT" \
        --eval_split ood --alpha "$ALPHA" --out "$out" > "$log" 2>&1
    status "paired OOD comparison done: seed=$seed"
done

"$PYTHON_BIN" scripts/summarize_late_fusion_test.py \
    --records experiments/records/movielens1m_lightgcn_latefusion_strict_ood_seed*_v3.json \
    --validation_report "$VALIDATION_REPORT" \
    --out experiments/reports/movielens_lightgcn_late_fusion_test_v3.json \
    > logs/movielens_lightgcn_late_fusion_test_summary_v3.log 2>&1
status "MOVIELENS_LIGHTGCN_LATE_FUSION_OOD_TEST_V3_COMPLETE"
