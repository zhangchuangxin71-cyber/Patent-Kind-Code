#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=/home/p520/anaconda3/envs/zpp1/bin/python
DATA_ROOT=data_strict/processed/yelp2018/v1_strict
VALIDATION_REPORT=experiments/reports/yelp_late_fusion_validation_v3.json
MASTER_LOG=logs/yelp_late_fusion_test_v3_master.log
SEEDS=(1024 2048 3072 4096 5120)

status() {
    printf '[%s] %s\n' "$(date '+%F %T')" "$*" >> "$MASTER_LOG"
}

# This gate is fixed before opening OOD test GT.  It requires a positive result
# in both validation groups, superiority to shuffled semantics, and improvement
# for every seed in the independent two-seed confirmation group.
ALPHA=$($PYTHON_BIN - "$VALIDATION_REPORT" <<'PY'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
s = d["summary"]
a = str(s["selected_alpha"])
checks = {
    "positive_alpha": s["selected_alpha"] > 0,
    "selection_beats_baseline": s["selection_real_ndcg20"] > s["selection_baseline_ndcg20"],
    "confirmation_beats_baseline": s["confirmation_real_ndcg20"] > s["confirmation_baseline_ndcg20"],
    "confirmation_beats_shuffle": s["confirmation_real_ndcg20"] > s["confirmation_shuffle_ndcg20"],
    "each_confirmation_seed_improves": all(
        d["per_seed"][str(seed)]["real"][a]
        > d["per_seed"][str(seed)]["real"]["0.0"]
        for seed in s["confirmation_seeds"]
    ),
}
if not all(checks.values()):
    raise SystemExit("OOD test gate failed: " + json.dumps(checks, sort_keys=True))
print(s["selected_alpha"])
PY
)

status "validation gate passed; frozen alpha=$ALPHA; starting one-shot OOD comparison"
for seed in "${SEEDS[@]}"; do
    source="experiments/records/yelp2018_lsci_strict_ood_corrected_v3_yelp_stage1_valonly_strict_ood_seed${seed}_v3.json"
    out="experiments/records/yelp2018_latefusion_alpha100_strict_ood_seed${seed}_v3.json"
    log="logs/yelp2018_latefusion_test_v3_seed${seed}.log"
    if [[ -f "$out" ]]; then
        status "skip existing test record: $out"
        continue
    fi
    status "one-shot test start: seed=$seed"
    env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" \
        scripts/evaluate_late_fusion.py \
        --source_record "$source" --data_root "$DATA_ROOT" \
        --eval_split ood --alpha "$ALPHA" --out "$out" > "$log" 2>&1
    status "one-shot test done: seed=$seed"
done

"$PYTHON_BIN" scripts/summarize_late_fusion_test.py \
    --records experiments/records/yelp2018_latefusion_alpha100_strict_ood_seed*_v3.json \
    --validation_report "$VALIDATION_REPORT" \
    --out experiments/reports/yelp_late_fusion_test_v3.json \
    > logs/yelp_late_fusion_test_summary_v3.log 2>&1
status "YELP_LATE_FUSION_ONE_SHOT_OOD_TEST_V3_COMPLETE"
