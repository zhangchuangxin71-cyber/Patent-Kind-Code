#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=/home/p520/anaconda3/envs/zpp1/bin/python
DATA_ROOT=data_strict/processed/movielens1m/v1_strict
MASTER_LOG=logs/movielens_lightgcn_retry4096_validation_v3_master.log
PRIOR_RECORD=experiments/records/movielens1m_lightgcn_converged_strict_ood_seed4096_v3.json
PRIOR_CHECKPOINT=checkpoints/movielens1m_lightgcn_converged_strict_ood_seed4096_v3.pt
RETRY_RECORD=experiments/records/movielens1m_lightgcn_converged_retry_strict_ood_seed4096_v3.json
RETRY_CHECKPOINT=checkpoints/movielens1m_lightgcn_converged_retry_strict_ood_seed4096_v3.pt

status() {
    printf '[%s] %s\n' "$(date '+%F %T')" "$*" >> "$MASTER_LOG"
}

status "MovieLens seed4096 premature-stop correction started"
if [[ ! -f "$RETRY_RECORD" ]]; then
    status "retry seed4096 from best epoch 5; enforce min_epochs=50"
    env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" \
        scripts/train_lightgcn_strict.py \
        --dataset movielens1m --data_root "$DATA_ROOT" --seed 4096 \
        --epochs 200 --min_epochs 50 --patience 20 \
        --resume_checkpoint "$PRIOR_CHECKPOINT" --resume_record "$PRIOR_RECORD" \
        --out "$RETRY_RECORD" --checkpoint "$RETRY_CHECKPOINT" \
        > logs/movielens1m_lightgcn_converged_retry_v3_seed4096.log 2>&1
fi
status "retry seed4096 complete; corrected five-seed semantic sweep start"

env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" \
    scripts/validate_late_fusion.py \
    --dataset movielens1m --data_root "$DATA_ROOT" \
    --source_records \
        experiments/records/movielens1m_lightgcn_converged_strict_ood_seed1024_v3.json \
        experiments/records/movielens1m_lightgcn_converged_strict_ood_seed2048_v3.json \
        experiments/records/movielens1m_lightgcn_converged_strict_ood_seed3072_v3.json \
        "$RETRY_RECORD" \
        experiments/records/movielens1m_lightgcn_converged_strict_ood_seed5120_v3.json \
    --alphas 0,0.025,0.05,0.075,0.1,0.15,0.2,0.25,0.5,0.75,1,1.25,1.5,2 \
    --out experiments/reports/movielens_lightgcn_converged_retry_late_fusion_validation_v3.json \
    > logs/movielens_lightgcn_converged_retry_late_fusion_validation_sweep_v3.log 2>&1
status "MOVIELENS_LIGHTGCN_RETRY4096_VALIDATION_V3_COMPLETE"
