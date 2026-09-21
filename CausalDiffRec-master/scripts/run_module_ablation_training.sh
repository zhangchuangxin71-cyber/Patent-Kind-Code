#!/usr/bin/env bash
set -euo pipefail

# Strict validation-only A0/A1/A2 training. A3 is evaluation-only and must
# reuse A2 checkpoints. Usage: bash scripts/run_module_ablation_training.sh DATASET [GPU]
DATASET="${1:?dataset required}"
GPU="${2:-0}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/p520/anaconda3/envs/zpp1/bin/python}"
DATA_ROOT="$ROOT/data_strict/processed/$DATASET/v1_strict"
PRIOR="$DATA_ROOT/features/semantic_prior.pt"
SEEDS=(1024 2048 3072 4096 5120)

cd "$ROOT"
mkdir -p logs experiments/records checkpoints
for required in "$DATA_ROOT/READY" "$PRIOR"; do
  [[ -f "$required" ]] || { echo "missing $required" >&2; exit 2; }
done

record_for() {
  local arm="$1" seed="$2"
  case "$arm" in
    a0) printf 'experiments/records/%s_lsci_strict_ood_v6_module_ablation_a0_seed%s_strict_ood_seed%s_v4.json' "$DATASET" "$seed" "$seed" ;;
    a1) printf 'experiments/records/%s_lsci_strict_ood_e4_lamsem0_v6_module_ablation_a1_seed%s_strict_ood_e4_lamsem0_seed%s_v4.json' "$DATASET" "$seed" "$seed" ;;
    a2) printf 'experiments/records/%s_lsci_strict_ood_e3_v6_module_ablation_a2_seed%s_strict_ood_e3_seed%s_v4.json' "$DATASET" "$seed" "$seed" ;;
    *) return 2 ;;
  esac
}

run_one() {
  local arm="$1" seed="$2" record log
  shift 2
  record="$(record_for "$arm" "$seed")"
  log="logs/${DATASET}_v6_module_ablation_${arm}_seed${seed}.log"
  if [[ -f "$record" ]]; then
    echo "[$(date '+%F %T')] skip existing $record"
    return 0
  fi
  echo "[$(date '+%F %T')] start dataset=$DATASET arm=$arm seed=$seed"
  env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -u train_lsci.py \
    --dataset "$DATASET" --data_root "$DATA_ROOT" --eval_split ood \
    --epochs 25 --seed "$seed" --validation_only \
    --edge_gate_mode none --lambda_rank_upstream 0 \
    --rec_refresh 0.25 --rec_lr 0.001 --rec_batch_size 1024 \
    --rec_epochs_per_outer 1 --semantic_score_alpha 0 \
    --run_id "v6_module_ablation_${arm}_seed${seed}" "$@" > "$log" 2>&1
  [[ -f "$record" ]] || { echo "expected record missing: $record" >&2; return 3; }
  echo "[$(date '+%F %T')] complete dataset=$DATASET arm=$arm seed=$seed"
}

for arm in a0 a1 a2; do
  for seed in "${SEEDS[@]}"; do
    case "$arm" in
      a0) run_one "$arm" "$seed" ;;
      a1) run_one "$arm" "$seed" --use_semantic_prior --semantic_prior_path "$PRIOR" --lambda_sem 0 ;;
      a2) run_one "$arm" "$seed" --use_semantic_prior --semantic_prior_path "$PRIOR" --lambda_sem 0.1 ;;
    esac
  done
done
echo "${DATASET}_V6_MODULE_ABLATION_TRAINING_COMPLETE"
