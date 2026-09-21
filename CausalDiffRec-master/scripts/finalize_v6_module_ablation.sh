#!/usr/bin/env bash
set -euo pipefail

# Background finalizer: waits for both 15-run training suites, fails closed,
# evaluates, packages compact checkpoints, commits and pushes public artifacts.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/p520/anaconda3/envs/zpp1/bin/python}"
MAX_WAIT_MINUTES="${MAX_WAIT_MINUTES:-480}"
POLL_SECONDS="${POLL_SECONDS:-60}"

cd "$ROOT"
mkdir -p logs experiments/reports artifacts

count_records() {
  local dataset="$1"
  local count=0 pattern
  for pattern in \
    "${dataset}_lsci_strict_ood_v6_module_ablation_a0_*_v4.json" \
    "${dataset}_lsci_strict_ood_e4_lamsem0_v6_module_ablation_a1_*_v4.json" \
    "${dataset}_lsci_strict_ood_e3_v6_module_ablation_a2_*_v4.json"; do
    count=$((count + $(find experiments/records -maxdepth 1 -type f -name "$pattern" | wc -l)))
  done
  echo "$count"
}

waited=0
while true; do
  yelp_count="$(count_records yelp2018)"
  movie_count="$(count_records movielens1m)"
  echo "[$(date '+%F %T')] waiting: yelp=$yelp_count/15 movielens=$movie_count/15"
  if [[ "$yelp_count" -eq 15 && "$movie_count" -eq 15 ]]; then
    break
  fi
  if ! systemctl --user is-active --quiet patent-yelp-v6-ablation.service && [[ "$yelp_count" -lt 15 ]]; then
    echo "Yelp training service stopped before producing 15 records" >&2
    exit 3
  fi
  if ! systemctl --user is-active --quiet patent-movielens-v6-ablation.service && [[ "$movie_count" -lt 15 ]]; then
    echo "MovieLens training service stopped before producing 15 records" >&2
    exit 3
  fi
  if (( waited >= MAX_WAIT_MINUTES * 60 )); then
    echo "Timed out waiting for training records" >&2
    exit 4
  fi
  sleep "$POLL_SECONDS"
  waited=$((waited + POLL_SECONDS))
done

echo "[$(date '+%F %T')] training records complete; starting frozen evaluation"
bash scripts/run_module_ablation_evaluation.sh yelp2018 0
bash scripts/run_module_ablation_evaluation.sh movielens1m 0

"$PYTHON_BIN" scripts/export_compact_evaluation_checkpoints.py \
  --root "$ROOT" --datasets yelp2018,movielens1m --out_dir artifacts/v6_evaluation
"$PYTHON_BIN" scripts/audit_strict_protocol.py \
  --out experiments/reports/strict_data_leakage_audit.json \
  --markdown experiments/reports/strict_data_leakage_audit.md
"$PYTHON_BIN" scripts/collect_reproducibility_evidence.py \
  --root "$ROOT" --datasets yelp2018,movielens1m,food,amazon_beauty \
  --out experiments/reports/reproducibility_evidence.json \
  --markdown experiments/reports/reproducibility_evidence.md
"$PYTHON_BIN" scripts/build_v6_module_ablation_report.py --root "$ROOT"
"$PYTHON_BIN" -m unittest discover -s tests -p 'test_*.py' -v \
  > logs/v6_final_test.log 2>&1

git add experiments/records/*_v6_module_ablation_*.json
git add experiments/reports/*_v6_module_ablation_*.json experiments/reports/*_v6_module_ablation_*.md
git add experiments/reports/strict_data_leakage_audit.json experiments/reports/strict_data_leakage_audit.md
git add experiments/reports/reproducibility_evidence.json experiments/reports/reproducibility_evidence.md
git add artifacts/v6_evaluation

if ! git diff --cached --quiet; then
  SKIP_GITHUB_AUTO_PUSH=1 git commit -m "Add corrected-v6 module ablation results"
fi
git push origin main
echo "[$(date '+%F %T')] V6_MODULE_ABLATION_FINALIZED_AND_PUSHED"
