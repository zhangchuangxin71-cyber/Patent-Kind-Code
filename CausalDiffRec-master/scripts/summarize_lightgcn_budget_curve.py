#!/usr/bin/env python
"""Summarize best-so-far validation metrics across BPR budgets."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--records", nargs="+", required=True)
    ap.add_argument("--budgets", default="25,50,100,200,300,400")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    budgets = [int(x) for x in args.budgets.split(",")]
    per_seed = {}
    for path in args.records:
        record = json.loads(Path(path).read_text(encoding="utf-8"))
        seed = int(record["settings"]["seed"])
        points = {
            int(row["epoch"]): float(row["val_metrics"]["Top20"]["NDCG"])
            for row in record["epoch_logs"]
        }
        per_seed[str(seed)] = {
            str(budget): max(value for epoch, value in points.items() if epoch <= budget)
            for budget in budgets
        }
    if len(per_seed) != 5:
        raise RuntimeError(f"expected five seeds, got {sorted(per_seed)}")
    aggregate = {}
    for budget in budgets:
        values = [row[str(budget)] for row in per_seed.values()]
        aggregate[str(budget)] = {
            "mean": statistics.mean(values),
            "sample_std": statistics.stdev(values),
            "values": values,
        }
    output = {
        "dataset": args.dataset,
        "protocol": "corrected_v4_lightgcn_budget_validation_only",
        "test_gt_loaded": False,
        "budgets_are_complete_bpr_passes": True,
        "aggregate": aggregate,
        "per_seed": per_seed,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
