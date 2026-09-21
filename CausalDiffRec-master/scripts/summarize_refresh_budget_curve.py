#!/usr/bin/env python
"""Summarize best-so-far validation NDCG across equal BPR pass budgets."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def stats(values):
    return {
        "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "values": values,
    }


def curve(paths, budgets, passes_per_log):
    records = [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]
    by_seed = {}
    for record in records:
        seed = int(record["settings"]["seed"])
        points = [
            (
                int(entry["epoch"]) * passes_per_log,
                float(entry["val_metrics"]["Top20"]["NDCG"]),
            )
            for entry in record["epoch_logs"]
        ]
        by_seed[seed] = {
            str(budget): max(value for used, value in points if used <= budget)
            for budget in budgets
        }
    return {
        "per_seed": {str(k): v for k, v in sorted(by_seed.items())},
        "aggregate": {
            str(budget): stats([row[str(budget)] for row in by_seed.values()])
            for budget in budgets
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pure_records", nargs="+", required=True)
    ap.add_argument("--stage1_once_records", nargs="+", required=True)
    ap.add_argument("--stage1_refresh_records", nargs="+", required=True)
    ap.add_argument("--random_refresh_records", nargs="+", required=True)
    ap.add_argument("--budgets", default="25,50,100,200")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    budgets = [int(x) for x in args.budgets.split(",")]
    output = {
        "dataset": "movielens1m",
        "protocol": "corrected_v4_equal_bpr_refresh_mechanism_validation_only",
        "test_gt_loaded": False,
        "budgets_are_complete_bpr_passes": True,
        "arms": {
            "pure_lightgcn": curve(args.pure_records, budgets, 1),
            "stage1_initialize_once": curve(args.stage1_once_records, budgets, 8),
            "stage1_refresh_025": curve(args.stage1_refresh_records, budgets, 8),
            "matched_random_refresh_025": curve(args.random_refresh_records, budgets, 8),
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v["aggregate"] for k, v in output["arms"].items()}, indent=2))
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
