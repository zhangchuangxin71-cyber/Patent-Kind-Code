#!/usr/bin/env python
"""Summarize a frozen paired baseline/late-fusion test comparison."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def metric(record, field, metric_name):
    return float(record[field]["Top20"][metric_name])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", nargs="+", required=True)
    ap.add_argument("--validation_report", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    validation = json.load(open(args.validation_report, encoding="utf-8"))
    records = [json.load(open(path, encoding="utf-8")) for path in args.records]
    records.sort(key=lambda x: int(x["settings"]["seed"]))
    expected = [1024, 2048, 3072, 4096, 5120]
    seeds = [int(x["settings"]["seed"]) for x in records]
    if seeds != expected:
        raise RuntimeError(f"Expected seeds {expected}, got {seeds}")

    per_seed = {}
    aggregate = {}
    for name in ("NDCG", "Recall", "Precision", "Hit Ratio"):
        baseline = np.asarray([metric(x, "baseline_metrics", name) for x in records])
        fusion = np.asarray([metric(x, "best_metrics", name) for x in records])
        delta = fusion - baseline
        aggregate[name] = {
            "baseline_mean": float(baseline.mean()),
            "baseline_sample_std": float(baseline.std(ddof=1)),
            "fusion_mean": float(fusion.mean()),
            "fusion_sample_std": float(fusion.std(ddof=1)),
            "absolute_delta_mean": float(delta.mean()),
            "relative_delta_percent": float(100.0 * delta.mean() / baseline.mean()),
            "improved_seeds": int((delta > 0).sum()),
        }
        if len(delta) > 1 and delta.std(ddof=1) > 0:
            se = delta.std(ddof=1) / math.sqrt(len(delta))
            aggregate[name]["paired_delta_t95_ci_approx"] = [
                float(delta.mean() - 2.776445 * se),
                float(delta.mean() + 2.776445 * se),
            ]
        else:
            aggregate[name]["paired_delta_t95_ci_approx"] = None

    for record in records:
        seed = str(record["settings"]["seed"])
        per_seed[seed] = {
            name: {
                "baseline": metric(record, "baseline_metrics", name),
                "fusion": metric(record, "best_metrics", name),
                "delta": metric(record, "best_metrics", name)
                - metric(record, "baseline_metrics", name),
            }
            for name in ("NDCG", "Recall", "Precision", "Hit Ratio")
        }

    out = {
        "dataset": records[0]["settings"]["dataset"],
        "protocol": "frozen_validation_selected_one_shot_ood_test",
        "alpha": float(records[0]["settings"]["alpha"]),
        "validation_decision": validation["summary"],
        "seeds": seeds,
        "aggregate_top20": aggregate,
        "per_seed_top20": per_seed,
        "caution": (
            "Five paired seeds give limited exact-significance resolution; "
            "the t interval is descriptive and assumes approximately normal paired differences."
        ),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps(out["aggregate_top20"], ensure_ascii=False, indent=2))
    print("Saved", args.out)


if __name__ == "__main__":
    main()
