#!/usr/bin/env python
"""Aggregate frozen corrected-v4 OOD semantic comparisons."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


SEEDS = (1024, 2048, 3072, 4096, 5120)
ARMS = ("none", "semantic_real", "semantic_shuffled")


def ndcg(record, field="best_metrics"):
    return float(record[field]["Top20"]["NDCG"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--confirmation", required=True)
    parser.add_argument("--records", default="experiments/records")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    confirmation = json.loads(Path(args.confirmation).read_text(encoding="utf-8"))
    if not confirmation.get("ood_test_allowed", False):
        raise RuntimeError("validation confirmation has not authorized OOD testing")

    by_arm = {arm: {} for arm in ARMS}
    for path in Path(args.records).glob(f"{args.dataset}_v4_ood_*_seed*_v4.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        settings = record.get("settings", {})
        if settings.get("protocol_version") != "corrected_v4_frozen_ood":
            continue
        arm = settings.get("comparison_label")
        if arm in by_arm:
            by_arm[arm][int(settings["seed"])] = record
    for arm, records in by_arm.items():
        if tuple(sorted(records)) != SEEDS:
            raise RuntimeError(f"{arm}: expected OOD records for {SEEDS}, got {tuple(sorted(records))}")

    def values(arm, field="best_metrics"):
        return [ndcg(by_arm[arm][seed], field) for seed in SEEDS]

    def stats(x):
        return {
            "mean": statistics.mean(x),
            "sample_std": statistics.stdev(x),
            "values": x,
        }

    none = values("none")
    real = values("semantic_real")
    shuffled = values("semantic_shuffled")
    real_alpha0 = values("semantic_real", "baseline_metrics")
    output = {
        "dataset": args.dataset,
        "protocol": "corrected_v4_frozen_validation_selected_ood",
        "alpha": float(by_arm["semantic_real"][SEEDS[0]]["settings"]["alpha"]),
        "validation_confirmation": confirmation,
        "top20_ndcg": {
            "none": stats(none),
            "semantic_real": stats(real),
            "semantic_shuffled": stats(shuffled),
            "semantic_real_alpha0_same_checkpoint": stats(real_alpha0),
            "real_minus_none": stats([a - b for a, b in zip(real, none)]),
            "real_minus_shuffled": stats([a - b for a, b in zip(real, shuffled)]),
            "real_minus_alpha0_same_checkpoint": stats([a - b for a, b in zip(real, real_alpha0)]),
        },
        "all_test_records_frozen_after_validation": True,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output["top20_ndcg"], ensure_ascii=False, indent=2))
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
