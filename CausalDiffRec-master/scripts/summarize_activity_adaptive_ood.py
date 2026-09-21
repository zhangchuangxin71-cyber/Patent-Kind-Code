#!/usr/bin/env python
"""Summarize frozen activity-adaptive OOD records."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


SEEDS = (1024, 2048, 3072, 4096, 5120)


def stats(values):
    return {"mean": statistics.mean(values), "sample_std": statistics.stdev(values), "values": values}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--validation_report", required=True)
    ap.add_argument("--real_records", nargs="+", required=True)
    ap.add_argument("--shuffle_records", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    validation = json.loads(Path(args.validation_report).read_text(encoding="utf-8"))["summary"]
    if not validation.get("ood_test_allowed"):
        raise RuntimeError("validation gate did not authorize OOD")

    def load(paths):
        records = [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]
        result = {int(r["settings"]["seed"]): r for r in records}
        if tuple(sorted(result)) != SEEDS:
            raise RuntimeError(f"expected {SEEDS}, got {tuple(sorted(result))}")
        return result

    real, shuffle = load(args.real_records), load(args.shuffle_records)
    metric = lambda record, name: float(record["test_metrics"][name]["Top20"]["NDCG"])
    baseline = [metric(real[s], "baseline") for s in SEEDS]
    uniform = [metric(real[s], "uniform") for s in SEEDS]
    adaptive = [metric(real[s], "adaptive") for s in SEEDS]
    adaptive_shuffle = [metric(shuffle[s], "adaptive") for s in SEEDS]
    output = {
        "dataset": args.dataset,
        "protocol": "corrected_v4_activity_adaptive_frozen_ood",
        "validation_summary": validation,
        "test_gt_loaded": True,
        "top20_ndcg": {
            "baseline": stats(baseline),
            "uniform_real": stats(uniform),
            "adaptive_real": stats(adaptive),
            "adaptive_shuffle": stats(adaptive_shuffle),
            "adaptive_minus_uniform": stats([a - b for a, b in zip(adaptive, uniform)]),
            "adaptive_minus_baseline": stats([a - b for a, b in zip(adaptive, baseline)]),
            "adaptive_minus_shuffle": stats([a - b for a, b in zip(adaptive, adaptive_shuffle)]),
        },
        "all_test_records_frozen_after_validation": True,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output["top20_ndcg"], indent=2))


if __name__ == "__main__":
    main()
