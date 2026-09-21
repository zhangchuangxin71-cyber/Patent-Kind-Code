#!/usr/bin/env python
"""Summarize frozen v4 score-level late-fusion OOD comparisons."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


SEEDS = (1024, 2048, 3072, 4096, 5120)


def metric(record, field):
    return float(record[field]["Top20"]["NDCG"])


def stats(values):
    return {
        "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values),
        "values": values,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--validation_report", required=True)
    ap.add_argument("--real_records", nargs="+", required=True)
    ap.add_argument("--shuffle_records", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    validation = json.loads(Path(args.validation_report).read_text(encoding="utf-8"))
    summary = validation["summary"]
    if not summary.get("ood_test_allowed", False):
        raise RuntimeError("validation report did not authorize OOD testing")

    def load(paths):
        records = [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]
        by_seed = {int(r["settings"]["seed"]): r for r in records}
        if tuple(sorted(by_seed)) != SEEDS:
            raise RuntimeError(f"Expected seeds {SEEDS}, got {tuple(sorted(by_seed))}")
        return by_seed

    real_records = load(args.real_records)
    shuffle_records = load(args.shuffle_records)
    baseline = [metric(real_records[s], "baseline_metrics") for s in SEEDS]
    real = [metric(real_records[s], "best_metrics") for s in SEEDS]
    shuffled = [metric(shuffle_records[s], "best_metrics") for s in SEEDS]
    output = {
        "dataset": args.dataset,
        "protocol": "corrected_v4_frozen_validation_selected_ood",
        "alpha": float(summary["selected_alpha"]),
        "validation_summary": summary,
        "top20_ndcg": {
            "baseline_alpha0": stats(baseline),
            "semantic_real": stats(real),
            "semantic_shuffled": stats(shuffled),
            "real_minus_baseline": stats([a - b for a, b in zip(real, baseline)]),
            "real_minus_shuffled": stats([a - b for a, b in zip(real, shuffled)]),
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
