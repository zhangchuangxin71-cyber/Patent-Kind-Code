#!/usr/bin/env python
"""Summarize the fair LightGCN/CausalDiffRec/full-method OOD comparison."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


SEEDS = (1024, 2048, 3072, 4096, 5120)
METRICS = ("NDCG", "Recall", "Precision", "Hit Ratio")


def metric(record, name):
    return float(record["best_metrics"]["Top20"][name])


def load(paths):
    records = [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]
    by_seed = {int(record["settings"]["seed"]): record for record in records}
    if len(records) != len(SEEDS) or tuple(sorted(by_seed)) != SEEDS:
        raise RuntimeError(f"Expected exactly seeds {SEEDS}, got {tuple(sorted(by_seed))}")
    return by_seed


def stats(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(values.mean()),
        "sample_std": float(values.std(ddof=1)),
        "values": [float(x) for x in values],
    }


def paired(values):
    values = np.asarray(values, dtype=float)
    result = stats(values)
    result["positive_seeds"] = int((values > 0).sum())
    result["negative_seeds"] = int((values < 0).sum())
    result["zero_seeds"] = int((values == 0).sum())
    if values.std(ddof=1) > 0:
        half = 2.776445 * values.std(ddof=1) / math.sqrt(len(values))
        result["paired_delta_t95_ci_approx"] = [
            float(values.mean() - half), float(values.mean() + half),
        ]
    else:
        result["paired_delta_t95_ci_approx"] = None
    wins, nonzero = int((values > 0).sum()), int((values != 0).sum())
    result["one_sided_exact_sign_p_positive"] = (
        float(sum(math.comb(nonzero, k) for k in range(wins, nonzero + 1)) / 2 ** nonzero)
        if nonzero else 1.0
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="movielens1m")
    parser.add_argument("--audit", required=True)
    parser.add_argument("--lightgcn_records", nargs="+", required=True)
    parser.add_argument("--causaldiffrec_records", nargs="+", required=True)
    parser.add_argument("--full_records", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    audit = json.loads(Path(args.audit).read_text(encoding="utf-8"))
    if not audit.get("ood_test_allowed", False):
        raise RuntimeError("Validation fairness audit did not authorize OOD testing")
    arms = {
        "lightgcn": load(args.lightgcn_records),
        "causaldiffrec": load(args.causaldiffrec_records),
        "full_method": load(args.full_records),
    }
    integrity = {
        "identical_seed_sets": all(tuple(sorted(records)) == SEEDS for records in arms.values()),
        "frozen_protocol_labels": all(
            record["settings"].get("protocol_version") == "corrected_v5_fair_bpr25_frozen_ood"
            for records in arms.values() for record in records.values()
        ),
        "expected_alphas": all(
            float(record["settings"]["alpha"]) == (0.75 if arm == "full_method" else 0.0)
            for arm, records in arms.items() for record in records.values()
        ),
    }
    if not all(integrity.values()):
        raise RuntimeError(f"OOD integrity check failed: {integrity}")

    aggregate = {}
    per_seed = {}
    for name in METRICS:
        values = {
            arm: np.asarray([metric(records[seed], name) for seed in SEEDS])
            for arm, records in arms.items()
        }
        aggregate[name] = {
            arm: stats(arm_values) for arm, arm_values in values.items()
        }
        aggregate[name].update({
            "full_minus_causaldiffrec": paired(values["full_method"] - values["causaldiffrec"]),
            "full_minus_lightgcn": paired(values["full_method"] - values["lightgcn"]),
            "causaldiffrec_minus_lightgcn": paired(values["causaldiffrec"] - values["lightgcn"]),
            "full_vs_causaldiffrec_relative_percent": float(
                100 * (values["full_method"].mean() - values["causaldiffrec"].mean())
                / values["causaldiffrec"].mean()
            ),
            "full_vs_lightgcn_relative_percent": float(
                100 * (values["full_method"].mean() - values["lightgcn"].mean())
                / values["lightgcn"].mean()
            ),
        })
    for seed in SEEDS:
        per_seed[str(seed)] = {
            name: {arm: metric(records[seed], name) for arm, records in arms.items()}
            for name in METRICS
        }

    output = {
        "dataset": args.dataset,
        "protocol_version": "corrected_v5_fair_three_method_bpr25_frozen_one_shot_ood",
        "validation_audit": args.audit,
        "shared_bpr_pass_budget": 25,
        "seeds": list(SEEDS),
        "method_definitions": audit["comparison_arms"],
        "integrity_checks": integrity,
        "aggregate_top20": aggregate,
        "per_seed_top20": per_seed,
        "all_test_records_frozen_after_validation": True,
        "interpretation_guard": (
            "The table supports only the observed pairwise directions. Do not claim the full "
            "method beats a baseline when its reported paired difference is non-positive."
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output["aggregate_top20"]["NDCG"], ensure_ascii=False, indent=2))
    print("Saved", out)


if __name__ == "__main__":
    main()
