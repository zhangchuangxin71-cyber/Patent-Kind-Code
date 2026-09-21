#!/usr/bin/env python
"""Summarize frozen one-shot OOD raw-text vs DeepSeek-text comparisons."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


SEEDS = (1024, 2048, 3072, 4096, 5120)
METRICS = ("NDCG", "Recall", "Precision", "Hit Ratio")


def value(record, field, metric):
    return float(record[field]["Top20"][metric])


def describe(values):
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "sample_std": float(array.std(ddof=1)),
        "values": [float(x) for x in array],
    }


def paired(values):
    array = np.asarray(values, dtype=float)
    result = describe(array)
    result["positive_seeds"] = int((array > 0).sum())
    result["negative_seeds"] = int((array < 0).sum())
    result["zero_seeds"] = int((array == 0).sum())
    if len(array) > 1 and array.std(ddof=1) > 0:
        half = 2.776445 * array.std(ddof=1) / math.sqrt(len(array))
        result["paired_delta_t95_ci_approx"] = [
            float(array.mean() - half), float(array.mean() + half),
        ]
    else:
        result["paired_delta_t95_ci_approx"] = None
    # With five pairs, five wins has a one-sided exact sign-test p=1/32.
    wins = int((array > 0).sum())
    nonzero = int((array != 0).sum())
    result["one_sided_exact_sign_p_positive"] = (
        float(sum(math.comb(nonzero, k) for k in range(wins, nonzero + 1)) / (2 ** nonzero))
        if nonzero else 1.0
    )
    return result


def load_by_seed(paths):
    records = [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]
    by_seed = {int(record["settings"]["seed"]): record for record in records}
    if tuple(sorted(by_seed)) != SEEDS or len(records) != len(SEEDS):
        raise RuntimeError(f"Expected exactly seeds {SEEDS}, got {tuple(sorted(by_seed))}")
    return by_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--validation_report", required=True)
    parser.add_argument("--llm_records", nargs="+", required=True)
    parser.add_argument("--raw_records", nargs="+", required=True)
    parser.add_argument("--raw_own_records", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    validation = json.loads(Path(args.validation_report).read_text(encoding="utf-8"))
    decision = validation["summary"]
    if not decision.get("ood_test_allowed", False):
        raise RuntimeError("Validation report did not authorize OOD testing")
    alpha = float(decision["selected_alpha"]["shared_llm_selected"])
    raw_alpha = float(decision["selected_alpha"]["raw_independently_selected"])
    llm = load_by_seed(args.llm_records)
    raw = load_by_seed(args.raw_records)
    raw_own = load_by_seed(args.raw_own_records)

    integrity = {
        "same_seed_set": (
            tuple(sorted(llm)) == tuple(sorted(raw)) == tuple(sorted(raw_own)) == SEEDS
        ),
        "same_shared_alpha": all(
            float(llm[s]["settings"]["alpha"]) == alpha
            and float(raw[s]["settings"]["alpha"]) == alpha
            for s in SEEDS
        ),
        "same_source_record_per_seed": all(
            llm[s]["source_record"] == raw[s]["source_record"] == raw_own[s]["source_record"]
            for s in SEEDS
        ),
        "same_checkpoint_per_seed": all(
            llm[s]["checkpoint"] == raw[s]["checkpoint"] == raw_own[s]["checkpoint"]
            for s in SEEDS
        ),
        "same_collaborative_baseline_per_seed": all(
            all(value(llm[s], "baseline_metrics", m)
                == value(raw[s], "baseline_metrics", m)
                == value(raw_own[s], "baseline_metrics", m)
                for m in METRICS)
            for s in SEEDS
        ),
        "validation_did_not_load_test_gt": decision.get("test_gt_loaded") is False,
        "raw_own_alpha_matches_validation": all(
            float(raw_own[s]["settings"]["alpha"]) == raw_alpha for s in SEEDS
        ),
    }
    if not all(integrity.values()):
        raise RuntimeError(f"Paired-comparison integrity check failed: {integrity}")

    aggregate = {}
    per_seed = {}
    for metric in METRICS:
        baseline = [value(llm[s], "baseline_metrics", metric) for s in SEEDS]
        raw_values = [value(raw[s], "best_metrics", metric) for s in SEEDS]
        raw_own_values = [value(raw_own[s], "best_metrics", metric) for s in SEEDS]
        llm_values = [value(llm[s], "best_metrics", metric) for s in SEEDS]
        aggregate[metric] = {
            "collaborative_baseline": describe(baseline),
            "raw_text_sbert_shared_alpha": describe(raw_values),
            "raw_text_sbert_own_validation_alpha": describe(raw_own_values),
            "deepseek_text_sbert": describe(llm_values),
            "raw_shared_minus_baseline": paired(np.asarray(raw_values) - np.asarray(baseline)),
            "raw_own_minus_baseline": paired(np.asarray(raw_own_values) - np.asarray(baseline)),
            "deepseek_minus_baseline": paired(np.asarray(llm_values) - np.asarray(baseline)),
            "deepseek_minus_raw_shared_alpha": paired(
                np.asarray(llm_values) - np.asarray(raw_values)
            ),
            "deepseek_minus_raw_each_own_alpha": paired(
                np.asarray(llm_values) - np.asarray(raw_own_values)
            ),
            "deepseek_vs_raw_shared_alpha_relative_percent": float(
                100.0 * (np.mean(llm_values) - np.mean(raw_values)) / np.mean(raw_values)
            ),
            "deepseek_vs_raw_each_own_alpha_relative_percent": float(
                100.0 * (np.mean(llm_values) - np.mean(raw_own_values)) / np.mean(raw_own_values)
            ),
        }
    for seed in SEEDS:
        per_seed[str(seed)] = {
            metric: {
                "baseline": value(llm[seed], "baseline_metrics", metric),
                "raw_text_sbert_shared_alpha": value(raw[seed], "best_metrics", metric),
                "raw_text_sbert_own_validation_alpha": value(
                    raw_own[seed], "best_metrics", metric
                ),
                "deepseek_text_sbert": value(llm[seed], "best_metrics", metric),
                "deepseek_minus_raw_shared_alpha": (
                    value(llm[seed], "best_metrics", metric)
                    - value(raw[seed], "best_metrics", metric)
                ),
                "deepseek_minus_raw_each_own_alpha": (
                    value(llm[seed], "best_metrics", metric)
                    - value(raw_own[seed], "best_metrics", metric)
                ),
            }
            for metric in METRICS
        }

    output = {
        "dataset": args.dataset,
        "protocol_version": "corrected_v5_raw_vs_deepseek_frozen_one_shot_ood",
        "comparison": (
            "primary: same frozen collaborative checkpoint, same SBERT encoder, same shared "
            "validation-selected alpha; secondary: equal alpha search spaces and each text "
            "source's own validation-selected alpha"
        ),
        "alpha": alpha,
        "raw_own_alpha": raw_alpha,
        "seeds": list(SEEDS),
        "validation_report": args.validation_report,
        "validation_decision": decision,
        "integrity_checks": integrity,
        "aggregate_top20": aggregate,
        "per_seed_top20": per_seed,
        "all_test_records_frozen_after_validation": True,
        "statistical_caution": (
            "Five paired seeds have low power. The paired t interval is descriptive; "
            "the exact sign test requires five of five wins for one-sided p=0.03125."
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output["aggregate_top20"]["NDCG"], ensure_ascii=False, indent=2))
    print("Saved", out)


if __name__ == "__main__":
    main()
