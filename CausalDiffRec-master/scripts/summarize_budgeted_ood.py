#!/usr/bin/env python
"""Compare two frozen five-seed OOD reports under an equal BPR budget."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def stats(values):
    return {
        "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values),
        "values": values,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pure_report", required=True)
    ap.add_argument("--lsci_report", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pure = json.loads(Path(args.pure_report).read_text(encoding="utf-8"))
    lsci = json.loads(Path(args.lsci_report).read_text(encoding="utf-8"))
    if pure["dataset"] != lsci["dataset"]:
        raise ValueError("arm reports use different datasets")
    if not (pure["all_test_records_frozen_after_validation"] and
            lsci["all_test_records_frozen_after_validation"]):
        raise RuntimeError("both arms must be frozen after validation")

    p = pure["top20_ndcg"]
    l = lsci["top20_ndcg"]
    p_base, l_base = p["baseline_alpha0"]["values"], l["baseline_alpha0"]["values"]
    p_real, l_real = p["semantic_real"]["values"], l["semantic_real"]["values"]
    output = {
        "dataset": pure["dataset"],
        "protocol": "corrected_v4_equal_bpr200_frozen_ood_comparison",
        "test_gt_loaded": True,
        "bpr_pass_budget_per_arm": 200,
        "alphas_frozen_from_validation": {
            "pure_lightgcn": pure["alpha"],
            "lsci_none": lsci["alpha"],
        },
        "pure_lightgcn": p,
        "lsci_none": l,
        "paired_arm_differences": {
            "pure_minus_lsci_baseline": stats([a - b for a, b in zip(p_base, l_base)]),
            "pure_minus_lsci_semantic_real": stats([a - b for a, b in zip(p_real, l_real)]),
        },
        "interpretation_guard": (
            "OOD ground truth was used only after both validation reports authorized testing; "
            "no parameter was selected from these OOD results."
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "alphas": output["alphas_frozen_from_validation"],
        "pure": p,
        "lsci": l,
        "paired": output["paired_arm_differences"],
    }, ensure_ascii=False, indent=2))
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
