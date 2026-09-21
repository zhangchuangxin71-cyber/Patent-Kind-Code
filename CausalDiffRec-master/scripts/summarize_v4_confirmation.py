#!/usr/bin/env python
"""Aggregate corrected-v4 validation-only semantic confirmation records."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--records", default="experiments/records")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    grouped = {"none": {}, "semantic_real": {}, "semantic_shuffled": {}}
    for path in Path(args.records).glob(f"{args.dataset}*v4_confirm*seed*_v4.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        settings = record.get("settings", {})
        if settings.get("protocol_version") != "corrected_v4_controlled":
            continue
        if not settings.get("validation_only", False) or record.get("best_metrics") is not None:
            continue
        name = path.name
        arm = next((candidate for candidate in grouped if f"v4_confirm_{candidate}_seed" in name), None)
        if arm is None:
            continue
        seed = int(settings["seed"])
        grouped[arm][seed] = float(record["best_val_metrics"]["Top20"]["NDCG"])

    arms = {}
    for arm, by_seed in grouped.items():
        values = [by_seed[s] for s in sorted(by_seed)]
        arms[arm] = {
            "by_seed": {str(k): v for k, v in sorted(by_seed.items())},
            "mean": statistics.mean(values) if values else None,
            "sample_std": statistics.stdev(values) if len(values) > 1 else None,
            "n": len(values),
        }

    checks = {"complete_five_seeds": all(arms[a]["n"] == 5 for a in arms)}
    if checks["complete_five_seeds"]:
        none, real, shuffled = grouped["none"], grouped["semantic_real"], grouped["semantic_shuffled"]
        checks.update({
            "real_mean_beats_none": arms["semantic_real"]["mean"] > arms["none"]["mean"],
            "real_mean_beats_shuffled": arms["semantic_real"]["mean"] > arms["semantic_shuffled"]["mean"],
            "confirmation_seeds_beat_none": all(real[s] > none[s] for s in (4096, 5120)),
            "confirmation_seeds_beat_shuffled": all(real[s] > shuffled[s] for s in (4096, 5120)),
            "relative_gain_at_least_1pct": (
                arms["semantic_real"]["mean"] / arms["none"]["mean"] - 1.0 >= 0.01
            ),
        })
    output = {
        "dataset": args.dataset,
        "protocol": "corrected_v4_controlled_validation_only",
        "arms": arms,
        "checks": checks,
        "ood_test_allowed": bool(checks) and all(checks.values()),
    }
    rendered = json.dumps(output, ensure_ascii=False, indent=2)
    out = Path(args.out) if args.out else Path("experiments/summaries") / f"{args.dataset}_v4_confirmation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
