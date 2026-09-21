#!/usr/bin/env python
"""Summarize validation-only corrected-v4 mechanism smoke records."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def ndcg20(record):
    return float(record["best_val_metrics"]["Top20"]["NDCG"])


def final_ndcg20(record):
    return float(record["epoch_logs"][-1]["val_metrics"]["Top20"]["NDCG"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--records", default="experiments/records")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    rows = {}
    for path in Path(args.records).glob(f"{args.dataset}*v4_smoke*seed*_v4.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("settings", {}).get("protocol_version") != "corrected_v4_controlled":
            continue
        run_id = str(record["settings"].get("run_id", path.stem))
        arm = run_id.split("v4_smoke_", 1)[-1].split("_strict_", 1)[0]
        rows[arm] = {
            "selected_ndcg20": ndcg20(record),
            "final_epoch_ndcg20": final_ndcg20(record),
            "path": str(path),
        }

    checks = {}
    if "learned" in rows and "random" in rows:
        checks["learned_beats_random_at_matched_final_epoch"] = (
            rows["learned"]["final_epoch_ndcg20"]
            > rows["random"]["final_epoch_ndcg20"]
        )
    if "semantic_real" in rows and "semantic_shuffled" in rows:
        checks["real_semantic_beats_shuffled"] = (
            rows["semantic_real"]["selected_ndcg20"]
            > rows["semantic_shuffled"]["selected_ndcg20"]
        )
    output = {"dataset": args.dataset, "validation_only": True, "arms": rows, "checks": checks}
    rendered = json.dumps(output, ensure_ascii=False, indent=2)
    out = Path(args.out) if args.out else Path("experiments/summaries") / f"{args.dataset}_v4_smoke.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
