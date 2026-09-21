#!/usr/bin/env python
"""Validate a processed strict dataset and write READY or fail."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import dgl
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.io import data_strict_root, load_registry, write_json  # noqa: E402

VERSION = {
    "kuairec": "v1_strict",
    "food": "v1_strict",
    "yelp2018": "v1_strict",
    "douban": "v1_strict",
    "movielens1m": "v1_strict",
    "amazon_beauty": "v1_strict",
}


def pairset(df):
    return set(zip(df["raw_user_id"].astype(str), df["raw_item_id"].astype(str)))


def validate(ds: str, version: str | None = None) -> int:
    reg = load_registry()
    root = data_strict_root(reg)
    if ds == "douban" and reg["datasets"]["douban"].get("status_default") == "blocked":
        out = root / "processed" / "douban" / (version or VERSION[ds])
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "BLOCKED.json", {
            "dataset": ds,
            "reason": reg["datasets"]["douban"].get("blocked_reason", "blocked"),
        })
        print("Douban BLOCKED (by design until source confirmed)")
        return 0

    out = root / "processed" / ds / (version or VERSION[ds])
    blocked = out / "BLOCKED.json"
    if blocked.exists() and reg["datasets"].get(ds, {}).get("status_default") != "blocked":
        blocked.unlink()

    errors = []
    ready = out / "READY"
    if ready.exists():
        ready.unlink()

    required = [
        out / "manifest.json",
        out / "interactions" / "train.parquet",
        out / "interactions" / "val.parquet",
        out / "interactions" / "iid_test.parquet",
        out / "interactions" / "ood_test.parquet",
        out / "graphs" / "train.bin",
        out / "graphs" / "eval_context_ood.bin",
        out / "metadata" / "items.jsonl",
        out / "mappings" / "user_id_map.json",
    ]
    for p in required:
        if not p.exists():
            errors.append(f"missing {p}")

    if errors:
        write_json(out / "validation_report.json", {"ok": False, "errors": errors})
        print("FAIL:", errors)
        return 1

    splits = {
        sp: pd.read_parquet(out / "interactions" / f"{sp}.parquet")
        for sp in ["train", "val", "iid_test", "ood_test"]
    }
    # pairwise overlap
    names = list(splits.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            inter = pairset(splits[a]) & pairset(splits[b])
            if inter:
                errors.append(f"overlap {a}&{b}: {len(inter)}")

    # graph contract: train.bin edges == train interactions (undirected x2)
    g = dgl.load_graphs(str(out / "graphs" / "train.bin"))[0][0]
    n_train = len(splits["train"])
    # undirected => 2 * n_train edges expected (approx; duplicates possible if multi)
    if g.number_of_edges() < n_train:
        errors.append(f"train graph edges {g.number_of_edges()} < train interactions {n_train}")

    # eval ood graph should not contain ood edges as majority — we require edge count == train graph
    g_ood = dgl.load_graphs(str(out / "graphs" / "eval_context_ood.bin"))[0][0]
    if g_ood.number_of_edges() != g.number_of_edges():
        errors.append("eval_context_ood edge count != train graph (must use train edges only)")

    # KuaiRec specific
    import json
    with open(out / "manifest.json") as f:
        manifest = json.load(f)
    if not manifest.get("license_notes"):
        errors.append("license_notes empty")
    if not manifest.get("split_seed"):
        errors.append("split_seed missing")
    if ds == "kuairec":
        stats = manifest.get("stats", {})
        if "overlap_removed_from_small" not in stats:
            errors.append("kuairec missing overlap_removed_from_small")

    report = {"ok": len(errors) == 0, "errors": errors, "split_counts": {k: len(v) for k, v in splits.items()}}
    write_json(out / "validation_report.json", report)
    if errors:
        print("FAIL:", errors)
        return 1
    ready.write_text("ok\n", encoding="utf-8")
    print("READY:", ready)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["food", "kuairec", "yelp2018", "douban", "movielens1m", "amazon_beauty"])
    ap.add_argument("--version", default=None)
    args = ap.parse_args()
    sys.exit(validate(args.dataset, args.version))


if __name__ == "__main__":
    main()
