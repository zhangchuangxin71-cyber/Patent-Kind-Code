#!/usr/bin/env python
"""Aggregate strict OOD records by dataset and edge-gate mode."""
import argparse
import json
import math
import os
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECORDS = os.path.join(ROOT, "experiments", "records")
OUT = os.path.join(ROOT, "experiments", "summaries")

SEEDS = {1024, 2048, 3072, 4096, 5120}
MODES = {"e0", "none", "soft_pair_environment", "random_pair_environment", "e3", "e4"}


def metric(record, top, name):
    metrics = record.get("best_metrics") or record["best_val_metrics"]
    return float(metrics[top][name])


def mean_std(values):
    if not values:
        return None
    mean = sum(values) / len(values)
    if len(values) < 2:
        return {"mean": mean, "std": None, "n": len(values), "values": values}
    var = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return {"mean": mean, "std": math.sqrt(var), "n": len(values), "values": values}


def load_records(dataset):
    grouped = defaultdict(list)
    for fn in os.listdir(RECORDS):
        if not fn.endswith(".json"):
            continue
        path = os.path.join(RECORDS, fn)
        try:
            with open(path, encoding="utf-8") as f:
                rec = json.load(f)
            settings = rec.get("settings", {})
            mode = settings.get("edge_gate_mode")
            # E3 uses the same ``none`` edge-gate setting as Stage-I.  It must
            # therefore be identified from the explicit method marker before
            # grouping, otherwise a semantic run is silently mixed into the
            # no-semantic Stage-I ablation.
            if settings.get("method") == "lsci_e3":
                mode = "e3"
            elif settings.get("method") == "lsci_e4_lamsem0":
                # E4 keeps semantic fusion but disables InfoNCE.  Like E3 it
                # uses edge_gate_mode=none, so preserve it as a distinct arm.
                mode = "e4"
            if not mode and "causaldiffrec_e0" in rec.get("checkpoint", ""):
                mode = "e0"
            expected_tag = {
                "e0": "strict_e0_seed",
                "none": "strict_none_seed",
                "soft_pair_environment": "strict_soft_pair_environment_seed",
                "random_pair_environment": "strict_random_pair_environment_seed",
                "e3": "strict_e3_semantic_seed",
                "e4": "strict_ood_e4_lamsem0_seed",
            }.get(mode, "")
            if mode in MODES and settings.get("protocol_version") == "corrected_v3_stable" \
                and settings.get("data_root", "").rstrip("/").endswith(
                f"processed/{dataset}/v1_strict"
            ) and settings.get("eval_split") == "ood" \
                and int(settings.get("epochs", 0)) == 25 \
                and not settings.get("validation_only", False) \
                and expected_tag in fn \
                and rec.get("best_val_metrics"):
                grouped[mode].append(rec)
        except (OSError, ValueError, KeyError):
            continue
    return grouped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()
    grouped = load_records(args.dataset)
    output = {"dataset": args.dataset, "protocol": "strict_ood", "methods": {}}
    for mode in sorted(MODES | set(grouped)):
        records = sorted(grouped.get(mode, []), key=lambda r: r.get("settings", {}).get("seed", -1))
        seeds = [r.get("settings", {}).get("seed") for r in records]
        missing = sorted(SEEDS - set(seeds))
        methods = {}
        for top, label in (("Top10", "10"), ("Top20", "20")):
            for name, metric_name in (("Recall", "Recall"), ("NDCG", "NDCG")):
                values = [metric(r, top, metric_name) for r in records]
                methods[f"Recall@{label}" if name == "Recall" else f"NDCG@{label}"] = mean_std(values)
        methods["seeds"] = seeds
        methods["missing_seeds"] = missing
        methods["n_runs"] = len(records)
        output["methods"][mode] = methods
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{args.dataset}_strict_ood_ablation.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(path)
    for mode, result in output["methods"].items():
        n20 = result.get("NDCG@20")
        if n20:
            std = "n/a" if n20["std"] is None else f"{n20['std']:.5f}"
            print(f"{mode}: n={n20['n']} NDCG@20={n20['mean']:.5f} ± {std} missing={result['missing_seeds']}")


if __name__ == "__main__":
    main()
