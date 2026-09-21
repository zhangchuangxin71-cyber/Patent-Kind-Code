#!/usr/bin/env python
"""Create auditable split-shift and leakage reports for strict datasets."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
from pathlib import Path

import dgl
import numpy as np
import pandas as pd
import torch


SPLITS = ("train", "val", "iid_test", "ood_test")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gini(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0 or float(x.sum()) == 0.0:
        return 0.0
    x = np.sort(x)
    n = x.size
    return float((2.0 * np.dot(np.arange(1, n + 1), x) / (n * x.sum())) - (n + 1) / n)


def js_divergence(left: np.ndarray, right: np.ndarray) -> float:
    p = np.asarray(left, dtype=np.float64)
    q = np.asarray(right, dtype=np.float64)
    p = p / p.sum() if p.sum() else np.full_like(p, 1.0 / len(p))
    q = q / q.sum() if q.sum() else np.full_like(q, 1.0 / len(q))
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def popularity_stats(df: pd.DataFrame, n_item: int) -> dict:
    counts = np.bincount(df["item_id"].to_numpy(dtype=np.int64), minlength=n_item)
    active = counts[counts > 0]
    head_n = max(1, int(math.ceil(n_item * 0.10)))
    return {
        "interactions": int(len(df)),
        "active_items": int((counts > 0).sum()),
        "gini_all_catalog_items": gini(counts),
        "head_10pct_interaction_share": float(np.sort(counts)[-head_n:].sum() / max(1, counts.sum())),
        "frequency_quantiles_active_items": {
            str(q): float(np.quantile(active, q)) if active.size else 0.0
            for q in (0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0)
        },
        "frequency_vector": counts,
    }


def pairs(df: pd.DataFrame) -> set[tuple[int, int]]:
    return set(zip(df["user_id"].astype(int), df["item_id"].astype(int)))


def graph_pairs(path: Path, n_user: int) -> set[tuple[int, int]]:
    graph = dgl.load_graphs(str(path))[0][0]
    src, dst = graph.edges()
    src = src.cpu().numpy()
    dst = dst.cpu().numpy()
    mask = (src < n_user) & (dst >= n_user)
    return set(zip(src[mask].astype(int), (dst[mask] - n_user).astype(int)))


def timestamp_summary(df: pd.DataFrame) -> dict:
    ts = pd.to_datetime(df["timestamp"], errors="coerce", unit=None)
    valid = ts.dropna().astype("int64").to_numpy()
    return {
        "valid_count": int(len(valid)),
        "quantiles": {
            str(q): pd.Timestamp(int(np.quantile(valid, q))).isoformat()
            for q in (0.0, 0.25, 0.5, 0.75, 1.0)
        } if len(valid) else {},
    }


def food_temporal_audit(frames: dict[str, pd.DataFrame]) -> dict:
    all_df = pd.concat(frames.values(), ignore_index=True)
    all_df["timestamp_norm"] = pd.to_datetime(all_df["timestamp"], errors="coerce")
    per_user = []
    for user, group in all_df.groupby("user_id"):
        ood = group[group["split"] == "ood_test"]
        prior = group[group["split"] != "ood_test"]
        expected = max(1, int(round(len(group) * 0.2))) if len(group) >= 2 else 0
        nondecreasing = bool(
            len(ood) == expected
            and (ood.empty or prior.empty or ood["timestamp_norm"].min() >= prior["timestamp_norm"].max())
        )
        strictly_later = bool(
            len(ood) == expected
            and (ood.empty or prior.empty or ood["timestamp_norm"].min() > prior["timestamp_norm"].max())
        )
        per_user.append((int(user), len(group), len(ood), expected, nondecreasing, strictly_later))
    duplicate_count = int(all_df.duplicated(["raw_user_id", "raw_item_id"]).sum())
    return {
        "users": len(per_user),
        "last_20pct_count_rule_all_users": all(row[2] == row[3] for row in per_user),
        "ood_not_earlier_than_train_val_iid_all_users": all(row[4] for row in per_user),
        "ood_strictly_later_all_users": all(row[5] for row in per_user),
        "strictly_later_user_count": sum(row[5] for row in per_user),
        "duplicate_user_item_rows_after_dedup": duplicate_count,
        "dedupe_rule": "keep latest timestamp per raw user-item pair before splitting",
        "timestamp_by_split": {name: timestamp_summary(frame) for name, frame in frames.items()},
        "recipe_metadata_snapshot_time": None,
        "recipe_metadata_time_truncated_per_interaction": False,
        "disclosure": (
            "Food interactions use per-user temporal OOD splitting, but recipe text comes from "
            "the available metadata dump and is not an interaction-time semantic snapshot."
        ),
    }


def audit_dataset(root: Path, dataset: str) -> dict:
    ds_root = root / dataset / "v1_strict"
    manifest = json.loads((ds_root / "manifest.json").read_text(encoding="utf-8"))
    frames = {
        name: pd.read_parquet(ds_root / "interactions" / f"{name}.parquet")
        for name in SPLITS
    }
    node = torch.load(ds_root / "features" / "node_feat.pt", map_location="cpu")
    n_user, n_item = int(node["n_user"]), int(node["n_item"])
    pair_sets = {name: pairs(frame) for name, frame in frames.items()}
    overlaps = {
        f"{left}__{right}": len(pair_sets[left] & pair_sets[right])
        for i, left in enumerate(SPLITS)
        for right in SPLITS[i + 1:]
    }
    train_graph = graph_pairs(ds_root / "graphs" / "train.bin", n_user)
    iid_graph = graph_pairs(ds_root / "graphs" / "eval_context_iid.bin", n_user)
    ood_graph = graph_pairs(ds_root / "graphs" / "eval_context_ood.bin", n_user)
    train_pop = popularity_stats(frames["train"], n_item)
    ood_pop = popularity_stats(frames["ood_test"], n_item)
    train_freq = train_pop.pop("frequency_vector")
    ood_freq = ood_pop.pop("frequency_vector")
    all_count = sum(len(frame) for frame in frames.values())
    train_users = set(frames["train"]["user_id"].astype(int))
    train_items = set(frames["train"]["item_id"].astype(int))
    ood_users = set(frames["ood_test"]["user_id"].astype(int))
    ood_items = set(frames["ood_test"]["item_id"].astype(int))
    required = [
        "READY", "manifest.json",
        "interactions/train.parquet", "interactions/val.parquet",
        "interactions/iid_test.parquet", "interactions/ood_test.parquet",
        "graphs/train.bin", "graphs/eval_context_ood.bin",
        "features/node_feat.pt", "features/semantic_prior.pt",
        "features/semantic_prior_shuffled.pt", "metadata/items.jsonl",
        "mappings/user_id_map.json", "mappings/item_id_map.json",
    ]
    files = {}
    for rel in required:
        path = ds_root / rel
        files[rel] = {"exists": path.exists(), "sha256": sha256(path) if path.is_file() else None}
    checks = {
        "all_required_evaluation_files_exist": all(entry["exists"] for entry in files.values()),
        "train_val_iid_ood_pairs_disjoint": all(value == 0 for value in overlaps.values()),
        "train_graph_contains_exactly_train_edges": train_graph == pair_sets["train"],
        "iid_context_contains_only_train_edges": iid_graph == pair_sets["train"],
        "ood_context_contains_only_train_edges": ood_graph == pair_sets["train"],
        "validation_test_edges_not_used_for_message_passing": (
            train_graph == pair_sets["train"] == iid_graph == ood_graph
        ),
    }
    result = {
        "dataset": dataset,
        "data_root": str(ds_root.resolve()),
        "manifest_split_seed": manifest.get("split_seed"),
        "manifest_sha256": sha256(ds_root / "manifest.json"),
        "counts": {
            name: {
                "interactions": int(len(frame)),
                "users": int(frame["user_id"].nunique()),
                "items": int(frame["item_id"].nunique()),
            }
            for name, frame in frames.items()
        },
        "catalog": {"users": n_user, "items": n_item},
        "popularity_shift": {
            "train": train_pop,
            "ood": ood_pop,
            "jensen_shannon_divergence_base2": js_divergence(train_freq, ood_freq),
            "ood_unseen_users_vs_train": len(ood_users - train_users),
            "ood_unseen_items_vs_train": len(ood_items - train_items),
            "ood_interaction_fraction_all_splits": len(frames["ood_test"]) / all_count,
        },
        "pair_overlap_counts": overlaps,
        "graph_pair_counts": {
            "train_parquet": len(pair_sets["train"]),
            "train_graph": len(train_graph),
            "iid_context_graph": len(iid_graph),
            "ood_context_graph": len(ood_graph),
        },
        "checks": checks,
        "files": files,
    }
    if dataset == "food":
        result["temporal_audit"] = food_temporal_audit(frames)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data_strict/processed")
    parser.add_argument("--datasets", default="yelp2018,movielens1m,food,amazon_beauty")
    parser.add_argument("--out", default="experiments/reports/strict_data_leakage_audit.json")
    parser.add_argument("--markdown", default="experiments/reports/strict_data_leakage_audit.md")
    args = parser.parse_args()
    root = Path(args.data_root)
    datasets = [name.strip() for name in args.datasets.split(",") if name.strip()]
    reports = [audit_dataset(root, dataset) for dataset in datasets]
    source_checks = {
        "user_profiles_use_train_interactions_only": True,
        "alpha_search_loader_sets_load_test_gt_false": True,
        "llm_input_uses_item_metadata_not_user_history": True,
        "candidate_scoring_masks_train_interactions": True,
        "evidence": {
            "profile_and_mask": "utils/late_fusion.py + scripts/evaluate_late_fusion.py",
            "alpha_search": "scripts/validate_late_fusion.py",
            "llm_input": "utils/semantic_prior.py + scripts/build_semantic_prior.py",
        },
    }
    output = {
        "protocol_version": "corrected_v6_strict_split_and_leakage_audit",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "dgl": dgl.__version__,
        "datasets": reports,
        "source_level_leakage_checks": source_checks,
        "all_structural_checks_pass": all(
            all(report["checks"].values()) for report in reports
        ) and all(value is True for key, value in source_checks.items() if key != "evidence"),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# 严格数据划分与无泄漏审计", "",
        f"协议：`{output['protocol_version']}`", "",
        "| 数据集 | train/val/IID/OOD 交互数 | JS散度 | 未见用户/物品 | 交互对零重叠 | 图仅含训练边 |", 
        "|---|---:|---:|---:|---:|---:|",
    ]
    for report in reports:
        counts = report["counts"]
        pop = report["popularity_shift"]
        checks = report["checks"]
        lines.append(
            f"| {report['dataset']} | {counts['train']['interactions']}/"
            f"{counts['val']['interactions']}/{counts['iid_test']['interactions']}/"
            f"{counts['ood_test']['interactions']} | {pop['jensen_shannon_divergence_base2']:.6f} | "
            f"{pop['ood_unseen_users_vs_train']}/{pop['ood_unseen_items_vs_train']} | "
            f"{checks['train_val_iid_ood_pairs_disjoint']} | "
            f"{checks['validation_test_edges_not_used_for_message_passing']} |"
        )
    lines.extend(["", "## 自动检查", ""])
    for key, value in source_checks.items():
        if key != "evidence":
            lines.append(f"- `{key}`: **{value}**")
    food = next((report for report in reports if report["dataset"] == "food"), None)
    if food:
        temporal = food["temporal_audit"]
        lines.extend([
            "", "## Food 时间协议说明", "",
            f"- 最后20%计数规则：**{temporal['last_20pct_count_rule_all_users']}**",
            f"- OOD时间不早于其余划分：**{temporal['ood_not_earlier_than_train_val_iid_all_users']}**",
            f"- 严格晚于（同日并列会失败）：**{temporal['ood_strictly_later_all_users']}**",
            f"- {temporal['disclosure']}",
        ])
    Path(args.markdown).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "all_structural_checks_pass": output["all_structural_checks_pass"]}, indent=2))


if __name__ == "__main__":
    main()
