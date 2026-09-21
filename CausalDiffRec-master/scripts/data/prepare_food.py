#!/usr/bin/env python
"""Prepare Food strict temporal OOD splits + recipe text metadata."""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.filters import dedupe_keep_latest, sequential_user_then_item  # noqa: E402
from common.graph_export import export_graphs  # noqa: E402
from common.io import append_jsonl, data_strict_root, ensure_dir, load_registry, write_json  # noqa: E402
from common.manifest import build_manifest, finalize_manifest  # noqa: E402
from common.mapping import apply_maps, build_maps  # noqa: E402
from common.splits import temporal_ood_per_user  # noqa: E402

VERSION = "v1_strict"


def main():
    reg = load_registry()
    root = data_strict_root(reg)
    seed = int(reg["split_seed"])
    cfg = reg["datasets"]["food"]
    raw = root / "raw" / "food" / "kaggle_food_com"
    inter_p = raw / "RAW_interactions.csv"
    recipe_p = raw / "RAW_recipes.csv"
    if not inter_p.exists() or not recipe_p.exists():
        raise FileNotFoundError(f"Missing RAW files under {raw}; run download_datasets.py --dataset food")

    print("Loading interactions...")
    inter = pd.read_csv(inter_p, low_memory=False)
    inter = inter.rename(columns={"user_id": "raw_user_id", "recipe_id": "raw_item_id", "rating": "value", "date": "timestamp"})
    inter["raw_user_id"] = pd.to_numeric(inter["raw_user_id"], errors="coerce")
    inter["raw_item_id"] = pd.to_numeric(inter["raw_item_id"], errors="coerce")
    inter["value"] = pd.to_numeric(inter["value"], errors="coerce")
    inter["timestamp"] = pd.to_datetime(inter["timestamp"], errors="coerce")
    inter = inter.dropna(subset=["raw_user_id", "raw_item_id", "value", "timestamp"])
    inter = inter[inter["value"] >= cfg["filters"]["min_rating"]]
    inter["raw_user_id"] = inter["raw_user_id"].astype(np.int64)
    inter["raw_item_id"] = inter["raw_item_id"].astype(np.int64)
    inter = dedupe_keep_latest(inter)
    print("After rating+time filter:", len(inter))
    # Paper-matched Food rule: user>=15 then item>=15 (sequential), not iterative 15/50 k-core.
    inter = sequential_user_then_item(
        inter,
        min_user=cfg["filters"]["min_user_inter"],
        min_item=cfg["filters"]["min_item_inter"],
    )
    inter["raw_user_id"] = inter["raw_user_id"].astype(str)
    inter["raw_item_id"] = inter["raw_item_id"].astype(str)
    print(
        "After sequential filter:",
        len(inter),
        "users",
        inter["raw_user_id"].nunique(),
        "items",
        inter["raw_item_id"].nunique(),
    )
    if inter.empty:
        raise RuntimeError(
            "过滤后数据为空：请检查 min_user_inter/min_item_inter 或运行 "
            "python scripts/data/debug_food_counts.py"
        )

    print("Temporal OOD split...")
    all_df = temporal_ood_per_user(inter, seed=seed, ood_ratio=0.2)
    user_map, item_map, node_map = build_maps(all_df)
    all_df = apply_maps(all_df, user_map, item_map)
    n_user, n_item = len(user_map), len(item_map)

    out = root / "processed" / "food" / VERSION
    for sub in ["interactions", "metadata", "mappings", "graphs", "features"]:
        ensure_dir(out / sub)

    file_paths = {}
    for sp in ["train", "val", "iid_test", "ood_test"]:
        p = out / "interactions" / f"{sp}.parquet"
        all_df[all_df["split"] == sp].to_parquet(p, index=False)
        file_paths[f"interactions/{sp}.parquet"] = str(p)

    write_json(out / "mappings" / "user_id_map.json", user_map)
    write_json(out / "mappings" / "item_id_map.json", item_map)
    write_json(out / "mappings" / "node_id_map.json", node_map)

    print("Loading recipes for text...")
    recipes = pd.read_csv(recipe_p, low_memory=False)
    # id column is recipe id
    id_col = "id" if "id" in recipes.columns else recipes.columns[0]
    recipes[id_col] = recipes[id_col].astype(str)
    recipes = recipes.set_index(id_col, drop=False)

    item_rows = []
    for raw_i, iid in item_map.items():
        title, categories, description = "", [], ""
        source_fields = []
        if raw_i in recipes.index:
            r = recipes.loc[raw_i]
            if isinstance(r, pd.DataFrame):
                r = r.iloc[0]
            if "name" in r and pd.notna(r["name"]):
                title = str(r["name"])[:200]
                source_fields.append("name")
            if "tags" in r and pd.notna(r["tags"]):
                try:
                    tags = ast.literal_eval(str(r["tags"])) if str(r["tags"]).startswith("[") else [str(r["tags"])]
                except Exception:
                    tags = [str(r["tags"])]
                categories = [str(t) for t in tags][:20]
                source_fields.append("tags")
            parts = []
            if "description" in r and pd.notna(r["description"]):
                parts.append(str(r["description"]))
                source_fields.append("description")
            if "ingredients" in r and pd.notna(r["ingredients"]):
                parts.append(str(r["ingredients"])[:300])
                source_fields.append("ingredients")
            description = " ".join(parts)[:512]
        text_ok = bool(title or description or categories)
        item_rows.append({
            "raw_item_id": raw_i,
            "item_id": iid,
            "title": title,
            "categories": categories,
            "description": description,
            "attributes": {},
            "text_available": text_ok,
            "source_fields": source_fields,
            "cold_start": False,
        })
    append_jsonl(out / "metadata" / "items.jsonl", item_rows)
    append_jsonl(out / "metadata" / "users.jsonl", [
        {"raw_user_id": u, "user_id": uid, "cold_start": False, "text_available": False}
        for u, uid in user_map.items()
    ])

    gpaths = export_graphs(all_df, out / "graphs", n_user, n_item, cfg["feat_dim"], seed=seed)
    file_paths.update(gpaths)

    paper = reg["paper_stats"]["food"]
    stats = {
        "n_user": n_user,
        "n_item": n_item,
        "split_counts": {k: int(v) for k, v in all_df["split"].value_counts().to_dict().items()},
        "text_coverage_items": float(np.mean([1 if r["text_available"] else 0 for r in item_rows])),
        "paper_target": paper,
        "discrepancy": {
            "users_delta": n_user - paper["users"],
            "items_delta": n_item - paper["items"],
        },
    }
    write_json(out / "validation_report.json", {"stats": stats, "pending_validate": True})
    manifest = build_manifest(
        dataset="food",
        version=VERSION,
        data_status=cfg["status_default"],
        split_seed=seed,
        source_urls=["https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions"],
        license_notes=cfg["license"],
        filters=cfg["filters"],
        graph_contract="train_encode_only; eval_context uses train edges; GT in parquet",
        stats=stats,
    )
    finalize_manifest(out / "manifest.json", manifest, file_paths)
    print("Prepared Food at", out)
    print("Stats:", stats)


if __name__ == "__main__":
    main()
