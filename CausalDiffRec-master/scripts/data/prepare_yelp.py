#!/usr/bin/env python
"""Prepare Yelp2018-style strict popularity-OOD splits + business text metadata.

Uses Kaggle/official Yelp Open Dataset JSON (rule-equivalent to paper 2018 snapshot).
Does NOT put review text into item descriptions (anti-leakage).
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.filters import dedupe_keep_latest, iterative_kcore, sequential_user_then_item  # noqa: E402
from common.graph_export import export_graphs  # noqa: E402
from common.io import append_jsonl, data_strict_root, ensure_dir, load_registry, write_json  # noqa: E402
from common.manifest import build_manifest, finalize_manifest  # noqa: E402
from common.mapping import apply_maps, build_maps  # noqa: E402
from common.splits import popularity_uniform_ood  # noqa: E402

VERSION = "v1_strict"
REVIEW_NAME = "yelp_academic_dataset_review.json"
BUSINESS_NAME = "yelp_academic_dataset_business.json"


def _find_raw(root: Path) -> Path:
    cands = [
        root / "raw" / "yelp2018" / "kaggle_yelp_dataset",
        root / "raw" / "yelp2018" / "official",
        root / "raw" / "yelp2018" / "pending",
    ]
    for base in cands:
        if not base.exists():
            continue
        hits_r = list(base.rglob(REVIEW_NAME))
        hits_b = list(base.rglob(BUSINESS_NAME))
        if hits_r and hits_b:
            return hits_r[0].parent
    raise FileNotFoundError(
        "Missing Yelp JSON. Run:\n"
        "  python scripts/data/download_datasets.py --dataset yelp2018\n"
        f"or place {REVIEW_NAME} + {BUSINESS_NAME} under data_strict/raw/yelp2018/"
    )


def _iter_review_chunks(path: Path, chunksize: int = 200_000) -> Iterator[pd.DataFrame]:
    """Stream line-delimited review JSON; keep only needed columns."""
    usecols = ["user_id", "business_id", "stars", "date"]
    reader = pd.read_json(
        path,
        lines=True,
        chunksize=chunksize,
        dtype={"stars": "float32"},
    )
    for chunk in reader:
        miss = [c for c in usecols if c not in chunk.columns]
        if miss:
            raise KeyError(f"review missing columns {miss}")
        yield chunk[usecols]


def load_reviews(path: Path, min_rating: float) -> pd.DataFrame:
    parts = []
    n_raw = 0
    for chunk in _iter_review_chunks(path):
        n_raw += len(chunk)
        chunk = chunk.rename(
            columns={
                "user_id": "raw_user_id",
                "business_id": "raw_item_id",
                "stars": "value",
                "date": "timestamp",
            }
        )
        chunk["value"] = pd.to_numeric(chunk["value"], errors="coerce")
        chunk["timestamp"] = pd.to_datetime(chunk["timestamp"], errors="coerce")
        chunk = chunk.dropna(subset=["raw_user_id", "raw_item_id", "value", "timestamp"])
        chunk = chunk[chunk["value"] >= min_rating]
        chunk["raw_user_id"] = chunk["raw_user_id"].astype(str)
        chunk["raw_item_id"] = chunk["raw_item_id"].astype(str)
        parts.append(chunk)
        print(f"  reviews scanned={n_raw:,} kept_so_far={sum(len(p) for p in parts):,}", flush=True)
    if not parts:
        return pd.DataFrame(columns=["raw_user_id", "raw_item_id", "value", "timestamp"])
    return pd.concat(parts, ignore_index=True)


def load_businesses(path: Path) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            rows.append({
                "raw_item_id": str(o.get("business_id", "")),
                "name": o.get("name") or "",
                "categories": o.get("categories") or "",
                "attributes": o.get("attributes") or {},
                "city": o.get("city") or "",
                "state": o.get("state") or "",
            })
    return pd.DataFrame(rows)


def _parse_categories(raw) -> list:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return []
    s = str(raw).strip()
    if not s:
        return []
    if s.startswith("["):
        try:
            v = ast.literal_eval(s)
            return [str(x) for x in v][:30]
        except Exception:
            pass
    return [c.strip() for c in s.split(",") if c.strip()][:30]


def apply_frequency_filter(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    mode = cfg.get("filter_mode", "iterative_kcore")
    mu = int(cfg["min_user_inter"])
    mi = int(cfg["min_item_inter"])
    if mode == "sequential_user_then_item":
        return sequential_user_then_item(df, min_user=mu, min_item=mi)
    max_rounds = int(cfg.get("max_rounds", 200))
    return iterative_kcore(df, min_user=mu, min_item=mi, max_rounds=max_rounds)


def main():
    reg = load_registry()
    root = data_strict_root(reg)
    seed = int(reg["split_seed"])
    cfg = reg["datasets"]["yelp2018"]
    raw = _find_raw(root)
    review_p = raw / REVIEW_NAME
    biz_p = raw / BUSINESS_NAME
    print("Using raw:", raw)

    interim = root / "interim" / "yelp2018" / "pos_deduped.parquet"
    if interim.exists():
        print("Loading interim parquet:", interim)
        inter = pd.read_parquet(interim)
        print("After interim load:", len(inter), "users", inter["raw_user_id"].nunique(), "items", inter["raw_item_id"].nunique())
    else:
        print("Loading reviews (streaming)...")
        inter = load_reviews(review_p, float(cfg["filters"]["min_rating"]))
        print("After rating filter:", len(inter))
        inter = dedupe_keep_latest(inter)
        print("After dedupe:", len(inter), "users", inter["raw_user_id"].nunique(), "items", inter["raw_item_id"].nunique())
        ensure_dir(interim.parent)
        inter.to_parquet(interim, index=False)

    inter = apply_frequency_filter(inter, cfg["filters"])
    print(
        "After frequency filter:",
        len(inter),
        "users",
        inter["raw_user_id"].nunique(),
        "items",
        inter["raw_item_id"].nunique(),
        "mode",
        cfg["filters"].get("filter_mode", "iterative_kcore"),
    )
    if inter.empty:
        raise RuntimeError("过滤后为空；检查 filters 或运行 debug_yelp_counts.py")

    print("Popularity-uniform OOD split...")
    all_df, ood_stats = popularity_uniform_ood(inter, seed=seed, ood_ratio=0.2)
    user_map, item_map, node_map = build_maps(all_df)
    all_df = apply_maps(all_df, user_map, item_map)
    n_user, n_item = len(user_map), len(item_map)

    train_users = set(all_df.loc[all_df["split"] == "train", "raw_user_id"])
    train_items = set(all_df.loc[all_df["split"] == "train", "raw_item_id"])
    ood_users = set(all_df.loc[all_df["split"] == "ood_test", "raw_user_id"])
    ood_items = set(all_df.loc[all_df["split"] == "ood_test", "raw_item_id"])
    cold_u = ood_users - train_users
    cold_i = ood_items - train_items

    out = root / "processed" / "yelp2018" / VERSION
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

    print("Loading businesses for text...")
    biz = load_businesses(biz_p)
    biz = biz.set_index("raw_item_id", drop=False)

    item_rows = []
    for raw_i, iid in item_map.items():
        title, categories, description = "", [], ""
        attributes = {}
        source_fields = []
        if raw_i in biz.index:
            r = biz.loc[raw_i]
            if isinstance(r, pd.DataFrame):
                r = r.iloc[0]
            if pd.notna(r.get("name")) and str(r["name"]).strip():
                title = str(r["name"])[:200]
                source_fields.append("name")
            categories = _parse_categories(r.get("categories"))
            if categories:
                source_fields.append("categories")
            attrs = r.get("attributes")
            if isinstance(attrs, dict) and attrs:
                # keep compact static attrs only (no review text)
                attributes = {str(k): str(v)[:80] for k, v in list(attrs.items())[:20]}
                source_fields.append("attributes")
            city = r.get("city") or ""
            state = r.get("state") or ""
            loc = ", ".join([x for x in [str(city), str(state)] if x and x != "nan"])
            if loc:
                description = loc[:512]
                source_fields.append("city_state")
        text_ok = bool(title or description or categories)
        item_rows.append({
            "raw_item_id": raw_i,
            "item_id": iid,
            "title": title,
            "categories": categories,
            "description": description,
            "attributes": attributes,
            "text_available": text_ok,
            "source_fields": source_fields,
            "cold_start": raw_i in cold_i,
        })
    append_jsonl(out / "metadata" / "items.jsonl", item_rows)
    append_jsonl(out / "metadata" / "users.jsonl", [
        {
            "raw_user_id": u,
            "user_id": uid,
            "cold_start": u in cold_u,
            "text_available": False,
        }
        for u, uid in user_map.items()
    ])

    gpaths = export_graphs(all_df, out / "graphs", n_user, n_item, cfg["feat_dim"], seed=seed)
    file_paths.update(gpaths)

    paper = reg["paper_stats"]["yelp2018"]
    stats = {
        "n_user": n_user,
        "n_item": n_item,
        "n_interactions": int(len(all_df)),
        "split_counts": {k: int(v) for k, v in all_df["split"].value_counts().to_dict().items()},
        "ood_stats": ood_stats,
        "cold_start_users": len(cold_u),
        "cold_start_items": len(cold_i),
        "text_coverage_items": float(np.mean([1 if r["text_available"] else 0 for r in item_rows])),
        "paper_target": paper,
        "discrepancy": {
            "users_delta": n_user - paper["users"],
            "items_delta": n_item - paper["items"],
            "interactions_delta": int(len(all_df)) - paper["interactions"],
            "note": "Kaggle yelp-dataset is rule-equivalent Open Dataset; not guaranteed 2018 snapshot",
        },
        "source_raw": str(raw),
    }
    write_json(out / "validation_report.json", {"stats": stats, "pending_validate": True})
    manifest = build_manifest(
        dataset="yelp2018",
        version=VERSION,
        data_status=cfg["status_default"],
        split_seed=seed,
        source_urls=[
            "https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset",
            "https://www.yelp.com/dataset",
        ],
        license_notes=cfg["license"],
        filters=cfg["filters"],
        graph_contract="train_encode_only; eval_context uses train edges; GT in parquet",
        stats=stats,
    )
    finalize_manifest(out / "manifest.json", manifest, file_paths)
    print("Prepared Yelp at", out)
    print("Stats:", stats)
    print("Run: python scripts/data/validate_dataset.py --dataset yelp2018")


if __name__ == "__main__":
    main()
