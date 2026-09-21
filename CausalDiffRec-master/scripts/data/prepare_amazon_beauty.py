#!/usr/bin/env python
"""Prepare Amazon Beauty 5-core for LSCI method validation (popularity OOD).

Source: McAuley SNAP Amazon — reviews_Beauty_5.json.gz + meta_Beauty.json.gz.
Join item text on asin (title/categories/description/brand).
Do NOT put reviewText into item metadata (anti-leakage).
"""
from __future__ import annotations

import ast
import gzip
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List

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
REVIEWS_NAME = "reviews_Beauty_5.json.gz"
META_NAME = "meta_Beauty.json.gz"
SOURCE_URL = "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/"


def _find_raw(root: Path) -> Path:
    cands = [
        root / "raw" / "amazon_beauty" / "mcauley",
        root / "raw" / "amazon_beauty",
    ]
    for base in cands:
        if (base / REVIEWS_NAME).exists() and (base / META_NAME).exists():
            return base
    raise FileNotFoundError(
        f"Missing {REVIEWS_NAME} / {META_NAME} under data_strict/raw/amazon_beauty/"
    )


def _parse_json_line(line: str) -> Dict[str, Any]:
    line = line.strip()
    if not line:
        return {}
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        # older McAuley meta files sometimes use python literals
        return ast.literal_eval(line)


def _iter_gz_json(path: Path) -> Iterator[Dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            o = _parse_json_line(line)
            if o:
                yield o


def apply_frequency_filter(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    mode = cfg.get("filter_mode", "iterative_kcore")
    mu = int(cfg["min_user_inter"])
    mi = int(cfg["min_item_inter"])
    if mode == "sequential_user_then_item":
        return sequential_user_then_item(df, min_user=mu, min_item=mi)
    max_rounds = int(cfg.get("max_rounds", 200))
    return iterative_kcore(df, min_user=mu, min_item=mi, max_rounds=max_rounds)


def maybe_subsample_users_for_dense_adj(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    max_nodes = int(cfg.get("max_nodes") or 0)
    if max_nodes <= 0:
        return df
    n_user = df["raw_user_id"].nunique()
    n_item = df["raw_item_id"].nunique()
    if n_user + n_item <= max_nodes:
        print(f"nodes={n_user + n_item} <= max_nodes={max_nodes}; no subsample")
        return df
    keep_users = max_nodes - n_item
    if keep_users < 1000:
        raise RuntimeError(f"max_nodes={max_nodes} too small for n_item={n_item}")
    rng = np.random.RandomState(int(cfg.get("subsample_seed", 1024)))
    users = df["raw_user_id"].unique()
    if keep_users >= len(users):
        return df
    chosen = set(rng.choice(users, size=keep_users, replace=False).tolist())
    out = df[df["raw_user_id"].isin(chosen)].copy()
    out = iterative_kcore(
        out,
        min_user=int(cfg.get("min_user_inter", 1)),
        min_item=int(cfg.get("min_item_inter", 1)),
        max_rounds=50,
    )
    print(
        f"subsample users {n_user}->{out['raw_user_id'].nunique()} "
        f"items {n_item}->{out['raw_item_id'].nunique()} "
        f"edges {len(df)}->{len(out)} (max_nodes={max_nodes})"
    )
    return out


def load_reviews(path: Path, min_rating: float) -> pd.DataFrame:
    rows = []
    n = 0
    for o in _iter_gz_json(path):
        n += 1
        uid = o.get("reviewerID") or o.get("user_id")
        iid = o.get("asin") or o.get("item_id")
        rating = o.get("overall") or o.get("rating")
        ts = o.get("unixReviewTime") or o.get("timestamp") or 0
        if uid is None or iid is None or rating is None:
            continue
        try:
            rating = float(rating)
            ts = float(ts)
        except (TypeError, ValueError):
            continue
        if rating < min_rating:
            continue
        rows.append({
            "raw_user_id": str(uid),
            "raw_item_id": str(iid),
            "value": rating,
            "timestamp": ts,
        })
        if n % 500_000 == 0:
            print(f"  reviews scanned={n:,} kept={len(rows):,}", flush=True)
    print(f"  reviews scanned={n:,} kept={len(rows):,}", flush=True)
    return pd.DataFrame(rows)


def _flatten_categories(raw) -> List[str]:
    if raw is None:
        return []
    out: List[str] = []
    if isinstance(raw, str):
        return [c.strip() for c in raw.split(",") if c.strip()][:30]
    if isinstance(raw, list):
        for x in raw:
            if isinstance(x, list):
                out.extend(str(t).strip() for t in x if str(t).strip())
            elif str(x).strip():
                out.append(str(x).strip())
    # unique preserve order
    seen = set()
    uniq = []
    for c in out:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
        if len(uniq) >= 30:
            break
    return uniq


def _as_text(v) -> str:
    if v is None:
        return ""
    if isinstance(v, list):
        return " ".join(str(x) for x in v if x).strip()
    return str(v).strip()


def load_meta(path: Path) -> pd.DataFrame:
    rows = []
    n = 0
    for o in _iter_gz_json(path):
        n += 1
        asin = o.get("asin")
        if not asin:
            continue
        rows.append({
            "raw_item_id": str(asin),
            "title": _as_text(o.get("title"))[:200],
            "description": _as_text(o.get("description"))[:800],
            "brand": _as_text(o.get("brand"))[:80],
            "categories": _flatten_categories(o.get("categories")),
        })
        if n % 200_000 == 0:
            print(f"  meta scanned={n:,}", flush=True)
    print(f"  meta scanned={n:,} rows={len(rows):,}", flush=True)
    df = pd.DataFrame(rows)
    # one row per asin
    return df.drop_duplicates("raw_item_id", keep="first")


def main():
    reg = load_registry()
    root = data_strict_root(reg)
    seed = int(reg["split_seed"])
    cfg = reg["datasets"]["amazon_beauty"]
    raw = _find_raw(root)
    print("Using raw:", raw)

    interim = root / "interim" / "amazon_beauty" / "pos_deduped.parquet"
    if interim.exists():
        print("Loading interim:", interim)
        inter = pd.read_parquet(interim)
    else:
        print("Loading reviews...")
        inter = load_reviews(raw / REVIEWS_NAME, float(cfg["filters"]["min_rating"]))
        print("After rating filter:", len(inter))
        inter = dedupe_keep_latest(inter)
        print(
            "After dedupe:",
            len(inter),
            "users",
            inter["raw_user_id"].nunique(),
            "items",
            inter["raw_item_id"].nunique(),
        )
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
    )
    inter = maybe_subsample_users_for_dense_adj(inter, cfg["filters"])
    print(
        "After max_nodes fit:",
        len(inter),
        "users",
        inter["raw_user_id"].nunique(),
        "items",
        inter["raw_item_id"].nunique(),
        "nodes",
        inter["raw_user_id"].nunique() + inter["raw_item_id"].nunique(),
    )
    if inter.empty:
        raise RuntimeError("过滤后为空")

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

    out = root / "processed" / "amazon_beauty" / VERSION
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

    print("Loading meta for text...")
    meta = load_meta(raw / META_NAME).set_index("raw_item_id", drop=False)
    item_rows = []
    for raw_i, iid in item_map.items():
        title, categories, description, brand = "", [], "", ""
        source_fields = []
        attributes = {}
        if raw_i in meta.index:
            r = meta.loc[raw_i]
            if isinstance(r, pd.DataFrame):
                r = r.iloc[0]
            title = str(r.get("title") or "").strip()[:200]
            if title:
                source_fields.append("title")
            categories = list(r.get("categories") or [])[:30]
            if categories:
                source_fields.append("categories")
            description = str(r.get("description") or "").strip()[:512]
            if description:
                source_fields.append("description")
            brand = str(r.get("brand") or "").strip()[:80]
            if brand:
                attributes["brand"] = brand
                source_fields.append("brand")
        text_ok = bool(title or description or categories or brand)
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
    append_jsonl(
        out / "metadata" / "users.jsonl",
        [
            {
                "raw_user_id": u,
                "user_id": uid,
                "cold_start": u in cold_u,
                "text_available": False,
            }
            for u, uid in user_map.items()
        ],
    )

    gpaths = export_graphs(all_df, out / "graphs", n_user, n_item, cfg["feat_dim"], seed=seed)
    file_paths.update(gpaths)

    stats = {
        "n_user": n_user,
        "n_item": n_item,
        "n_interactions": int(len(all_df)),
        "split_counts": {k: int(v) for k, v in all_df["split"].value_counts().to_dict().items()},
        "ood_stats": ood_stats,
        "cold_start_users": len(cold_u),
        "cold_start_items": len(cold_i),
        "text_coverage_items": float(np.mean([1 if r["text_available"] else 0 for r in item_rows])),
        "source_raw": str(raw),
        "filters_applied": cfg["filters"],
        "note": "Amazon Beauty 5-core for LSCI method validation; rich item text; not CausalDiffRec Table-1.",
    }
    write_json(out / "validation_report.json", {"stats": stats, "pending_validate": True})
    manifest = build_manifest(
        dataset="amazon_beauty",
        version=VERSION,
        data_status=cfg["status_default"],
        split_seed=seed,
        source_urls=[
            SOURCE_URL + REVIEWS_NAME,
            SOURCE_URL + META_NAME,
        ],
        license_notes=cfg["license"],
        filters=cfg["filters"],
        graph_contract="train_encode_only; eval_context uses train edges; GT in parquet",
        stats=stats,
    )
    finalize_manifest(out / "manifest.json", manifest, file_paths)
    print("Prepared Amazon Beauty at", out)
    print("Stats:", stats)
    print("Run: python scripts/data/validate_dataset.py --dataset amazon_beauty")


if __name__ == "__main__":
    main()
