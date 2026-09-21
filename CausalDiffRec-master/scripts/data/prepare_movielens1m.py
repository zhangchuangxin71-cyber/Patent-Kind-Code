#!/usr/bin/env python
"""Prepare MovieLens-1M for LSCI method validation (popularity OOD + item text).

Source: GroupLens ml-1m.zip (ratings.dat + movies.dat).
MovieID aligns with title/genres — suitable for semantic prior.
"""
from __future__ import annotations

import sys
from pathlib import Path

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
SOURCE_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"


def _find_ml1m_dir(root: Path) -> Path:
    cands = [
        root / "raw" / "movielens1m" / "grouplens" / "ml-1m",
        root / "raw" / "movielens1m" / "ml-1m",
    ]
    for p in cands:
        if (p / "ratings.dat").exists() and (p / "movies.dat").exists():
            return p
    hits = list((root / "raw" / "movielens1m").rglob("ratings.dat")) if (root / "raw" / "movielens1m").exists() else []
    for r in hits:
        if (r.parent / "movies.dat").exists():
            return r.parent
    raise FileNotFoundError(
        "Missing ml-1m ratings.dat/movies.dat under data_strict/raw/movielens1m/"
    )


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
    n = n_user + n_item
    if n <= max_nodes:
        print(f"nodes={n} <= max_nodes={max_nodes}; no subsample")
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
    mi = int(cfg.get("min_item_inter", 1))
    mu = int(cfg.get("min_user_inter", 1))
    out = iterative_kcore(out, min_user=mu, min_item=mi, max_rounds=50)
    print(
        f"subsample users {n_user}->{out['raw_user_id'].nunique()} "
        f"items {n_item}->{out['raw_item_id'].nunique()} "
        f"edges {len(df)}->{len(out)} (max_nodes={max_nodes})"
    )
    return out


def load_ratings(path: Path, min_rating: float) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=["raw_user_id", "raw_item_id", "value", "timestamp"],
        encoding="latin-1",
    )
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["raw_user_id", "raw_item_id", "value", "timestamp"])
    df = df[df["value"] >= min_rating].copy()
    df["raw_user_id"] = df["raw_user_id"].astype(str)
    df["raw_item_id"] = df["raw_item_id"].astype(str)
    return df.reset_index(drop=True)


def load_movies(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=["raw_item_id", "title", "genres"],
        encoding="latin-1",
    )
    df["raw_item_id"] = df["raw_item_id"].astype(str)
    df["title"] = df["title"].fillna("").astype(str)
    df["genres"] = df["genres"].fillna("").astype(str)
    return df


def main():
    reg = load_registry()
    root = data_strict_root(reg)
    seed = int(reg["split_seed"])
    cfg = reg["datasets"]["movielens1m"]
    raw_dir = _find_ml1m_dir(root)
    print("Using raw:", raw_dir)

    interim = root / "interim" / "movielens1m" / "pos_deduped.parquet"
    if interim.exists():
        print("Loading interim parquet:", interim)
        inter = pd.read_parquet(interim)
    else:
        print("Loading ratings...")
        inter = load_ratings(raw_dir / "ratings.dat", float(cfg["filters"]["min_rating"]))
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

    out = root / "processed" / "movielens1m" / VERSION
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

    movies = load_movies(raw_dir / "movies.dat").set_index("raw_item_id", drop=False)
    item_rows = []
    for raw_i, iid in item_map.items():
        title, categories, source_fields = "", [], []
        if raw_i in movies.index:
            r = movies.loc[raw_i]
            if isinstance(r, pd.DataFrame):
                r = r.iloc[0]
            title = str(r.get("title") or "").strip()[:200]
            if title:
                source_fields.append("title")
            graw = str(r.get("genres") or "").strip()
            if graw and graw != "(no genres listed)":
                categories = [g for g in graw.split("|") if g.strip()][:20]
                if categories:
                    source_fields.append("genres")
        text_ok = bool(title or categories)
        item_rows.append({
            "raw_item_id": raw_i,
            "item_id": iid,
            "title": title,
            "categories": categories,
            "description": "",
            "attributes": {},
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
        "source_raw": str(raw_dir),
        "filters_applied": cfg["filters"],
        "note": "MovieLens-1M for LSCI method validation (popularity OOD); not CausalDiffRec Table-1.",
    }
    write_json(out / "validation_report.json", {"stats": stats, "pending_validate": True})
    manifest = build_manifest(
        dataset="movielens1m",
        version=VERSION,
        data_status=cfg["status_default"],
        split_seed=seed,
        source_urls=[SOURCE_URL],
        license_notes=cfg["license"],
        filters=cfg["filters"],
        graph_contract="train_encode_only; eval_context uses train edges; GT in parquet",
        stats=stats,
    )
    finalize_manifest(out / "manifest.json", manifest, file_paths)
    print("Prepared MovieLens-1M at", out)
    print("Stats:", stats)
    print("Run: python scripts/data/validate_dataset.py --dataset movielens1m")


if __name__ == "__main__":
    main()
