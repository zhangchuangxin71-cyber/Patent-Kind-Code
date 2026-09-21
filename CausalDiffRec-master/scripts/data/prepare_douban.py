#!/usr/bin/env python
"""Prepare DeepGraph DoubanMovie for method validation (popularity OOD).

Source: DeepGraphLearning Douban.tar.gz (Song et al., WSDM'19 DGRec).
NOT aligned to CausalDiffRec Table-1 Douban stats — disclosed in manifest.

Item IDs in the release are remapped integers (0..N-1); no title/genres
are shipped. items.jsonl therefore has text_available=False unless an
external id→text map is provided later.
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
MOVIE_TSV = "douban_movie.tsv"
SOURCE_URL = "https://www.dropbox.com/s/u2ejjezjk08lz1o/Douban.tar.gz"


def _find_movie_tsv(root: Path) -> Path:
    cands = [
        root / "raw" / "douban" / "deepgraph" / "Douban" / "movie" / MOVIE_TSV,
        root / "raw" / "douban" / "pending" / "deepgraph_extract" / "Douban" / "movie" / MOVIE_TSV,
    ]
    for p in cands:
        if p.exists():
            return p
    hits = list((root / "raw" / "douban").rglob(MOVIE_TSV)) if (root / "raw" / "douban").exists() else []
    if hits:
        return hits[0]
    raise FileNotFoundError(
        f"Missing {MOVIE_TSV}. Expected under data_strict/raw/douban/ "
        f"(DeepGraph Douban.tar.gz extract)."
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
    """CausalDiffRec uses dense n×n adjacency; keep n_user+n_item <= max_nodes."""
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
        raise RuntimeError(
            f"max_nodes={max_nodes} too small for n_item={n_item} "
            f"(need max_nodes > n_item + 1000)"
        )
    rng = np.random.RandomState(int(cfg.get("subsample_seed", 1024)))
    users = df["raw_user_id"].unique()
    if keep_users >= len(users):
        return df
    chosen = set(rng.choice(users, size=keep_users, replace=False).tolist())
    out = df[df["raw_user_id"].isin(chosen)].copy()
    # drop items that lost all interactions; light re-core at min_item only
    mi = int(cfg.get("min_item_inter", 1))
    mu = int(cfg.get("min_user_inter", 1))
    out = iterative_kcore(out, min_user=mu, min_item=mi, max_rounds=50)
    print(
        f"subsample users {n_user}->{out['raw_user_id'].nunique()} "
        f"items {n_item}->{out['raw_item_id'].nunique()} "
        f"edges {len(df)}->{len(out)} (max_nodes={max_nodes})"
    )
    return out


def load_interactions(path: Path, min_rating: float) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    need = {"UserId", "ItemId", "Rating", "Timestamp"}
    miss = need - set(df.columns)
    if miss:
        raise KeyError(f"{path} missing columns {miss}")
    # DeepGraph: blank rating encoded as -1 (watched but unrated) — drop
    df = df[df["Rating"] >= min_rating].copy()
    df = df.rename(
        columns={
            "UserId": "raw_user_id",
            "ItemId": "raw_item_id",
            "Rating": "value",
            "Timestamp": "timestamp",
        }
    )
    df["raw_user_id"] = df["raw_user_id"].astype(str)
    df["raw_item_id"] = df["raw_item_id"].astype(str)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["raw_user_id", "raw_item_id", "value", "timestamp"])
    return df.reset_index(drop=True)


def main():
    reg = load_registry()
    root = data_strict_root(reg)
    seed = int(reg["split_seed"])
    cfg = reg["datasets"]["douban"]
    if cfg.get("status_default") == "blocked":
        raise RuntimeError(
            "douban status_default=blocked in dataset_registry.yaml; "
            "set to deepgraph-method-validation to proceed."
        )

    movie_p = _find_movie_tsv(root)
    print("Using raw:", movie_p)

    interim = root / "interim" / "douban" / "pos_deduped.parquet"
    if interim.exists():
        print("Loading interim parquet:", interim)
        inter = pd.read_parquet(interim)
        print(
            "After interim load:",
            len(inter),
            "users",
            inter["raw_user_id"].nunique(),
            "items",
            inter["raw_item_id"].nunique(),
        )
    else:
        print("Loading DeepGraph movie ratings...")
        inter = load_interactions(movie_p, float(cfg["filters"]["min_rating"]))
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
        "mode",
        cfg["filters"].get("filter_mode", "iterative_kcore"),
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
        raise RuntimeError("过滤后为空；检查 filters")

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

    out = root / "processed" / "douban" / VERSION
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

    # No shipped movie text in DeepGraph release (IDs remapped).
    item_rows = []
    for raw_i, iid in item_map.items():
        item_rows.append({
            "raw_item_id": raw_i,
            "item_id": iid,
            "title": "",
            "categories": [],
            "description": "",
            "attributes": {},
            "text_available": False,
            "source_fields": [],
            "cold_start": raw_i in cold_i,
            "text_note": "DeepGraph DoubanMovie has remapped ItemId only; no title/genres in release",
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

    paper = reg["paper_stats"]["douban"]
    stats = {
        "n_user": n_user,
        "n_item": n_item,
        "n_interactions": int(len(all_df)),
        "split_counts": {k: int(v) for k, v in all_df["split"].value_counts().to_dict().items()},
        "ood_stats": ood_stats,
        "cold_start_users": len(cold_u),
        "cold_start_items": len(cold_i),
        "text_coverage_items": 0.0,
        "paper_target": paper,
        "discrepancy": {
            "users_delta": n_user - paper["users"],
            "items_delta": n_item - paper["items"],
            "interactions_delta": int(len(all_df)) - paper["interactions"],
            "note": (
                "DeepGraph DoubanMovie for LSCI method validation; "
                "NOT CausalDiffRec Table-1 Douban; do not compare absolute metrics to paper. "
                "Users subsampled to fit dense adj (max_nodes) — see filters.max_nodes."
            ),
        },
        "source_raw": str(movie_p),
        "source_citation": "Song et al. WSDM'19 DGRec / DeepGraphLearning Douban.tar.gz",
        "filters_applied": cfg["filters"],
    }
    write_json(out / "validation_report.json", {"stats": stats, "pending_validate": True})
    manifest = build_manifest(
        dataset="douban",
        version=VERSION,
        data_status=cfg["status_default"],
        split_seed=seed,
        source_urls=[SOURCE_URL, "https://github.com/DeepGraphLearning/RecommenderSystems"],
        license_notes=cfg["license"],
        filters=cfg["filters"],
        graph_contract="train_encode_only; eval_context uses train edges; GT in parquet",
        stats=stats,
    )
    finalize_manifest(out / "manifest.json", manifest, file_paths)
    blocked = out / "BLOCKED.json"
    if blocked.exists():
        blocked.unlink()
    print("Prepared Douban (DeepGraph) at", out)
    print("Stats:", stats)
    print("NOTE: text_coverage_items=0 — LSCI semantic prior needs external movie text map.")
    print("Run: python scripts/data/validate_dataset.py --dataset douban")


if __name__ == "__main__":
    main()
