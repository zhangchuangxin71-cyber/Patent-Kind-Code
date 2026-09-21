#!/usr/bin/env python
"""Prepare KuaiRec strict splits + caption text metadata."""
from __future__ import annotations

import ast
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.filters import dedupe_keep_latest  # noqa: E402
from common.graph_export import export_graphs  # noqa: E402
from common.io import append_jsonl, data_strict_root, ensure_dir, load_registry, write_json  # noqa: E402
from common.manifest import build_manifest, finalize_manifest  # noqa: E402
from common.mapping import apply_maps, build_maps  # noqa: E402
from common.splits import random_712_with_empty_ood  # noqa: E402

VERSION = "v1_strict"


def _find_raw(root: Path) -> Path:
    base = root / "raw" / "kuairec" / "zenodo_18164998"
    cands = [base, base / "KuaiRec", base / "KuaiRec 2.0"]
    # also discover nested folders
    if base.exists():
        cands.extend(sorted(base.glob("KuaiRec*")))
    for c in cands:
        if (c / "data" / "big_matrix.csv").exists():
            return c
        if (c / "big_matrix.csv").exists():
            return c
    raise FileNotFoundError("KuaiRec raw not found; run download_datasets.py --dataset kuairec")


def _data_dir(raw: Path) -> Path:
    if (raw / "data" / "big_matrix.csv").exists():
        return raw / "data"
    return raw


def _read_csv(path: Path, **kwargs) -> pd.DataFrame:
    """Robust CSV read for large/messy KuaiRec files."""
    try:
        return pd.read_csv(path, low_memory=False, **kwargs)
    except Exception:
        return pd.read_csv(path, engine="python", on_bad_lines="skip", **kwargs)


def load_caption(raw: Path) -> pd.DataFrame:
    # Prefer zip-bundled caption under data/, avoid corrupt standalone cache copies
    cands = list(raw.rglob("kuairec_caption_category.csv"))
    # put data/ paths first
    cands = sorted(cands, key=lambda p: (0 if "data" in p.parts else 1, len(str(p))))
    for p in cands:
        try:
            df = _read_csv(p)
            print(f"Loaded caption: {p} rows={len(df)}")
            return df
        except Exception as e:
            print(f"Skip caption {p}: {e}")
    return pd.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v2_exposure")
    ap.add_argument("--small_ood_ratio", type=float, default=0.2)
    cli = ap.parse_args()
    reg = load_registry()
    root = data_strict_root(reg)
    seed = int(reg["split_seed"])
    cfg = reg["datasets"]["kuairec"]
    raw = _find_raw(root)
    ddir = _data_dir(raw)
    print("Using raw:", raw)

    big = _read_csv(ddir / "big_matrix.csv")
    small = _read_csv(ddir / "small_matrix.csv")
    # normalize columns
    for df in (big, small):
        cols = {c.lower(): c for c in df.columns}
        # expected: user_id, video_id, watch_ratio, optional timestamp
    thr = float(cfg["filters"]["min_watch_ratio"])

    def pos(df, tag):
        u = "user_id" if "user_id" in df.columns else df.columns[0]
        i = "video_id" if "video_id" in df.columns else df.columns[1]
        w = "watch_ratio" if "watch_ratio" in df.columns else None
        out = pd.DataFrame({
            "raw_user_id": df[u].astype(str),
            "raw_item_id": df[i].astype(str),
            "value": df[w].astype(float) if w else 1.0,
        })
        if "timestamp" in df.columns:
            out["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        elif "time" in df.columns:
            out["timestamp"] = pd.to_datetime(df["time"], errors="coerce")
        else:
            out["timestamp"] = pd.NaT
        out = out[out["value"] >= thr]
        out["source_matrix"] = tag
        return out

    big_p = pos(big, "big")
    small_p = pos(small, "small")
    big_p = dedupe_keep_latest(big_p)
    small_p = dedupe_keep_latest(small_p)

    # force remove overlapping (u,i) from small
    big_keys = set(zip(big_p["raw_user_id"], big_p["raw_item_id"]))
    before = len(small_p)
    mask = [ (u, i) not in big_keys for u, i in zip(small_p["raw_user_id"], small_p["raw_item_id"]) ]
    small_p = small_p.loc[mask].reset_index(drop=True)
    overlap_removed = before - len(small_p)
    print(f"KuaiRec small overlap removed: {overlap_removed}")

    # Keep only big-matrix training interactions.  Validation and OOD are
    # both drawn from small-matrix exposure, so validation matches the OOD
    # universe without consuming the existing v1 benchmark.
    big_train = big_p.copy()
    big_train["split"] = "train"
    small_parts = []
    for uid, group in small_p.groupby("raw_user_id", sort=False):
        group = group.sort_values("timestamp", ascending=True)
        n_ood = max(1, int(round(len(group) * cli.small_ood_ratio))) if len(group) >= 2 else 0
        if n_ood:
            val = group.iloc[:-n_ood].copy()
            ood = group.iloc[-n_ood:].copy()
            val["split"] = "val"
            ood["split"] = "ood_test"
            small_parts.extend([val, ood])
        else:
            group["split"] = "val"
            small_parts.append(group)
    small_split = pd.concat(small_parts, ignore_index=True) if small_parts else small_p.iloc[:0].copy()
    all_df = pd.concat([big_train, small_split], ignore_index=True)

    user_map, item_map, node_map = build_maps(all_df)
    all_df = apply_maps(all_df, user_map, item_map)
    n_user, n_item = len(user_map), len(item_map)

    # cold-start flags: only in ood
    train_users = set(all_df.loc[all_df["split"] == "train", "raw_user_id"])
    train_items = set(all_df.loc[all_df["split"] == "train", "raw_item_id"])
    ood_users = set(all_df.loc[all_df["split"] == "ood_test", "raw_user_id"])
    ood_items = set(all_df.loc[all_df["split"] == "ood_test", "raw_item_id"])
    cold_u = ood_users - train_users
    cold_i = ood_items - train_items

    out = root / "processed" / "kuairec" / cli.version
    for sub in ["interactions", "metadata", "mappings", "graphs", "features"]:
        ensure_dir(out / sub)

    # The semantic prior is keyed by the canonical item mapping.  v2 keeps
    # the same union of raw entities as v1, so reuse the audited prior rather
    # than silently producing a dataset that cannot run semantic experiments.
    prior_src = root / "processed" / "kuairec" / "v1_strict" / "features"
    for name in ["semantic_prior.pt", "semantic_prior_shuffled.pt", "semantic_prior_hash.pt"]:
        src = prior_src / name
        if src.exists():
            shutil.copy2(src, out / "features" / name)

    file_paths = {}
    for sp in ["train", "val", "iid_test", "ood_test"]:
        p = out / "interactions" / f"{sp}.parquet"
        all_df[all_df["split"] == sp].to_parquet(p, index=False)
        file_paths[f"interactions/{sp}.parquet"] = str(p)

    write_json(out / "mappings" / "user_id_map.json", user_map)
    write_json(out / "mappings" / "item_id_map.json", item_map)
    write_json(out / "mappings" / "node_id_map.json", node_map)

    # item text from caption / categories
    cap = load_caption(raw)
    cat_path = ddir / "item_categories.csv"
    cats = _read_csv(cat_path) if cat_path.exists() else pd.DataFrame()
    item_rows = []
    cap_map = {}
    if len(cap):
        # flexible column names
        id_col = "video_id" if "video_id" in cap.columns else cap.columns[0]
        for _, r in cap.iterrows():
            cap_map[str(r[id_col])] = r.to_dict()
    cat_map = {}
    if len(cats):
        id_col = "video_id" if "video_id" in cats.columns else cats.columns[0]
        for _, r in cats.iterrows():
            cat_map[str(r[id_col])] = r.to_dict()

    for raw_i, iid in item_map.items():
        title = ""
        categories = []
        description = ""
        source_fields = []
        meta = cap_map.get(raw_i, {})
        if meta:
            for key in ["caption", "manual_cover_text", "ocr_text", "asr_text", "category"]:
                if key in meta and pd.notna(meta[key]) and str(meta[key]).strip():
                    if key == "caption" or not title:
                        if key in ("caption", "manual_cover_text"):
                            title = str(meta[key])[:200] if not title else title
                        if key == "category":
                            try:
                                categories = ast.literal_eval(str(meta[key])) if str(meta[key]).startswith("[") else [str(meta[key])]
                            except Exception:
                                categories = [str(meta[key])]
                        else:
                            description = (description + " " + str(meta[key])).strip()[:512]
                        source_fields.append(key)
        cmeta = cat_map.get(raw_i, {})
        if cmeta and not categories:
            for key in ["feat", "categories", "category"]:
                if key in cmeta and pd.notna(cmeta[key]):
                    try:
                        categories = ast.literal_eval(str(cmeta[key])) if str(cmeta[key]).startswith("[") else [str(cmeta[key])]
                    except Exception:
                        categories = [str(cmeta[key])]
                    source_fields.append(f"item_categories.{key}")
                    break
        if not title and categories:
            title = " | ".join(map(str, categories))[:200]
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
            "cold_start": raw_i in cold_i,
        })
    append_jsonl(out / "metadata" / "items.jsonl", item_rows)
    user_rows = [{
        "raw_user_id": u,
        "user_id": uid,
        "cold_start": u in cold_u,
        "text_available": False,
    } for u, uid in user_map.items()]
    append_jsonl(out / "metadata" / "users.jsonl", user_rows)

    gpaths = export_graphs(all_df, out / "graphs", n_user, n_item, cfg["feat_dim"], seed=seed)
    file_paths.update(gpaths)

    paper = reg["paper_stats"]["kuairec"]
    inter_train = int((all_df["split"] == "train").sum())
    stats = {
        "n_user": n_user,
        "n_item": n_item,
        "split_counts": all_df["split"].value_counts().to_dict(),
        "overlap_removed_from_small": overlap_removed,
        "cold_start_users": len(cold_u),
        "cold_start_items": len(cold_i),
        "cold_user_ratio_in_ood": float(len(cold_u) / max(1, len(ood_users))),
        "cold_item_ratio_in_ood": float(len(cold_i) / max(1, len(ood_items))),
        "text_coverage_items": float(np.mean([1 if r["text_available"] else 0 for r in item_rows])),
        "paper_target": paper,
        "discrepancy": {
            "users_delta": n_user - paper["users"],
            "items_delta": n_item - paper["items"],
            "note": "counts after watch_ratio>=2 and mapping union; interactions counted per split",
        },
        "train_interactions": inter_train,
    }
    write_json(out / "validation_report.json", {"stats": stats, "pending_validate": True})

    manifest = build_manifest(
        dataset="kuairec",
        version=cli.version,
        data_status=cfg["status_default"],
        split_seed=seed,
        source_urls=["https://zenodo.org/records/18164998"],
        license_notes=cfg["license"],
        filters=cfg["filters"],
        graph_contract="train_encode_only; eval_context uses train edges; GT in parquet",
        stats=stats,
    )
    finalize_manifest(out / "manifest.json", manifest, file_paths)
    print("Prepared KuaiRec at", out)
    print("Stats:", stats)
    print("Run: python scripts/data/validate_dataset.py --dataset kuairec")


if __name__ == "__main__":
    main()
