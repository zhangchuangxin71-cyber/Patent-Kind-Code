"""Load data_strict/processed/<ds>/v1_strict for training.

Graph contract:
  - Train/validation encode graph: graphs/train.bin (train edges only)
  - Test encode graph: graphs/eval_context_*.bin (train/context edges only)
  - Validation GT: interactions/val.parquet
  - Test GT: interactions/{ood,iid}_test.parquet
  - Ground-truth interactions are never added to an encode graph
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import dgl
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch


def resolve_strict_root(dataset: str, data_root: Optional[str] = None) -> Path:
    if data_root:
        root = Path(data_root)
    else:
        root = Path("data_strict") / "processed" / dataset / "v1_strict"
    ready = root / "READY"
    if not ready.exists():
        raise FileNotFoundError(
            f"Strict dataset not READY: missing {ready}. "
            f"Run prepare + validate for '{dataset}' first."
        )
    return root


def load_manifest(root: Path) -> dict:
    with open(root / "manifest.json", encoding="utf-8") as f:
        return json.load(f)


def load_node_counts(root: Path) -> Tuple[int, int]:
    pt = root / "features" / "node_feat.pt"
    if pt.exists():
        obj = torch.load(pt, map_location="cpu")
        return int(obj["n_user"]), int(obj["n_item"])
    man = load_manifest(root)
    stats = man.get("stats") or {}
    return int(stats["n_user"]), int(stats["n_item"])


def load_strict_datasets(
    dataset: str,
    data_root: Optional[str] = None,
    eval_split: str = "ood",
    load_test_gt: bool = True,
) -> Tuple[Dict, dict]:
    """Return (datasets_dict, meta).

    datasets keys:
      train/test/val        — encode graphs (never contain their GT edges)
      val_origin_inter      — validation GT dict
      val_user_set          — validation users
      test_origin_inter     — selected OOD/IID test GT dict
      test_user_set         — test users

    ``origin_inter`` and ``user_set`` remain test aliases for compatibility.
    """
    root = resolve_strict_root(dataset, data_root)
    if eval_split not in ("ood", "iid"):
        raise ValueError(f"eval_split must be ood|iid, got {eval_split}")

    n_user, n_item = load_node_counts(root)
    train_g = dgl.load_graphs(str(root / "graphs" / "train.bin"))[0][0]
    ctx_name = f"eval_context_{eval_split}.bin"
    test_g = dgl.load_graphs(str(root / "graphs" / ctx_name))[0][0]

    val_gt_path = root / "interactions" / "val.parquet"
    val_origin_inter, val_user_set = origin_inter_from_parquet(val_gt_path, n_user=n_user)
    gt_name = "ood_test.parquet" if eval_split == "ood" else "iid_test.parquet"
    test_gt_path = root / "interactions" / gt_name
    if load_test_gt:
        test_origin_inter, test_user_set = origin_inter_from_parquet(
            test_gt_path, n_user=n_user,
        )
    else:
        test_origin_inter, test_user_set = None, None

    man = load_manifest(root)
    meta = {
        "data_version": man.get("version", "v1_strict"),
        "data_status": man.get("data_status", "rule-equivalent"),
        "data_root": str(root),
        "eval_split": eval_split,
        "n_user": n_user,
        "n_item": n_item,
        "val_gt_path": str(val_gt_path),
        "val_gt_users": len(val_user_set),
        "val_gt_interactions": sum(len(v) for v in val_origin_inter.values()),
        "gt_path": str(test_gt_path),
        "test_gt_loaded": bool(load_test_gt),
        "gt_users": len(test_user_set) if test_user_set is not None else None,
        "gt_interactions": (
            sum(len(v) for v in test_origin_inter.values())
            if test_origin_inter is not None else None
        ),
        "manifest_sha256": (man.get("file_hashes") or {}).get("manifest.json"),
    }
    datasets = {
        "train": train_g,
        "val": train_g,
        "test": test_g,
        "val_origin_inter": val_origin_inter,
        "val_user_set": val_user_set,
        "strict": True,
    }
    if load_test_gt:
        datasets.update({
            "test_origin_inter": test_origin_inter,
            "test_user_set": test_user_set,
            "origin_inter": test_origin_inter,
            "user_set": test_user_set,
        })
    print(
        f"[strict] {dataset} root={root} n_user={n_user} n_item={n_item} "
        f"val_users={meta['val_gt_users']} val_inter={meta['val_gt_interactions']} "
        f"eval={eval_split} test_gt_loaded={load_test_gt} "
        f"test_users={meta['gt_users']} test_inter={meta['gt_interactions']}"
    )
    return datasets, meta


def origin_inter_from_parquet(path: Path, n_user: int) -> Tuple[dict, set]:
    """Build GT map: user_node -> {str(item_node): 1.0} (absolute node ids)."""
    df = pd.read_parquet(path)
    if "user_node" in df.columns and "item_node" in df.columns:
        u_col, i_col = "user_node", "item_node"
    elif "user_id" in df.columns and "item_id" in df.columns:
        # item_id is 0-based; convert to absolute node id
        df = df.copy()
        df["user_node"] = df["user_id"].astype(int)
        df["item_node"] = df["item_id"].astype(int) + int(n_user)
        u_col, i_col = "user_node", "item_node"
    else:
        raise KeyError(f"{path} missing user_node/item_node or user_id/item_id")

    user_interactions = {}
    for u, g in df.groupby(u_col, sort=False):
        uid = int(u)
        user_interactions[uid] = {str(int(i)): 1.0 for i in g[i_col].tolist()}
    return user_interactions, set(user_interactions.keys())


def user_item_matrix_from_graph(graph, n_user: int, n_item: int) -> Tuple[sp.csr_matrix, int, int]:
    """Build (n_user, n_item) CSR from bipartite edges; works with undirected strict graphs."""
    src, dst = graph.edges()
    src = src.cpu().numpy()
    dst = dst.cpu().numpy()
    mask = (src < n_user) & (dst >= n_user)
    row = src[mask].astype(np.int64)
    col = (dst[mask] - n_user).astype(np.int64)
    if len(row) == 0:
        raise RuntimeError("No user→item edges found; check n_user and graph orientation")
    data = np.ones(len(row), dtype=np.float32)
    mat = sp.csr_matrix((data, (row, col)), shape=(n_user, n_item), dtype=np.float32)
    print(f"[strict] interaction matrix: users={n_user} items={n_item} nnz={mat.nnz}")
    return mat, n_user, n_item
