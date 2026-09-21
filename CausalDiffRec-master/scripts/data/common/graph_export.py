"""Export DGL graphs under the graph contract."""
from __future__ import annotations

from pathlib import Path

import dgl
import numpy as np
import pandas as pd
import torch


def _edges_from_split(df: pd.DataFrame, splits) -> tuple:
    sub = df[df["split"].isin(splits)]
    src = sub["user_node"].to_numpy()
    dst = sub["item_node"].to_numpy()
    u = np.concatenate([src, dst])
    v = np.concatenate([dst, src])
    return u, v


def build_feat(num_nodes: int, feat_dim: int, seed: int = 1024) -> torch.Tensor:
    rng = np.random.RandomState(seed)
    feat = rng.randn(num_nodes, feat_dim).astype(np.float32)
    feat = feat / (np.linalg.norm(feat, axis=1, keepdims=True) + 1e-8)
    return torch.from_numpy(feat)


def export_graphs(
    df: pd.DataFrame,
    out_dir: Path,
    n_user: int,
    n_item: int,
    feat_dim: int,
    seed: int = 1024,
) -> dict:
    """
    train.bin: train edges only (training encode graph).
    eval_context_*: train edges for MP; GT lives in parquet only.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    feat_dir = out_dir.parent / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)
    num_nodes = n_user + n_item
    feat = build_feat(num_nodes, feat_dim, seed=seed)
    torch.save({"feat": feat, "n_user": n_user, "n_item": n_item}, feat_dir / "node_feat.pt")

    def save(name: str, splits) -> str:
        u, v = _edges_from_split(df, splits)
        g = dgl.graph((u, v), num_nodes=num_nodes)
        g.ndata["feat"] = feat.clone()
        path = out_dir / name
        dgl.save_graphs(str(path), [g])
        return str(path)

    return {
        "train": save("train.bin", ["train"]),
        "train_plus_val": save("train_plus_val.bin", ["train", "val"]),
        "eval_context_iid": save("eval_context_iid.bin", ["train"]),
        "eval_context_ood": save("eval_context_ood.bin", ["train"]),
    }
