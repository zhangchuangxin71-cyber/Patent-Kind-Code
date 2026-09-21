"""ID mapping: users first, then items, contiguous."""
from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd


def build_maps(
    df: pd.DataFrame,
    user_col: str = "raw_user_id",
    item_col: str = "raw_item_id",
) -> Tuple[Dict, Dict, Dict]:
    users = sorted(df[user_col].astype(str).unique().tolist())
    items = sorted(df[item_col].astype(str).unique().tolist())
    user_map = {u: i for i, u in enumerate(users)}
    item_map = {it: i for i, it in enumerate(items)}
    n_user = len(user_map)
    node_map = {f"u:{u}": user_map[u] for u in users}
    node_map.update({f"i:{it}": n_user + item_map[it] for it in items})
    return user_map, item_map, node_map


def apply_maps(
    df: pd.DataFrame,
    user_map: Dict,
    item_map: Dict,
    user_col: str = "raw_user_id",
    item_col: str = "raw_item_id",
) -> pd.DataFrame:
    out = df.copy()
    out["user_id"] = out[user_col].astype(str).map(user_map)
    out["item_id"] = out[item_col].astype(str).map(item_map)
    n_user = len(user_map)
    out["item_node"] = out["item_id"] + n_user
    out["user_node"] = out["user_id"]
    return out
