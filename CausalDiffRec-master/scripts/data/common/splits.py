"""Deterministic train/val/iid/ood splits."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd


def _assign_remainder_712(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Split rows into train/val/iid_test with ratio 7:1:2."""
    rng = np.random.RandomState(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    n = len(idx)
    n_train = int(n * 0.7)
    n_val = int(n * 0.1)
    labels = np.empty(n, dtype=object)
    labels[idx[:n_train]] = "train"
    labels[idx[n_train:n_train + n_val]] = "val"
    labels[idx[n_train + n_val:]] = "iid_test"
    out = df.copy()
    out["split"] = labels
    return out


def _stable_user_seed(seed: int, user_id: str) -> int:
    import zlib
    return seed + (zlib.adler32(str(user_id).encode("utf-8")) % 100000)


def temporal_ood_per_user(
    df: pd.DataFrame,
    seed: int,
    ood_ratio: float = 0.2,
    user_col: str = "raw_user_id",
    time_col: str = "timestamp",
) -> pd.DataFrame:
    """Per-user newest ood_ratio -> ood_test; remainder 7:1:2."""
    if df.empty:
        raise ValueError("temporal_ood_per_user: empty dataframe after filtering")
    parts = []
    for uid, g in df.groupby(user_col, sort=False):
        g = g.sort_values(time_col, ascending=False)
        n = len(g)
        n_ood = max(1, int(round(n * ood_ratio))) if n >= 2 else 0
        u_seed = _stable_user_seed(seed, str(uid))
        if n_ood == 0 or n_ood >= n:
            rem = _assign_remainder_712(g.copy(), u_seed)
            parts.append(rem)
            continue
        ood = g.iloc[:n_ood].copy()
        ood["split"] = "ood_test"
        rem = _assign_remainder_712(g.iloc[n_ood:].copy(), u_seed)
        parts.append(pd.concat([ood, rem], ignore_index=True))
    if not parts:
        raise ValueError("temporal_ood_per_user: no user groups produced")
    return pd.concat(parts, ignore_index=True)


def random_712_with_empty_ood(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    out = _assign_remainder_712(df, seed)
    return out


def popularity_uniform_ood(
    df: pd.DataFrame,
    seed: int,
    ood_ratio: float = 0.2,
    item_col: str = "raw_item_id",
    tol: float = 0.01,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Sample ~ood_ratio interactions as OOD with near-uniform item frequency.
    Deterministic algorithm disclosed in manifest stats.
    """
    rng = np.random.RandomState(seed)
    counts = df.groupby(item_col).size()
    median_c = float(np.median(counts.values))
    # per-item quota toward uniform: up to median (capped)
    per_item_cap = max(1, int(median_c))
    target_n = int(len(df) * ood_ratio)
    ood_idx = []
    # first pass: sample up to cap from each item
    for item, g in df.groupby(item_col, sort=False):
        take = min(len(g), per_item_cap)
        chosen = g.sample(n=take, random_state=rng.randint(0, 1 << 31 - 1)).index.tolist()
        ood_idx.extend(chosen)
    ood_idx = list(dict.fromkeys(ood_idx))
    # trim or top-up to target_n ± tol
    low = int(len(df) * (ood_ratio - tol))
    high = int(len(df) * (ood_ratio + tol))
    if len(ood_idx) > high:
        rng.shuffle(ood_idx)
        ood_idx = ood_idx[:target_n]
    elif len(ood_idx) < low:
        remain = df.index.difference(ood_idx)
        need = target_n - len(ood_idx)
        extra = rng.choice(remain.values, size=min(need, len(remain)), replace=False)
        ood_idx.extend(extra.tolist())

    ood_set = set(ood_idx)
    out = df.copy()
    out["split"] = "pending"
    out.loc[out.index.isin(ood_set), "split"] = "ood_test"
    rem = out[out["split"] != "ood_test"].copy()
    rem = _assign_remainder_712(rem, seed + 7)
    out.loc[rem.index, "split"] = rem["split"].values

    # stats
    def gini(x: np.ndarray) -> float:
        if len(x) == 0:
            return 0.0
        x = np.sort(x.astype(float))
        n = len(x)
        return float((2 * np.sum((np.arange(1, n + 1) * x)) / (n * x.sum()) - (n + 1) / n))

    train_items = out[out["split"] == "train"].groupby(item_col).size().values
    ood_items = out[out["split"] == "ood_test"].groupby(item_col).size().values
    stats = {
        "ood_ratio_actual": float((out["split"] == "ood_test").mean()),
        "gini_train": gini(train_items) if len(train_items) else None,
        "gini_ood": gini(ood_items) if len(ood_items) else None,
        "per_item_cap": per_item_cap,
        "median_item_count": median_c,
        "target_n": target_n,
        "ood_n": int((out["split"] == "ood_test").sum()),
    }
    return out, stats
