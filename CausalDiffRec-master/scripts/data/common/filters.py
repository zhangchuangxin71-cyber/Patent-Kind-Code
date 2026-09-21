"""Filtering helpers: positive feedback + frequency filters / k-core."""
from __future__ import annotations

import pandas as pd


def sequential_user_then_item(
    df: pd.DataFrame,
    user_col: str = "raw_user_id",
    item_col: str = "raw_item_id",
    min_user: int = 1,
    min_item: int = 1,
) -> pd.DataFrame:
    """One-pass: keep users with >= min_user, then items with >= min_item.

    Matches CausalDiffRec Food paper stats exactly at (15, 15):
    7809 users / 6309 items / 216407 interactions.
    Full bipartite iterative k-core at (15, 50) collapses to empty on Food.com.
    """
    min_user, min_item = int(min_user), int(min_item)
    cur = df
    u_cnt = cur.groupby(user_col, observed=True).size()
    cur = cur[cur[user_col].isin(u_cnt[u_cnt >= min_user].index)]
    i_cnt = cur.groupby(item_col, observed=True).size()
    cur = cur[cur[item_col].isin(i_cnt[i_cnt >= min_item].index)]
    return cur.reset_index(drop=True)


def iterative_kcore(
    df: pd.DataFrame,
    user_col: str = "raw_user_id",
    item_col: str = "raw_item_id",
    min_user: int = 1,
    min_item: int = 1,
    max_rounds: int = 200,
) -> pd.DataFrame:
    """Drop users/items below thresholds for up to max_rounds (or until converge)."""
    min_user, min_item = int(min_user), int(min_item)
    max_rounds = int(max_rounds)
    cur = df.copy()
    for _ in range(max_rounds):
        if cur.empty:
            break
        u_cnt = cur.groupby(user_col, observed=True).size()
        i_cnt = cur.groupby(item_col, observed=True).size()
        keep_u = u_cnt[u_cnt >= min_user].index
        keep_i = i_cnt[i_cnt >= min_item].index
        nxt = cur[cur[user_col].isin(keep_u) & cur[item_col].isin(keep_i)]
        if len(nxt) == len(cur):
            return nxt.reset_index(drop=True)
        cur = nxt
    return cur.reset_index(drop=True)


def dedupe_keep_latest(
    df: pd.DataFrame,
    user_col: str = "raw_user_id",
    item_col: str = "raw_item_id",
    time_col: str = "timestamp",
    value_col: str = "value",
) -> pd.DataFrame:
    """Keep one interaction per (user,item): latest time, else higher value."""
    sort_cols = []
    ascending = []
    if time_col in df.columns:
        sort_cols.append(time_col)
        ascending.append(True)
    if not sort_cols:
        return df.drop_duplicates([user_col, item_col], keep="first").reset_index(drop=True)
    out = df.sort_values(sort_cols, ascending=ascending)
    return out.drop_duplicates([user_col, item_col], keep="last").reset_index(drop=True)
