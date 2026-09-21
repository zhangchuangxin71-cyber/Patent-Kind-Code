#!/usr/bin/env python
"""Quick Food degree stats (background diagnostic)."""
import pandas as pd
from pathlib import Path

p = Path(__file__).resolve().parents[2] / "data_strict/raw/food/kaggle_food_com/RAW_interactions.csv"
print("reading", p)
df = pd.read_csv(p, low_memory=False)
df = df.rename(columns={"user_id": "u", "recipe_id": "i", "rating": "r", "date": "t"})
df["r"] = pd.to_numeric(df["r"], errors="coerce")
df["t"] = pd.to_datetime(df["t"], errors="coerce")
df = df.dropna(subset=["u", "i", "r", "t"])
df = df[df["r"] >= 4]
df = df.sort_values("t").drop_duplicates(["u", "i"], keep="last")
print("rows", len(df), "users", df["u"].nunique(), "items", df["i"].nunique())
us = df.groupby("u").size()
is_ = df.groupby("i").size()
print("users>=15", int((us >= 15).sum()), "items>=50", int((is_ >= 50).sum()))
# one k-core round
keep_u = set(us[us >= 15].index)
keep_i = set(is_[is_ >= 50].index)
sub = df[df["u"].isin(keep_u) & df["i"].isin(keep_i)]
print("after 1 round", len(sub), sub["u"].nunique(), sub["i"].nunique())
