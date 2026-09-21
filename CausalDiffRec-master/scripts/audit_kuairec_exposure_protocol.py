#!/usr/bin/env python
"""Read-only audit of KuaiRec train/validation/OOD exposure contracts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def summarize(df):
    return {
        "rows": int(len(df)),
        "users": int(df.user_id.nunique()),
        "items": int(df.item_id.nunique()),
        "source_matrix": {
            str(k): int(v) for k, v in df.source_matrix.value_counts(dropna=False).to_dict().items()
        },
        "user_item_pairs": int(df[["user_id", "item_id"]].drop_duplicates().shape[0]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root = Path(args.data_root) / "interactions"
    frames = {name: pd.read_parquet(root / f"{name}.parquet") for name in ("train", "val", "iid_test", "ood_test")}
    summary = {name: summarize(df) for name, df in frames.items()}
    train_pairs = set(zip(frames["train"].user_id, frames["train"].item_id))
    val_pairs = set(zip(frames["val"].user_id, frames["val"].item_id))
    ood_pairs = set(zip(frames["ood_test"].user_id, frames["ood_test"].item_id))
    train_users = set(frames["train"].user_id)
    train_items = set(frames["train"].item_id)
    summary["overlap_checks"] = {
        "val_pair_overlap_with_train": len(train_pairs & val_pairs),
        "ood_pair_overlap_with_train": len(train_pairs & ood_pairs),
        "ood_users_seen_in_train": len(set(frames["ood_test"].user_id) & train_users),
        "ood_items_seen_in_train": len(set(frames["ood_test"].item_id) & train_items),
        "ood_users": len(set(frames["ood_test"].user_id)),
        "ood_items": len(set(frames["ood_test"].item_id)),
    }
    summary["protocol_findings"] = [
        "train/val are entirely source_matrix=big while ood_test is source_matrix=small",
        "validation and OOD therefore do not share the same exposure universe",
        "model or alpha selection on current val must not be treated as an OOD claim",
    ]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
