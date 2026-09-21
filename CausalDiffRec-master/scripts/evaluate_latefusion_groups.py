#!/usr/bin/env python
"""Validation-only cohort diagnostics for frozen collaborative checkpoints.

User cohorts report user-averaged NDCG@20.  Item cohorts report the recall of
validation positives at 20, so a user with positives in several popularity
bands contributes correctly to every relevant band.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.rec_model import LGCN_Encoder  # noqa: E402
from utils.experiment_utils import save_run_record  # noqa: E402
from utils.late_fusion import build_user_semantic_profiles, late_fusion_scores  # noqa: E402
from utils.load_strict import load_strict_datasets, user_item_matrix_from_graph  # noqa: E402
from utils.semantic_prior import load_semantic_prior  # noqa: E402
from utils.util_loss import generate_interaction_matrix_from_dgl, normalize_graph_mat  # noqa: E402


def _bins(values: np.ndarray, low_label: str, mid_label: str, high_label: str):
    q1, q3 = np.quantile(values, [0.25, 0.75])
    return q1, q3, {
        low_label: values <= q1,
        mid_label: (values > q1) & (values <= q3),
        high_label: values > q3,
    }


def _ndcg(recommendations: np.ndarray, positives: set[int]) -> float:
    dcg = sum(
        1.0 / math.log2(rank + 2)
        for rank, item in enumerate(recommendations) if int(item) in positives
    )
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(min(20, len(positives))))
    return dcg / idcg if idcg else 0.0


def _summarize(values: list[float]):
    values = np.asarray(values, dtype=float)
    return {
        "n": int(len(values)),
        "mean": float(values.mean()) if len(values) else None,
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0 if len(values) else None,
    }


def _model_scores(source, n_user, n_item, norm_adj, user_idx, device):
    checkpoint = torch.load(source["checkpoint"], map_location="cpu")
    state = checkpoint["rec_model"]
    model = LGCN_Encoder(
        n_user, int(source["settings"].get("lgcn_layers", 3)), norm_adj,
        state["embedding_dict.user_emb"], state["embedding_dict.item_emb"],
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        emb = model()
        return emb[:n_user][user_idx] @ emb[n_user:].t()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--source_records", nargs="+", required=True)
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--semantic_prior", choices=("real", "shuffle", "none"), default="real")
    ap.add_argument("--semantic_prior_path", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--protocol_version", default="corrected_v4_group_validation_only")
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    datasets, meta = load_strict_datasets(args.dataset, args.data_root, "ood", load_test_gt=False)
    n_user, n_item = int(meta["n_user"]), int(meta["n_item"])
    train, _, _ = user_item_matrix_from_graph(datasets["train"], n_user, n_item)
    norm_adj = normalize_graph_mat(generate_interaction_matrix_from_dgl(datasets["val"], n_user, n_item))
    gt = {int(u): {int(i) - n_user for i in items} for u, items in datasets["val_origin_inter"].items()}
    users = np.asarray(sorted(gt), dtype=np.int64)
    user_idx = torch.as_tensor(users, dtype=torch.long, device=device)
    user_degree = np.diff(train.indptr)[users]
    user_q1, user_q3, user_masks = _bins(user_degree, "low_activity", "mid_activity", "high_activity")
    item_degree = np.asarray(train.sum(axis=0)).ravel().astype(np.int64)
    positive_items = np.asarray(sorted({item for values in gt.values() for item in values}), dtype=np.int64)
    item_q1, item_q3, item_masks = _bins(item_degree[positive_items], "tail", "mid", "head")
    item_sets = {name: set(positive_items[mask].tolist()) for name, mask in item_masks.items()}
    item_sets["cold_le_5"] = set(positive_items[item_degree[positive_items] <= 5].tolist())

    semantic_scores = None
    if args.semantic_prior != "none":
        default = "semantic_prior.pt" if args.semantic_prior == "real" else "semantic_prior_shuffled.pt"
        path = args.semantic_prior_path or str(Path(args.data_root) / "features" / default)
        item_semantic = torch.nn.functional.normalize(
            load_semantic_prior(path, "cpu")["item_emb"].float().to(device), dim=1,
        )
        semantic_scores = build_user_semantic_profiles(item_semantic, train)[user_idx] @ item_semantic.t()

    per_seed = {}
    for record_path in args.source_records:
        source = json.load(open(record_path, encoding="utf-8"))
        seed = str(int(source["settings"]["seed"]))
        scores = _model_scores(source, n_user, n_item, norm_adj, user_idx, device)
        if semantic_scores is not None:
            scores = late_fusion_scores(scores, semantic_scores, args.alpha)
        for row, user in enumerate(users):
            seen = train.indices[train.indptr[user]:train.indptr[user + 1]]
            scores[row, torch.as_tensor(seen, device=device)] = -torch.inf
        recs = torch.topk(scores, 20, dim=1).indices.cpu().numpy()
        user_values = np.asarray([_ndcg(recs[row], gt[int(user)]) for row, user in enumerate(users)])
        user_cohorts = {
            name: _summarize(user_values[mask].tolist()) for name, mask in user_masks.items()
        }
        item_cohorts = {}
        for name, members in item_sets.items():
            eligible = hits = 0
            for row, user in enumerate(users):
                target = gt[int(user)] & members
                if target:
                    eligible += len(target)
                    hits += sum(int(item) in target for item in recs[row])
            item_cohorts[name] = {
                "positive_count": eligible,
                "recall_at_20": float(hits / eligible) if eligible else None,
            }
        per_seed[seed] = {
            "source_record": record_path,
            "user_ndcg20": user_cohorts,
            "item_target_recall20": item_cohorts,
        }

    aggregate = {"user_ndcg20": {}, "item_target_recall20": {}}
    for name in user_masks:
        aggregate["user_ndcg20"][name] = _summarize([
            row["user_ndcg20"][name]["mean"] for row in per_seed.values()
        ])
    for name in item_sets:
        values = [row["item_target_recall20"][name]["recall_at_20"] for row in per_seed.values()]
        aggregate["item_target_recall20"][name] = _summarize([v for v in values if v is not None])
        aggregate["item_target_recall20"][name]["positive_count"] = int(sum(
            row["item_target_recall20"][name]["positive_count"] for row in per_seed.values()
        ))
    summary = {
        "dataset": args.dataset,
        "protocol_version": args.protocol_version,
        "test_gt_loaded": False,
        "alpha": args.alpha,
        "semantic_prior": args.semantic_prior,
        "cohort_definition": {
            "user_activity": {"q1": float(user_q1), "q3": float(user_q3)},
            "item_train_popularity": {"q1": float(item_q1), "q3": float(item_q3), "cold": "degree <= 5"},
        },
        "aggregate": aggregate,
    }
    save_run_record(args.out, {"summary": summary, "per_seed": per_seed})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
