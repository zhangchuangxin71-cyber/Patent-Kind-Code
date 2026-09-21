#!/usr/bin/env python
"""Validation-only mechanism ablation for Yelp semantic late fusion."""
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
from utils.late_fusion import build_user_semantic_profiles, late_fusion_scores  # noqa: E402
from utils.load_strict import load_strict_datasets, user_item_matrix_from_graph  # noqa: E402
from utils.semantic_prior import load_semantic_prior  # noqa: E402
from utils.util_loss import generate_interaction_matrix_from_dgl, normalize_graph_mat  # noqa: E402


def per_user_ndcg20(scores, users, ground_truth, train_interactions):
    for row, user in enumerate(users):
        start, end = train_interactions.indptr[user:user + 2]
        seen = train_interactions.indices[start:end]
        if len(seen):
            scores[row, torch.as_tensor(seen, device=scores.device)] = -torch.inf
    top = torch.topk(scores, 20, dim=1).indices.cpu().numpy()
    out = np.zeros(len(users), dtype=np.float64)
    for row, user in enumerate(users):
        gt = ground_truth[user]
        dcg = sum(
            1.0 / math.log2(rank + 2)
            for rank, item in enumerate(top[row]) if int(item) in gt
        )
        idcg = sum(1.0 / math.log2(rank + 2) for rank in range(min(20, len(gt))))
        out[row] = dcg / idcg
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--source_records", nargs="+", required=True)
    ap.add_argument("--validation_report", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    validation = json.load(open(args.validation_report, encoding="utf-8"))
    alpha = float(validation["summary"]["selected_alpha"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    datasets, meta = load_strict_datasets(
        "yelp2018", args.data_root, eval_split="ood", load_test_gt=False,
    )
    n_user, n_item = int(meta["n_user"]), int(meta["n_item"])
    train, _, _ = user_item_matrix_from_graph(datasets["train"], n_user, n_item)
    norm_adj = normalize_graph_mat(generate_interaction_matrix_from_dgl(
        datasets["val"], n_user, n_item,
    ))
    users = sorted(int(x) for x in datasets["val_user_set"])
    user_idx = torch.as_tensor(users, dtype=torch.long, device=device)
    ground_truth = {
        int(user): {int(item) - n_user for item in items}
        for user, items in datasets["val_origin_inter"].items()
    }

    prior_paths = {
        "real": Path(args.data_root) / "features" / "semantic_prior.pt",
        "shuffle": Path(args.data_root) / "features" / "semantic_prior_shuffled.pt",
    }
    semantic_scores = {}
    for name, path in prior_paths.items():
        item = load_semantic_prior(str(path), "cpu")["item_emb"].float().to(device)
        item = torch.nn.functional.normalize(item, dim=1)
        profiles = build_user_semantic_profiles(item, train)
        semantic_scores[name] = profiles[user_idx] @ item.t()

    counts = np.diff(train.indptr)[np.asarray(users)]
    # Thresholds were determined from the validation-user train-history quartiles.
    segment_masks = {
        "Q1_4_to_16": counts <= 16,
        "Q2_17_to_22": (counts > 16) & (counts <= 22),
        "Q3_23_to_32": (counts > 22) & (counts <= 32),
        "Q4_33_plus": counts > 32,
    }
    popularity = torch.as_tensor(
        np.asarray(train.sum(axis=0)).ravel(), device=device, dtype=torch.float32,
    ).expand(len(users), -1)
    fixed_vectors = {
        "pure_semantic": per_user_ndcg20(
            semantic_scores["real"].clone(), users, ground_truth, train,
        ),
        "popularity": per_user_ndcg20(popularity.clone(), users, ground_truth, train),
    }

    per_seed = {}
    for source_path in args.source_records:
        source = json.load(open(source_path, encoding="utf-8"))
        seed = int(source["settings"]["seed"])
        checkpoint = torch.load(source["checkpoint"], map_location="cpu")
        state = checkpoint["rec_model"]
        model = LGCN_Encoder(
            n_user, 3, norm_adj,
            state["embedding_dict.user_emb"], state["embedding_dict.item_emb"],
        ).to(device)
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            emb = model()
            collaborative = emb[:n_user][user_idx] @ emb[n_user:].t()
            vectors = {
                "collaborative": per_user_ndcg20(
                    collaborative.clone(), users, ground_truth, train,
                ),
                "real_fusion": per_user_ndcg20(
                    late_fusion_scores(
                        collaborative, semantic_scores["real"], alpha,
                    ), users, ground_truth, train,
                ),
                "shuffle_fusion": per_user_ndcg20(
                    late_fusion_scores(
                        collaborative, semantic_scores["shuffle"], alpha,
                    ), users, ground_truth, train,
                ),
                **fixed_vectors,
            }
        per_seed[str(seed)] = {
            method: {
                "all": float(values.mean()),
                **{
                    segment: float(values[mask].mean())
                    for segment, mask in segment_masks.items()
                },
            }
            for method, values in vectors.items()
        }
        del model, emb, collaborative
        torch.cuda.empty_cache()

    methods = next(iter(per_seed.values())).keys()
    groups = ["all", *segment_masks]
    aggregate = {
        method: {
            group: float(np.mean([x[method][group] for x in per_seed.values()]))
            for group in groups
        }
        for method in methods
    }
    aggregate["real_minus_collaborative"] = {
        group: aggregate["real_fusion"][group] - aggregate["collaborative"][group]
        for group in groups
    }
    out = {
        "dataset": "yelp2018",
        "protocol": "validation_only_activity_segment_mechanism_ablation",
        "test_gt_loaded": False,
        "alpha_frozen_from_selection": alpha,
        "activity_definition": "number of train interactions among validation users",
        "segment_counts": {name: int(mask.sum()) for name, mask in segment_masks.items()},
        "aggregate_ndcg20": aggregate,
        "per_seed_ndcg20": per_seed,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
