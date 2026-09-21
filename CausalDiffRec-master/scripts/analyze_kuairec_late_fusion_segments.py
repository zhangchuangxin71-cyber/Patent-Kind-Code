#!/usr/bin/env python
"""Exploratory validation-only segment analysis for KuaiRec late fusion."""
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


def ranked_items(scores, users, train_interactions):
    scores = scores.clone()
    for row, user in enumerate(users):
        start, end = train_interactions.indptr[user:user + 2]
        seen = train_interactions.indices[start:end]
        if len(seen):
            scores[row, torch.as_tensor(seen, device=scores.device)] = -torch.inf
    return torch.topk(scores, 20, dim=1).indices.cpu().numpy()


def ndcg_vector(top_items, users, ground_truth, allowed_items=None):
    out = np.full(len(users), np.nan, dtype=np.float64)
    for row, user in enumerate(users):
        truth = ground_truth[user]
        if allowed_items is not None:
            truth = truth.intersection(allowed_items)
        if not truth:
            continue
        dcg = sum(
            1.0 / math.log2(rank + 2)
            for rank, item in enumerate(top_items[row]) if int(item) in truth
        )
        idcg = sum(
            1.0 / math.log2(rank + 2)
            for rank in range(min(20, len(truth)))
        )
        out[row] = dcg / idcg
    return out


def finite_mean(values, mask=None):
    selected = values if mask is None else values[mask]
    selected = selected[np.isfinite(selected)]
    return float(selected.mean()) if len(selected) else None


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
        "kuairec", args.data_root, eval_split="ood", load_test_gt=False,
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

    user_counts = np.diff(train.indptr)[np.asarray(users)]
    user_q = np.quantile(user_counts, [0.25, 0.5, 0.75]).astype(int)
    user_masks = {
        f"user_activity_q1_le_{user_q[0]}": user_counts <= user_q[0],
        f"user_activity_q2_{user_q[0] + 1}_to_{user_q[1]}": (
            (user_counts > user_q[0]) & (user_counts <= user_q[1])
        ),
        f"user_activity_q3_{user_q[1] + 1}_to_{user_q[2]}": (
            (user_counts > user_q[1]) & (user_counts <= user_q[2])
        ),
        f"user_activity_q4_gt_{user_q[2]}": user_counts > user_q[2],
    }

    item_pop = np.asarray(train.sum(axis=0)).ravel().astype(int)
    positive_pop = item_pop[item_pop > 0]
    item_q = np.quantile(positive_pop, [0.25, 0.75]).astype(int)
    item_segments = {
        "gt_item_cold_pop_0": set(np.flatnonzero(item_pop == 0).tolist()),
        f"gt_item_tail_pop_1_to_{item_q[0]}": set(
            np.flatnonzero((item_pop > 0) & (item_pop <= item_q[0])).tolist()
        ),
        f"gt_item_mid_pop_{item_q[0] + 1}_to_{item_q[1]}": set(
            np.flatnonzero((item_pop > item_q[0]) & (item_pop <= item_q[1])).tolist()
        ),
        f"gt_item_head_pop_gt_{item_q[1]}": set(
            np.flatnonzero(item_pop > item_q[1]).tolist()
        ),
    }

    group_meta = {
        "all": {"users": len(users), "validation_interactions": sum(map(len, ground_truth.values()))},
    }
    for name, mask in user_masks.items():
        selected_users = [users[i] for i in np.flatnonzero(mask)]
        group_meta[name] = {
            "users": int(mask.sum()),
            "validation_interactions": int(sum(len(ground_truth[u]) for u in selected_users)),
        }
    for name, allowed in item_segments.items():
        counts = [len(ground_truth[u].intersection(allowed)) for u in users]
        group_meta[name] = {
            "users": int(sum(x > 0 for x in counts)),
            "validation_interactions": int(sum(counts)),
            "candidate_items_in_segment": len(allowed),
        }

    popularity_scores = torch.as_tensor(
        item_pop, device=device, dtype=torch.float32,
    ).expand(len(users), -1)
    fixed_top = {
        "pure_semantic": ranked_items(semantic_scores["real"], users, train),
        "popularity": ranked_items(popularity_scores, users, train),
    }

    def summarize_top(top):
        overall = ndcg_vector(top, users, ground_truth)
        values = {"all": finite_mean(overall)}
        values.update({name: finite_mean(overall, mask) for name, mask in user_masks.items()})
        values.update({
            name: finite_mean(ndcg_vector(top, users, ground_truth, allowed))
            for name, allowed in item_segments.items()
        })
        return values

    fixed_summary = {name: summarize_top(top) for name, top in fixed_top.items()}
    per_seed = {}
    for source_path in args.source_records:
        source = json.load(open(source_path, encoding="utf-8"))
        seed = int(source["settings"]["seed"])
        checkpoint = torch.load(source["checkpoint"], map_location="cpu")
        state = checkpoint["rec_model"]
        model = LGCN_Encoder(
            n_user, int(source["settings"]["lgcn_layers"]), norm_adj,
            state["embedding_dict.user_emb"], state["embedding_dict.item_emb"],
        ).to(device)
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            emb = model()
            collaborative = emb[:n_user][user_idx] @ emb[n_user:].t()
            tops = {
                "collaborative": ranked_items(collaborative, users, train),
                "real_fusion": ranked_items(
                    late_fusion_scores(collaborative, semantic_scores["real"], alpha),
                    users, train,
                ),
                "shuffle_fusion": ranked_items(
                    late_fusion_scores(collaborative, semantic_scores["shuffle"], alpha),
                    users, train,
                ),
            }
        per_seed[str(seed)] = {
            **{name: summarize_top(top) for name, top in tops.items()},
            **fixed_summary,
        }
        del model, emb, collaborative
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    groups = list(group_meta)
    methods = list(next(iter(per_seed.values())))
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
    aggregate["real_improved_seed_count"] = {
        group: int(sum(
            x["real_fusion"][group] > x["collaborative"][group]
            for x in per_seed.values()
        ))
        for group in groups
    }
    aggregate["real_minus_shuffle"] = {
        group: aggregate["real_fusion"][group] - aggregate["shuffle_fusion"][group]
        for group in groups
    }

    out = {
        "dataset": "kuairec",
        "protocol": "exploratory_validation_only_user_activity_and_item_popularity_segments",
        "test_gt_loaded": False,
        "alpha_frozen_from_prior_selection": alpha,
        "caution": (
            "All five seeds have already been inspected. This is mechanism diagnosis, "
            "not independent confirmation; any redesigned method needs new confirmation seeds."
        ),
        "user_activity_quartile_thresholds": user_q.tolist(),
        "positive_item_popularity_thresholds": item_q.tolist(),
        "group_metadata": group_meta,
        "aggregate_ndcg20": aggregate,
        "per_seed_ndcg20": per_seed,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps({
        "alpha": alpha,
        "group_metadata": group_meta,
        "aggregate_ndcg20": aggregate,
    }, ensure_ascii=False, indent=2))
    print("Saved", args.out)


if __name__ == "__main__":
    main()
