#!/usr/bin/env python
"""Select semantic late-fusion alpha using validation GT only."""
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


def ndcg_at_20(scores, users, ground_truth, train_interactions):
    for row, user in enumerate(users):
        start, end = train_interactions.indptr[user:user + 2]
        seen = train_interactions.indices[start:end]
        if len(seen):
            scores[row, torch.as_tensor(seen, device=scores.device)] = -torch.inf
    top = torch.topk(scores, 20, dim=1).indices.cpu().numpy()
    values = []
    for row, user in enumerate(users):
        gt = ground_truth[user]
        dcg = sum(
            1.0 / math.log2(rank + 2)
            for rank, item in enumerate(top[row]) if int(item) in gt
        )
        idcg = sum(
            1.0 / math.log2(rank + 2)
            for rank in range(min(20, len(gt)))
        )
        values.append(dcg / idcg)
    return float(np.mean(values))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--source_records", nargs="+", required=True)
    ap.add_argument("--semantic_prior_path", default="")
    ap.add_argument("--shuffled_prior_path", default="")
    ap.add_argument(
        "--alphas",
        default="0,0.025,0.05,0.075,0.1,0.15,0.2,0.25,0.5,0.75,1,1.25,1.5,2",
    )
    ap.add_argument("--selection_seeds", default="1024,2048,3072")
    ap.add_argument("--confirmation_seeds", default="4096,5120")
    ap.add_argument("--out", required=True)
    ap.add_argument("--protocol_version", default="corrected_v3_late_fusion_validation_only")
    args = ap.parse_args()

    alphas = [float(x) for x in args.alphas.split(",")]
    selection = [int(x) for x in args.selection_seeds.split(",")]
    confirmation = [int(x) for x in args.confirmation_seeds.split(",")]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    datasets, meta = load_strict_datasets(
        args.dataset, args.data_root, eval_split="ood", load_test_gt=False,
    )
    n_user, n_item = int(meta["n_user"]), int(meta["n_item"])
    train_interactions, _, _ = user_item_matrix_from_graph(
        datasets["train"], n_user, n_item,
    )
    norm_adj = normalize_graph_mat(generate_interaction_matrix_from_dgl(
        datasets["val"], n_user, n_item,
    ))
    real_path = args.semantic_prior_path or str(
        Path(args.data_root) / "features" / "semantic_prior.pt"
    )
    shuffled_path = args.shuffled_prior_path or str(
        Path(args.data_root) / "features" / "semantic_prior_shuffled.pt"
    )
    priors = {
        "real": load_semantic_prior(real_path, "cpu")["item_emb"].float(),
        "shuffle": load_semantic_prior(shuffled_path, "cpu")["item_emb"].float(),
    }
    profiles = {}
    for name, item in priors.items():
        item = torch.nn.functional.normalize(item.to(device), dim=1)
        priors[name] = item
        profiles[name] = build_user_semantic_profiles(item, train_interactions)

    ground_truth = {
        int(user): {int(item) - n_user for item in items}
        for user, items in datasets["val_origin_inter"].items()
    }
    users = sorted(ground_truth)
    user_idx = torch.as_tensor(users, dtype=torch.long, device=device)
    semantic_scores = {
        name: profiles[name][user_idx] @ priors[name].t()
        for name in priors
    }

    results = {}
    for source_path in args.source_records:
        source = json.load(open(source_path, encoding="utf-8"))
        seed = int(source["settings"]["seed"])
        checkpoint = torch.load(source["checkpoint"], map_location="cpu")
        state = checkpoint["rec_model"]
        rec_model = LGCN_Encoder(
            n_user, int(source["settings"].get("lgcn_layers", 3)), norm_adj,
            state["embedding_dict.user_emb"], state["embedding_dict.item_emb"],
        ).to(device)
        rec_model.load_state_dict(state)
        rec_model.eval()
        with torch.no_grad():
            emb = rec_model()
            collaborative = emb[:n_user][user_idx] @ emb[n_user:].t()
            results[str(seed)] = {"source_record": source_path}
            for prior_name in ("real", "shuffle"):
                results[str(seed)][prior_name] = {}
                for alpha in alphas:
                    fused = late_fusion_scores(
                        collaborative, semantic_scores[prior_name], alpha,
                    )
                    results[str(seed)][prior_name][str(alpha)] = ndcg_at_20(
                        fused.clone(), users, ground_truth, train_interactions,
                    )

    def mean_for(prior_name, alpha, seeds):
        return float(np.mean([
            results[str(seed)][prior_name][str(alpha)] for seed in seeds
        ]))

    selected_alpha = max(alphas, key=lambda a: mean_for("real", a, selection))
    selection_real = mean_for("real", selected_alpha, selection)
    selection_shuffle = mean_for("shuffle", selected_alpha, selection)
    confirmation_real = mean_for("real", selected_alpha, confirmation)
    confirmation_shuffle = mean_for("shuffle", selected_alpha, confirmation)
    selection_baseline = mean_for("real", 0.0, selection)
    confirmation_baseline = mean_for("real", 0.0, confirmation)
    checks = {
        "test_gt_not_loaded": True,
        "positive_alpha": selected_alpha > 0,
        "selection_beats_baseline": selection_real > selection_baseline,
        "confirmation_beats_baseline": confirmation_real > confirmation_baseline,
        "confirmation_beats_shuffle": confirmation_real > confirmation_shuffle,
        "confirmation_relative_gain_at_least_1pct": (
            confirmation_real / confirmation_baseline - 1.0 >= 0.01
        ),
        "each_confirmation_seed_improves": all(
            results[str(seed)]["real"][str(selected_alpha)]
            > results[str(seed)]["real"]["0.0"]
            for seed in confirmation
        ),
    }
    summary = {
        "dataset": args.dataset,
        "protocol_version": args.protocol_version,
        "test_gt_loaded": False,
        "alpha_frozen_before_ood": True,
        "alphas": alphas,
        "selection_seeds": selection,
        "confirmation_seeds": confirmation,
        "selected_alpha": selected_alpha,
        "selection_real_ndcg20": selection_real,
        "selection_shuffle_ndcg20": selection_shuffle,
        "confirmation_real_ndcg20": confirmation_real,
        "confirmation_shuffle_ndcg20": confirmation_shuffle,
        "selection_baseline_ndcg20": selection_baseline,
        "confirmation_baseline_ndcg20": confirmation_baseline,
        "semantic_prior_path": real_path,
        "shuffled_prior_path": shuffled_path,
        "checks": checks,
        "ood_test_allowed": all(checks.values()),
    }
    save_run_record(args.out, {"summary": summary, "per_seed": results})
    print(json.dumps(summary, indent=2))
    print("Saved", args.out)


if __name__ == "__main__":
    main()
