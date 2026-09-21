#!/usr/bin/env python
"""Evaluate a frozen Stage1 checkpoint with train-only semantic late fusion."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.rec_model import LGCN_Encoder  # noqa: E402
from utils.experiment_utils import parse_measure_block, save_run_record  # noqa: E402
from utils.late_fusion import build_user_semantic_profiles, late_fusion_scores  # noqa: E402
from utils.load_strict import load_strict_datasets, user_item_matrix_from_graph  # noqa: E402
from utils.semantic_prior import load_semantic_prior  # noqa: E402
from utils.util_loss import (  # noqa: E402
    generate_interaction_matrix_from_dgl,
    get_rec_list,
    mask_seen_items,
    normalize_graph_mat,
    ranking_evaluation,
)


def score_split(
    datasets, split, rec_model, user_profiles, item_semantic,
    train_interactions, n_user, alpha, device,
):
    rec_model.eval()
    with torch.no_grad():
        emb = rec_model()
        user_emb, item_emb = emb[:n_user], emb[n_user:]
        users = datasets[f"{split}_user_set"]
        user_list = sorted(int(u) for u in users)
        user_idx = torch.as_tensor(user_list, dtype=torch.long, device=device)
        collaborative = user_emb[user_idx] @ item_emb.t()
        semantic = user_profiles[user_idx] @ item_semantic.t()
        fused = late_fusion_scores(collaborative, semantic, alpha)
        # mask_seen_items expects rows indexed by absolute user id.
        full_scores = fused.new_full((n_user, fused.shape[1]), -torch.inf)
        full_scores[user_idx] = fused
        full_scores = mask_seen_items(full_scores, train_interactions, users)
        recs = get_rec_list(users, full_scores, n_user, topk=20)
    return ranking_evaluation(
        datasets[f"{split}_origin_inter"], recs, [10, 20],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_record", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--semantic_prior_path", default="")
    ap.add_argument("--eval_split", default="ood", choices=["ood", "iid"])
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--protocol_version", default="corrected_v3_late_fusion")
    ap.add_argument("--comparison_label", default="")
    args = ap.parse_args()

    source = json.load(open(args.source_record, encoding="utf-8"))
    settings = source["settings"]
    dataset = settings["dataset"]
    seed = int(settings["seed"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    datasets, meta = load_strict_datasets(
        dataset, data_root=args.data_root, eval_split=args.eval_split,
    )
    n_user, n_item = int(meta["n_user"]), int(meta["n_item"])
    train_interactions, _, _ = user_item_matrix_from_graph(
        datasets["train"], n_user, n_item,
    )
    prior_path = args.semantic_prior_path or str(
        Path(args.data_root) / "features" / "semantic_prior.pt"
    )
    prior = load_semantic_prior(prior_path, map_location="cpu")
    item_semantic = torch.nn.functional.normalize(
        prior["item_emb"].float().to(device), dim=1,
    )
    profiles = build_user_semantic_profiles(item_semantic, train_interactions)

    checkpoint = torch.load(source["checkpoint"], map_location="cpu")
    state = checkpoint["rec_model"]
    context = datasets["test"]
    norm_adj = normalize_graph_mat(
        generate_interaction_matrix_from_dgl(context, n_user, n_item),
    )
    rec_model = LGCN_Encoder(
        n_user, int(settings.get("lgcn_layers", 3)), norm_adj,
        state["embedding_dict.user_emb"], state["embedding_dict.item_emb"],
    ).to(device)
    rec_model.load_state_dict(state)

    # Alpha and the comparison set are already frozen.  Score the collaborative
    # baseline and semantic fusion from the same loaded test context so the
    # paired comparison does not require a second test-data pass.
    baseline_val_measure = score_split(
        datasets, "val", rec_model, profiles, item_semantic,
        train_interactions, n_user, 0.0, device,
    )
    val_measure = score_split(
        datasets, "val", rec_model, profiles, item_semantic,
        train_interactions, n_user, args.alpha, device,
    )
    baseline_test_measure = score_split(
        datasets, "test", rec_model, profiles, item_semantic,
        train_interactions, n_user, 0.0, device,
    )
    test_measure = score_split(
        datasets, "test", rec_model, profiles, item_semantic,
        train_interactions, n_user, args.alpha, device,
    )
    record = {
        "settings": {
            "dataset": dataset,
            "seed": seed,
            "method": (
                "pure_lightgcn_semantic_late_fusion"
                if settings.get("method") == "pure_lightgcn_bpr"
                else "lsci_stage1_semantic_late_fusion"
            ),
            "protocol_version": args.protocol_version,
            "comparison_label": args.comparison_label or None,
            "source_protocol_version": settings.get("protocol_version"),
            "data_version": meta["data_version"],
            "eval_split": args.eval_split,
            "selection_split": "val",
            "selection_metric": "NDCG@20",
            "alpha": float(args.alpha),
            "alpha_selection_seeds": [1024, 2048, 3072],
            "alpha_confirmation_seeds": [4096, 5120],
            "score_normalization": "per_user_zscore",
            "user_semantic_profile": "mean_train_item_semantics",
            "semantic_prior_path": prior_path,
            "semantic_prior_meta": prior.get("meta", {}),
            "frozen_comparison_set": ["collaborative_alpha_0", "semantic_late_fusion"],
            "test_gt_loaded": True,
        },
        "source_record": args.source_record,
        "checkpoint": source["checkpoint"],
        "best_epoch": source["best_epoch"],
        "baseline_val_metrics": parse_measure_block(baseline_val_measure),
        "best_val_metrics": parse_measure_block(val_measure),
        "baseline_metrics": parse_measure_block(baseline_test_measure),
        "best_metrics": parse_measure_block(test_measure),
    }
    save_run_record(args.out, record)
    print("Validation collaborative baseline:\n" + "".join(baseline_val_measure))
    print("Validation late-fusion:\n" + "".join(val_measure))
    print("One-shot test collaborative baseline:\n" + "".join(baseline_test_measure))
    print("One-shot test late-fusion:\n" + "".join(test_measure))
    print("Saved", args.out)


if __name__ == "__main__":
    main()
