#!/usr/bin/env python
"""One-shot scoring with validation-frozen activity-cohort semantic weights."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.rec_model import LGCN_Encoder  # noqa: E402
from utils.experiment_utils import parse_measure_block, save_run_record  # noqa: E402
from utils.late_fusion import (  # noqa: E402
    adaptive_late_fusion_scores,
    build_user_semantic_profiles,
    late_fusion_scores,
)
from utils.load_strict import load_strict_datasets, user_item_matrix_from_graph  # noqa: E402
from utils.semantic_prior import load_semantic_prior  # noqa: E402
from utils.util_loss import (  # noqa: E402
    generate_interaction_matrix_from_dgl,
    get_rec_list,
    mask_seen_items,
    normalize_graph_mat,
    ranking_evaluation,
)


def evaluate(scores, users, origin_inter, train, n_user):
    full = scores.new_full((n_user, scores.shape[1]), -torch.inf)
    full[torch.as_tensor(users, device=scores.device)] = scores
    full = mask_seen_items(full, train, set(users))
    recs = get_rec_list(set(users), full, n_user, topk=20)
    return parse_measure_block(ranking_evaluation(origin_inter, recs, [10, 20]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validation_report", required=True)
    ap.add_argument("--source_record", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--semantic_prior", choices=("real", "shuffle"), required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    validation = json.loads(Path(args.validation_report).read_text(encoding="utf-8"))["summary"]
    if not validation.get("ood_test_allowed"):
        raise RuntimeError("activity-adaptive validation gate did not authorize OOD")
    source = json.loads(Path(args.source_record).read_text(encoding="utf-8"))
    dataset = source["settings"]["dataset"]
    seed = int(source["settings"]["seed"])
    uniform_alpha = float(validation["uniform_alpha"])
    group_alphas = validation["selected_group_alphas"]
    q1 = float(validation["activity_thresholds_frozen"]["q1"])
    q3 = float(validation["activity_thresholds_frozen"]["q3"])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    datasets, meta = load_strict_datasets(dataset, args.data_root, "ood", load_test_gt=True)
    n_user, n_item = int(meta["n_user"]), int(meta["n_item"])
    train, _, _ = user_item_matrix_from_graph(datasets["train"], n_user, n_item)
    prior_file = "semantic_prior.pt" if args.semantic_prior == "real" else "semantic_prior_shuffled.pt"
    item_semantic = torch.nn.functional.normalize(load_semantic_prior(
        str(Path(args.data_root) / "features" / prior_file), "cpu",
    )["item_emb"].float().to(device), dim=1)
    profiles = build_user_semantic_profiles(item_semantic, train)

    state = torch.load(source["checkpoint"], map_location="cpu")["rec_model"]
    norm_adj = normalize_graph_mat(generate_interaction_matrix_from_dgl(datasets["test"], n_user, n_item))
    model = LGCN_Encoder(
        n_user, int(source["settings"].get("lgcn_layers", 3)), norm_adj,
        state["embedding_dict.user_emb"], state["embedding_dict.item_emb"],
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        emb = model()
        user_emb, item_emb = emb[:n_user], emb[n_user:]
        split_metrics = {}
        for split in ("val", "test"):
            users = sorted(int(u) for u in datasets[f"{split}_user_set"])
            idx = torch.as_tensor(users, dtype=torch.long, device=device)
            collaborative = user_emb[idx] @ item_emb.t()
            semantic = profiles[idx] @ item_semantic.t()
            degrees = np.diff(train.indptr)[np.asarray(users)]
            labels = np.where(
                degrees <= q1, "low_activity",
                np.where(degrees <= q3, "mid_activity", "high_activity"),
            )
            row_alpha = torch.as_tensor(
                [group_alphas[str(label)] for label in labels],
                dtype=collaborative.dtype, device=device,
            )
            variants = {
                "baseline": late_fusion_scores(collaborative, semantic, 0.0),
                "uniform": late_fusion_scores(collaborative, semantic, uniform_alpha),
                "adaptive": adaptive_late_fusion_scores(collaborative, semantic, row_alpha),
            }
            split_metrics[split] = {
                name: evaluate(
                    scores, users, datasets[f"{split}_origin_inter"], train, n_user,
                )
                for name, scores in variants.items()
            }

    record = {
        "settings": {
            "dataset": dataset,
            "seed": seed,
            "protocol_version": "corrected_v4_activity_adaptive_frozen_ood",
            "semantic_prior": args.semantic_prior,
            "test_gt_loaded": True,
            "uniform_alpha": uniform_alpha,
            "selected_group_alphas": group_alphas,
            "activity_thresholds_frozen": {"q1": q1, "q3": q3},
        },
        "source_record": args.source_record,
        "checkpoint": source["checkpoint"],
        "validation_metrics": split_metrics["val"],
        "test_metrics": split_metrics["test"],
    }
    save_run_record(args.out, record)
    print(json.dumps(record["test_metrics"], indent=2))


if __name__ == "__main__":
    main()
