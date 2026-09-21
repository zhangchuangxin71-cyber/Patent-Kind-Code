#!/usr/bin/env python
"""Select activity-cohort semantic weights using validation ground truth only."""
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
from utils.late_fusion import (  # noqa: E402
    adaptive_late_fusion_scores,
    build_user_semantic_profiles,
    late_fusion_scores,
)
from utils.load_strict import load_strict_datasets, user_item_matrix_from_graph  # noqa: E402
from utils.semantic_prior import load_semantic_prior  # noqa: E402
from utils.util_loss import generate_interaction_matrix_from_dgl, normalize_graph_mat  # noqa: E402


GROUPS = ("low_activity", "mid_activity", "high_activity")


def per_user_ndcg(scores, users, ground_truth, train):
    for row, user in enumerate(users):
        seen = train.indices[train.indptr[user]:train.indptr[user + 1]]
        scores[row, torch.as_tensor(seen, device=scores.device)] = -torch.inf
    top = torch.topk(scores, 20, dim=1).indices.cpu().numpy()
    values = []
    for row, user in enumerate(users):
        gt = ground_truth[int(user)]
        dcg = sum(1.0 / math.log2(rank + 2) for rank, item in enumerate(top[row]) if int(item) in gt)
        idcg = sum(1.0 / math.log2(rank + 2) for rank in range(min(20, len(gt))))
        values.append(dcg / idcg)
    return np.asarray(values, dtype=float)


def group_labels(degrees, q1, q3):
    return np.where(degrees <= q1, GROUPS[0], np.where(degrees <= q3, GROUPS[1], GROUPS[2]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--source_records", nargs="+", required=True)
    ap.add_argument("--uniform_validation_report", required=True)
    ap.add_argument(
        "--alphas",
        default="0,0.025,0.05,0.075,0.1,0.15,0.2,0.25,0.5,0.75,1,1.25,1.5,2",
        help="Must cover the uniform alpha so adaptive fusion has a fair search space.",
    )
    ap.add_argument("--selection_seeds", default="1024,2048,3072")
    ap.add_argument("--confirmation_seeds", default="4096,5120")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    alphas = [float(x) for x in args.alphas.split(",")]
    selection = [int(x) for x in args.selection_seeds.split(",")]
    confirmation = [int(x) for x in args.confirmation_seeds.split(",")]
    uniform_report = json.loads(Path(args.uniform_validation_report).read_text(encoding="utf-8"))
    uniform_summary = uniform_report["summary"]
    if uniform_summary.get("test_gt_loaded", True):
        raise RuntimeError("uniform alpha must come from a validation-only report")
    uniform_alpha = float(uniform_summary["selected_alpha"])
    if uniform_alpha not in alphas:
        raise ValueError(
            f"alpha candidates must include uniform alpha {uniform_alpha}; got {alphas}"
        )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    datasets, meta = load_strict_datasets(args.dataset, args.data_root, "ood", load_test_gt=False)
    n_user, n_item = int(meta["n_user"]), int(meta["n_item"])
    train, _, _ = user_item_matrix_from_graph(datasets["train"], n_user, n_item)
    norm_adj = normalize_graph_mat(generate_interaction_matrix_from_dgl(datasets["val"], n_user, n_item))
    gt = {int(u): {int(i) - n_user for i in items} for u, items in datasets["val_origin_inter"].items()}
    users = np.asarray(sorted(gt), dtype=np.int64)
    user_idx = torch.as_tensor(users, dtype=torch.long, device=device)
    degrees = np.diff(train.indptr)[users]
    q1, q3 = [float(x) for x in np.quantile(degrees, [0.25, 0.75])]
    labels = group_labels(degrees, q1, q3)

    real_item = torch.nn.functional.normalize(load_semantic_prior(
        str(Path(args.data_root) / "features" / "semantic_prior.pt"), "cpu",
    )["item_emb"].float().to(device), dim=1)
    shuffle_item = torch.nn.functional.normalize(load_semantic_prior(
        str(Path(args.data_root) / "features" / "semantic_prior_shuffled.pt"), "cpu",
    )["item_emb"].float().to(device), dim=1)
    real_semantic = build_user_semantic_profiles(real_item, train)[user_idx] @ real_item.t()
    shuffle_semantic = build_user_semantic_profiles(shuffle_item, train)[user_idx] @ shuffle_item.t()

    cached = {}
    for source_path in args.source_records:
        source = json.loads(Path(source_path).read_text(encoding="utf-8"))
        seed = int(source["settings"]["seed"])
        state = torch.load(source["checkpoint"], map_location="cpu")["rec_model"]
        model = LGCN_Encoder(
            n_user, int(source["settings"].get("lgcn_layers", 3)), norm_adj,
            state["embedding_dict.user_emb"], state["embedding_dict.item_emb"],
        ).to(device)
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            emb = model()
            collaborative = emb[:n_user][user_idx] @ emb[n_user:].t()
            cached[seed] = {
                "source_record": source_path,
                "collaborative": collaborative,
            }

    missing = set(selection + confirmation) - set(cached)
    if missing:
        raise RuntimeError(f"missing seeds: {sorted(missing)}")

    real_grid = {}
    for seed, data in cached.items():
        real_grid[seed] = {}
        for alpha in alphas:
            fused = late_fusion_scores(data["collaborative"], real_semantic, alpha)
            real_grid[seed][str(alpha)] = per_user_ndcg(
                fused.clone(), users, gt, train,
            )

    selected = {}
    selection_grid = {}
    for group in GROUPS:
        mask = labels == group
        selection_grid[group] = {}
        for alpha in alphas:
            values = np.concatenate([
                real_grid[seed][str(alpha)][mask] for seed in selection
            ])
            selection_grid[group][str(alpha)] = float(np.mean(values))
        selected[group] = max(alphas, key=lambda a: selection_grid[group][str(a)])

    row_alpha = torch.as_tensor(
        [selected[str(group)] for group in labels], dtype=torch.float32, device=device,
    )
    per_seed = {}
    for seed, data in sorted(cached.items()):
        collaborative = data["collaborative"]
        variants = {
            "baseline": late_fusion_scores(collaborative, real_semantic, 0.0),
            "uniform_real": late_fusion_scores(collaborative, real_semantic, uniform_alpha),
            "adaptive_real": adaptive_late_fusion_scores(collaborative, real_semantic, row_alpha),
            "adaptive_shuffle": adaptive_late_fusion_scores(collaborative, shuffle_semantic, row_alpha),
        }
        scores = {name: per_user_ndcg(value.clone(), users, gt, train) for name, value in variants.items()}
        per_seed[str(seed)] = {
            "source_record": data["source_record"],
            **{name: float(value.mean()) for name, value in scores.items()},
            "groups": {
                group: {name: float(value[labels == group].mean()) for name, value in scores.items()}
                for group in GROUPS
            },
        }

    def mean(name, seeds):
        return float(np.mean([per_seed[str(seed)][name] for seed in seeds]))

    checks = {
        "test_gt_not_loaded": True,
        "confirmation_beats_uniform": mean("adaptive_real", confirmation) > mean("uniform_real", confirmation),
        "confirmation_beats_baseline": mean("adaptive_real", confirmation) > mean("baseline", confirmation),
        "confirmation_beats_shuffle": mean("adaptive_real", confirmation) > mean("adaptive_shuffle", confirmation),
        "each_confirmation_seed_beats_uniform": all(
            per_seed[str(seed)]["adaptive_real"] > per_seed[str(seed)]["uniform_real"]
            for seed in confirmation
        ),
    }
    summary = {
        "dataset": args.dataset,
        "protocol_version": "corrected_v4_activity_adaptive_validation_only",
        "test_gt_loaded": False,
        "selection_seeds": selection,
        "confirmation_seeds": confirmation,
        "activity_thresholds_frozen": {"q1": q1, "q3": q3},
        "group_rule": "low: degree<=q1; mid: q1<degree<=q3; high: degree>q3",
        "alpha_candidates": alphas,
        "selected_group_alphas": selected,
        "uniform_alpha": uniform_alpha,
        "uniform_validation_ood_gate": bool(uniform_summary.get("ood_test_allowed", False)),
        "selection": {name: mean(name, selection) for name in variants},
        "confirmation": {name: mean(name, confirmation) for name in variants},
        "checks": checks,
        "ood_test_allowed": all(checks.values()),
    }
    save_run_record(args.out, {"summary": summary, "selection_grid": selection_grid, "per_seed": per_seed})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
