#!/usr/bin/env python
"""Food validation-only diagnostic for temporal semantic profile drift."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.rec_model import LGCN_Encoder  # noqa: E402
from utils.experiment_utils import save_run_record  # noqa: E402
from utils.late_fusion import build_user_semantic_profiles, late_fusion_scores  # noqa: E402
from utils.load_strict import load_strict_datasets, user_item_matrix_from_graph  # noqa: E402
from utils.semantic_prior import load_semantic_prior  # noqa: E402
from utils.util_loss import generate_interaction_matrix_from_dgl, normalize_graph_mat  # noqa: E402


def ndcg(scores, users, gt, train):
    for row, user in enumerate(users):
        seen = train.indices[train.indptr[user]:train.indptr[user + 1]]
        scores[row, torch.as_tensor(seen, device=scores.device)] = -torch.inf
    top = torch.topk(scores, 20, dim=1).indices.cpu().numpy()
    values = []
    for row, user in enumerate(users):
        target = gt[int(user)]
        dcg = sum(1 / math.log2(rank + 2) for rank, item in enumerate(top[row]) if int(item) in target)
        ideal = sum(1 / math.log2(rank + 2) for rank in range(min(20, len(target))))
        values.append(dcg / ideal)
    return float(np.mean(values))


def recent_profiles(item_emb, train_df, n_user, recent_n, device):
    profiles = item_emb.new_zeros((n_user, item_emb.shape[1]))
    ordered = train_df.sort_values(["user_id", "timestamp"], kind="mergesort")
    for user, rows in ordered.groupby("user_id", sort=False):
        items = rows["item_id"].to_numpy()[-recent_n:]
        profiles[int(user)] = item_emb[torch.as_tensor(items, device=device)].mean(0)
    return torch.nn.functional.normalize(profiles, dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="food")
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--source_records", nargs="+", required=True)
    ap.add_argument("--alphas", default="0,0.025,0.05,0.075,0.1,0.15,0.2,0.25,0.5")
    ap.add_argument("--recent_windows", default="1,3,5,10")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    alphas = [float(v) for v in args.alphas.split(",")]
    windows = [int(v) for v in args.recent_windows.split(",")]
    selection, confirmation = (1024, 2048, 3072), (4096, 5120)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    datasets, meta = load_strict_datasets(args.dataset, args.data_root, "ood", load_test_gt=False)
    n_user, n_item = int(meta["n_user"]), int(meta["n_item"])
    train, _, _ = user_item_matrix_from_graph(datasets["train"], n_user, n_item)
    norm_adj = normalize_graph_mat(generate_interaction_matrix_from_dgl(datasets["val"], n_user, n_item))
    users = sorted(int(u) for u in datasets["val_user_set"])
    idx = torch.as_tensor(users, dtype=torch.long, device=device)
    gt = {int(u): {int(i) - n_user for i in items} for u, items in datasets["val_origin_inter"].items()}
    train_df = pd.read_parquet(Path(args.data_root) / "interactions" / "train.parquet")
    priors = {
        "real": torch.nn.functional.normalize(load_semantic_prior(
            str(Path(args.data_root) / "features" / "semantic_prior.pt"), "cpu",
        )["item_emb"].float().to(device), dim=1),
        "shuffle": torch.nn.functional.normalize(load_semantic_prior(
            str(Path(args.data_root) / "features" / "semantic_prior_shuffled.pt"), "cpu",
        )["item_emb"].float().to(device), dim=1),
    }
    profiles = {}
    for prior_name, emb in priors.items():
        profiles[(prior_name, "all")] = build_user_semantic_profiles(emb, train)[idx]
        for window in windows:
            profiles[(prior_name, f"recent_{window}")] = recent_profiles(
                emb, train_df, n_user, window, device,
            )[idx]

    collaborative = {}
    for path in args.source_records:
        source = json.loads(Path(path).read_text(encoding="utf-8"))
        seed = int(source["settings"]["seed"])
        state = torch.load(source["checkpoint"], map_location="cpu")["rec_model"]
        model = LGCN_Encoder(n_user, int(source["settings"].get("lgcn_layers", 3)), norm_adj,
                            state["embedding_dict.user_emb"], state["embedding_dict.item_emb"]).to(device)
        model.load_state_dict(state)
        with torch.no_grad():
            emb = model()
            collaborative[seed] = emb[:n_user][idx] @ emb[n_user:].t()

    modes = ["all"] + [f"recent_{w}" for w in windows]
    grid = {}
    for mode in modes:
        for alpha in alphas:
            values = []
            for seed in selection:
                semantic = profiles[("real", mode)] @ priors["real"].t()
                values.append(ndcg(late_fusion_scores(collaborative[seed], semantic, alpha).clone(), users, gt, train))
            grid[f"{mode}|{alpha}"] = float(np.mean(values))
    selected_key = max(grid, key=grid.get)
    mode, alpha_text = selected_key.split("|")
    alpha = float(alpha_text)
    per_seed = {}
    for seed, score in collaborative.items():
        baseline = ndcg(late_fusion_scores(score, profiles[("real", "all")] @ priors["real"].t(), 0).clone(), users, gt, train)
        real = ndcg(late_fusion_scores(score, profiles[("real", mode)] @ priors["real"].t(), alpha).clone(), users, gt, train)
        shuffled = ndcg(late_fusion_scores(score, profiles[("shuffle", mode)] @ priors["shuffle"].t(), alpha).clone(), users, gt, train)
        per_seed[str(seed)] = {"baseline": baseline, "real": real, "shuffle": shuffled}
    mean = lambda name, seeds: float(np.mean([per_seed[str(s)][name] for s in seeds]))
    checks = {
        "test_gt_not_loaded": True,
        "confirmation_beats_baseline": mean("real", confirmation) > mean("baseline", confirmation),
        "confirmation_beats_shuffle": mean("real", confirmation) > mean("shuffle", confirmation),
        "each_confirmation_seed_improves": all(per_seed[str(s)]["real"] > per_seed[str(s)]["baseline"] for s in confirmation),
    }
    summary = {
        "dataset": args.dataset, "protocol": "corrected_v4_temporal_profile_validation_only",
        "test_gt_loaded": False, "selection_seeds": selection, "confirmation_seeds": confirmation,
        "selected_profile": mode, "selected_alpha": alpha, "grid": grid,
        "selection": {name: mean(name, selection) for name in ("baseline", "real", "shuffle")},
        "confirmation": {name: mean(name, confirmation) for name in ("baseline", "real", "shuffle")},
        "checks": checks, "ood_test_allowed": all(checks.values()),
    }
    save_run_record(args.out, {"summary": summary, "per_seed": per_seed})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
