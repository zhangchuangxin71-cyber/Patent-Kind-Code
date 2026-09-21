#!/usr/bin/env python
"""Validation-only raw-text vs LLM-rewritten semantic-prior ablation.

The primary comparison freezes one alpha selected for the LLM prior and applies
that same alpha to the raw-text prior.  A secondary comparison reports the best
validation-selected alpha for each prior separately.  OOD ground truth is never
loaded by this script.
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


def ndcg_at_20(scores, users, ground_truth, train_interactions):
    scores = scores.clone()
    for row, user in enumerate(users):
        start, end = train_interactions.indptr[user:user + 2]
        seen = train_interactions.indices[start:end]
        if len(seen):
            scores[row, torch.as_tensor(seen, device=scores.device)] = -torch.inf
    top = torch.topk(scores, 20, dim=1).indices.cpu().numpy()
    values = []
    for row, user in enumerate(users):
        truth = ground_truth[user]
        dcg = sum(
            1.0 / math.log2(rank + 2)
            for rank, item in enumerate(top[row]) if int(item) in truth
        )
        idcg = sum(
            1.0 / math.log2(rank + 2)
            for rank in range(min(20, len(truth)))
        )
        values.append(dcg / idcg if idcg else 0.0)
    return float(np.mean(values))


def parse_csv(text, cast):
    return [cast(value) for value in text.split(",") if value.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--source_records", nargs="+", required=True)
    ap.add_argument("--llm_prior_path", required=True)
    ap.add_argument("--raw_prior_path", required=True)
    ap.add_argument(
        "--alphas",
        default="0,0.025,0.05,0.075,0.1,0.15,0.2,0.25,0.5,0.75,1,1.25,1.5,2",
    )
    ap.add_argument("--selection_seeds", default="1024,2048,3072")
    ap.add_argument("--confirmation_seeds", default="4096,5120")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    alphas = parse_csv(args.alphas, float)
    selection = parse_csv(args.selection_seeds, int)
    confirmation = parse_csv(args.confirmation_seeds, int)
    expected_seeds = selection + confirmation
    if 0.0 not in alphas:
        raise ValueError("alpha grid must contain 0 for the collaborative baseline")
    if set(selection) & set(confirmation):
        raise ValueError("selection and confirmation seeds must be disjoint")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    datasets, meta = load_strict_datasets(
        args.dataset, args.data_root, eval_split="ood", load_test_gt=False,
    )
    n_user, n_item = int(meta["n_user"]), int(meta["n_item"])
    train, _, _ = user_item_matrix_from_graph(datasets["train"], n_user, n_item)
    norm_adj = normalize_graph_mat(generate_interaction_matrix_from_dgl(
        datasets["val"], n_user, n_item,
    ))

    prior_paths = {"llm": args.llm_prior_path, "raw": args.raw_prior_path}
    prior_bundles = {
        name: load_semantic_prior(path, "cpu")
        for name, path in prior_paths.items()
    }
    llm_meta = prior_bundles["llm"].get("meta", {})
    raw_meta = prior_bundles["raw"].get("meta", {})
    for name, bundle in prior_bundles.items():
        if int(bundle["item_emb"].shape[0]) != n_item:
            raise ValueError(f"{name} prior item count does not match dataset")
    if prior_bundles["llm"]["item_emb"].shape != prior_bundles["raw"]["item_emb"].shape:
        raise ValueError("LLM and raw priors must have identical shapes")
    if llm_meta.get("backend") != "sbert" or raw_meta.get("backend") != "sbert":
        raise ValueError("both priors must use the SBERT backend")
    if llm_meta.get("sbert_model") != raw_meta.get("sbert_model"):
        raise ValueError("both priors must use the same SBERT model")

    priors = {}
    profiles = {}
    for name, bundle in prior_bundles.items():
        item = torch.nn.functional.normalize(
            bundle["item_emb"].float().to(device), dim=1,
        )
        priors[name] = item
        profiles[name] = build_user_semantic_profiles(item, train)

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

    records = {}
    for source_path in args.source_records:
        source = json.loads(Path(source_path).read_text(encoding="utf-8"))
        seed = int(source["settings"]["seed"])
        checkpoint = torch.load(source["checkpoint"], map_location="cpu")
        state = checkpoint["rec_model"]
        model = LGCN_Encoder(
            n_user,
            int(source["settings"].get("lgcn_layers", 3)),
            norm_adj,
            state["embedding_dict.user_emb"],
            state["embedding_dict.item_emb"],
        ).to(device)
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            emb = model()
            collaborative = emb[:n_user][user_idx] @ emb[n_user:].t()
            grid = {name: {} for name in priors}
            for name in priors:
                for alpha in alphas:
                    fused = late_fusion_scores(
                        collaborative, semantic_scores[name], alpha,
                    )
                    grid[name][str(alpha)] = ndcg_at_20(
                        fused, users, ground_truth, train,
                    )
        records[str(seed)] = {
            "source_record": source_path,
            "checkpoint": source["checkpoint"],
            "source_validation_only": source["settings"].get("validation_only") is True,
            "source_test_metrics_absent": source.get("best_metrics") is None,
            "source_selection_split": source["settings"].get("selection_split"),
            "source_selection_metric": source["settings"].get("selection_metric"),
            **grid,
        }

    missing = sorted(set(expected_seeds) - {int(seed) for seed in records})
    if missing:
        raise RuntimeError(f"missing source records for seeds {missing}")

    def mean(prior, alpha, seeds):
        return float(np.mean([
            records[str(seed)][prior][str(alpha)] for seed in seeds
        ]))

    llm_alpha = max(alphas, key=lambda alpha: mean("llm", alpha, selection))
    raw_alpha = max(alphas, key=lambda alpha: mean("raw", alpha, selection))
    shared_alpha = llm_alpha
    baseline_selection = mean("llm", 0.0, selection)
    baseline_confirmation = mean("llm", 0.0, confirmation)

    checks = {
        "test_gt_not_loaded": meta.get("test_gt_loaded") is False,
        "same_item_count_and_dimension": (
            prior_bundles["llm"]["item_emb"].shape
            == prior_bundles["raw"]["item_emb"].shape
        ),
        "same_sbert_model": (
            llm_meta.get("sbert_model") == raw_meta.get("sbert_model")
        ),
        "expected_text_source_modes": (
            llm_meta.get("mode") == "llm" and raw_meta.get("mode") == "raw"
        ),
        "complete_equal_semantic_masks": (
            torch.equal(
                prior_bundles["llm"]["semantic_mask"],
                prior_bundles["raw"]["semantic_mask"],
            )
            and bool(prior_bundles["llm"]["semantic_mask"].all())
        ),
        "source_checkpoints_validation_selected_only": all(
            records[str(seed)]["source_validation_only"]
            and records[str(seed)]["source_test_metrics_absent"]
            and records[str(seed)]["source_selection_split"] == "val"
            and records[str(seed)]["source_selection_metric"] == "NDCG@20"
            for seed in expected_seeds
        ),
        "shared_alpha_positive": shared_alpha > 0,
        "selection_llm_beats_raw_same_alpha": (
            mean("llm", shared_alpha, selection)
            > mean("raw", shared_alpha, selection)
        ),
        "confirmation_llm_beats_raw_same_alpha": (
            mean("llm", shared_alpha, confirmation)
            > mean("raw", shared_alpha, confirmation)
        ),
        "each_confirmation_seed_llm_beats_raw_same_alpha": all(
            records[str(seed)]["llm"][str(shared_alpha)]
            > records[str(seed)]["raw"][str(shared_alpha)]
            for seed in confirmation
        ),
    }
    summary = {
        "dataset": args.dataset,
        "protocol_version": "corrected_v5_raw_vs_deepseek_validation_only",
        "test_gt_loaded": False,
        "primary_comparison": "same frozen checkpoint and LLM-selected shared alpha",
        "selection_seeds": selection,
        "confirmation_seeds": confirmation,
        "alphas": alphas,
        "llm_prior_path": args.llm_prior_path,
        "raw_prior_path": args.raw_prior_path,
        "llm_prior_meta": llm_meta,
        "raw_prior_meta": raw_meta,
        "selected_alpha": {
            "shared_llm_selected": shared_alpha,
            "llm_independently_selected": llm_alpha,
            "raw_independently_selected": raw_alpha,
        },
        "selection": {
            "baseline": baseline_selection,
            "llm_shared": mean("llm", shared_alpha, selection),
            "raw_shared": mean("raw", shared_alpha, selection),
            "llm_own_alpha": mean("llm", llm_alpha, selection),
            "raw_own_alpha": mean("raw", raw_alpha, selection),
        },
        "confirmation": {
            "baseline": baseline_confirmation,
            "llm_shared": mean("llm", shared_alpha, confirmation),
            "raw_shared": mean("raw", shared_alpha, confirmation),
            "llm_own_alpha": mean("llm", llm_alpha, confirmation),
            "raw_own_alpha": mean("raw", raw_alpha, confirmation),
        },
        "checks": checks,
        "ood_test_allowed": all(checks.values()),
    }
    save_run_record(args.out, {"summary": summary, "per_seed": records})
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Saved", args.out)


if __name__ == "__main__":
    main()
