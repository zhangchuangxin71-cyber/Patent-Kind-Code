#!/usr/bin/env python
"""Audit the validation-only inputs for the fair three-method comparison."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


SEEDS = (1024, 2048, 3072, 4096, 5120)


def load_arm(paths):
    records = [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]
    by_seed = {int(record["settings"]["seed"]): record for record in records}
    if len(records) != len(SEEDS) or tuple(sorted(by_seed)) != SEEDS:
        raise RuntimeError(f"Expected exactly seeds {SEEDS}, got {tuple(sorted(by_seed))}")
    return by_seed


def validation_only(record):
    settings = record["settings"]
    return (
        settings.get("validation_only") is True
        and record.get("best_metrics") is None
        and settings.get("selection_split") == "val"
        and settings.get("selection_metric") == "NDCG@20"
    )


def bpr_passes(arm, record):
    settings = record["settings"]
    if arm == "lightgcn":
        return int(settings["epochs"])
    return int(settings["epochs"]) * int(settings.get("rec_epochs_per_outer", 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="movielens1m")
    parser.add_argument("--lightgcn_records", nargs="+", required=True)
    parser.add_argument("--causaldiffrec_records", nargs="+", required=True)
    parser.add_argument("--full_records", nargs="+", required=True)
    parser.add_argument(
        "--semantic_confirmation", default="",
        help=(
            "Optional validation-only semantic confirmation report. When omitted, "
            "the full method's own validation-only records and semantic settings are audited directly."
        ),
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    arms = {
        "lightgcn": load_arm(args.lightgcn_records),
        "causaldiffrec": load_arm(args.causaldiffrec_records),
        "full_method": load_arm(args.full_records),
    }
    confirmation = None
    if args.semantic_confirmation:
        confirmation = json.loads(
            Path(args.semantic_confirmation).read_text(encoding="utf-8")
        )
        # This optional input is generated solely from validation-only records
        # and contains no OOD metrics.

    checks = {
        "five_identical_seeds": all(tuple(sorted(records)) == SEEDS for records in arms.values()),
        "all_validation_only": all(
            validation_only(record) for records in arms.values() for record in records.values()
        ),
        "same_dataset": all(
            record["settings"].get("dataset") == args.dataset
            for records in arms.values() for record in records.values()
        ),
        "same_data_version": all(
            record["settings"].get("data_version") == "v1_strict"
            for records in arms.values() for record in records.values()
        ),
        "same_eval_split": all(
            record["settings"].get("eval_split") == "ood"
            for records in arms.values() for record in records.values()
        ),
        "equal_25_bpr_passes": all(
            bpr_passes(arm, record) == 25
            for arm, records in arms.items() for record in records.values()
        ),
        "same_embedding_dimension": all(
            int(record["settings"].get("embedding_dim", record["settings"].get("hidden2", 8))) == 8
            for records in arms.values() for record in records.values()
        ),
        "same_lightgcn_layers": all(
            int(record["settings"].get("lgcn_layers", 3)) == 3
            for records in arms.values() for record in records.values()
        ),
        "same_ranking_lr": all(
            float(record["settings"].get("learning_rate") if arm == "lightgcn"
                  else record["settings"].get("rec_lr")) == 0.001
            for arm, records in arms.items() for record in records.values()
        ),
        "same_ranking_batch_size": all(
            int(record["settings"].get("batch_size") if arm == "lightgcn"
                else record["settings"].get("rec_batch_size")) == 1024
            for arm, records in arms.items() for record in records.values()
        ),
        "expected_method_implementations": (
            all(record["settings"].get("method") == "pure_lightgcn_bpr"
                for record in arms["lightgcn"].values())
            and all(record["settings"].get("method") == "causaldiffrec_e0"
                    for record in arms["causaldiffrec"].values())
            and all(str(record["settings"].get("method", "")).startswith("lsci_e3")
                    for record in arms["full_method"].values())
        ),
        "all_selected_checkpoints_exist": all(
            Path(record["checkpoint"]).exists()
            for records in arms.values() for record in records.values()
        ),
        "full_uses_semantic_prior": all(
            record["settings"].get("use_semantic_prior") is True
            for record in arms["full_method"].values()
        ),
        "full_uses_infonce_lambda_0_1": all(
            float(record["settings"].get("lambda_sem")) == 0.1
            for record in arms["full_method"].values()
        ),
        "full_uses_semantic_score_alpha_0_75": all(
            float(record["settings"].get("semantic_score_alpha")) == 0.75
            for record in arms["full_method"].values()
        ),
        "unsupported_learned_edge_gate_disabled": all(
            record["settings"].get("edge_gate_mode") == "none"
            for record in arms["full_method"].values()
        ),
    }

    per_arm = {}
    for arm, records in arms.items():
        per_arm[arm] = {
            "source_records": {str(seed): record.get("checkpoint") for seed, record in records.items()},
            "validation_ndcg20": {
                str(seed): float(record["best_val_metrics"]["Top20"]["NDCG"])
                for seed, record in records.items()
            },
            "bpr_passes": {str(seed): bpr_passes(arm, record) for seed, record in records.items()},
            "peak_vram_mb": {str(seed): record.get("peak_vram_mb") for seed, record in records.items()},
            "wall_time_sec": {str(seed): record.get("wall_time_sec") for seed, record in records.items()},
        }

    output = {
        "dataset": args.dataset,
        "protocol_version": "corrected_v5_fair_three_method_bpr25_validation_audit",
        "comparison_arms": {
            "lightgcn": "LightGCN, collaborative-only",
            "causaldiffrec": "original CausalDiffRec E0",
            "full_method": (
                "Causal diffusion/LSCI backbone + semantic ConditionFusion + "
                "InfoNCE(lambda=0.1) + train-history semantic score fusion(alpha=0.75); "
                "learned edge gate disabled"
            ),
        },
        "shared_protocol": {
            "data": f"{args.dataset} v1_strict",
            "evaluation": "full-candidate OOD, train items masked, Recall/NDCG at 20",
            "checkpoint_selection": "validation NDCG@20 only",
            "seeds": list(SEEDS),
            "bpr_pass_budget": 25,
            "embedding_dimension": 8,
            "lightgcn_layers": 3,
            "ranking_learning_rate": 0.001,
            "ranking_batch_size": 1024,
            "ranking_l2": 0.001,
            "negative_sampling": "one random unobserved item per positive per BPR pass",
        },
        "semantic_confirmation_source": args.semantic_confirmation or None,
        "semantic_validation_confirmation": confirmation,
        "semantic_score_alpha_policy": {
            "value": 0.75,
            "mode": "fixed_full_method_parameter",
            "external_validation_confirmation": (
                "provided_and_allowed" if confirmation is not None
                else "not_provided_not_claimed"
            ),
            "interpretation": (
                "The audit verifies that every full-method record used alpha=0.75. "
                "When no external confirmation file is supplied, this is not evidence that "
                "alpha was independently selected or confirmed on validation; it is a fixed "
                "comparison parameter."
            ),
        },
        "checks": checks,
        "ood_test_allowed": all(checks.values()),
        "per_arm_validation": per_arm,
        "scope_note": (
            "Equal budget means equal complete BPR passes. CausalDiffRec/full method also "
            "optimize their architecture-specific representation objectives, which is part of each method. "
            "Checkpoint selection is validation-only. This audit does not assert an independent "
            "validation confirmation for a fixed semantic score alpha when no confirmation file is supplied."
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"checks": checks, "ood_test_allowed": output["ood_test_allowed"]}, indent=2))
    print("Saved", out)


if __name__ == "__main__":
    main()
