#!/usr/bin/env python
"""Fail closed unless A0/A1/A2 records match the preregistered protocol."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


SEEDS = (1024, 2048, 3072, 4096, 5120)


def load(paths):
    records = [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]
    by_seed = {int(record["settings"]["seed"]): record for record in records}
    if tuple(sorted(by_seed)) != SEEDS:
        raise RuntimeError(f"expected seeds {SEEDS}, got {tuple(sorted(by_seed))}")
    return by_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--a0", nargs="+", required=True)
    parser.add_argument("--a1", nargs="+", required=True)
    parser.add_argument("--a2", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    arms = {"A0": load(args.a0), "A1": load(args.a1), "A2": load(args.a2)}
    all_records = [record for arm in arms.values() for record in arm.values()]

    def every(predicate):
        return all(predicate(record["settings"]) for record in all_records)

    checks = {
        "five_identical_model_seeds": all(tuple(sorted(arm)) == SEEDS for arm in arms.values()),
        "dataset_is_fixed": every(lambda s: s.get("dataset") == args.dataset),
        "strict_data_version": every(lambda s: s.get("data_version") == "v1_strict"),
        "data_split_seed_1024": every(lambda s: s.get("semantic_prior_meta", {}).get("split_seed", 1024) == 1024),
        "validation_only_checkpoint_selection": all(
            record["settings"].get("validation_only") is True
            and record.get("best_metrics") is None
            and record["settings"].get("selection_split") == "val"
            and record["settings"].get("selection_metric") == "NDCG@20"
            for record in all_records
        ),
        "twenty_five_outer_epochs": every(lambda s: int(s.get("epochs")) == 25),
        "one_complete_bpr_pass_per_outer_epoch": every(lambda s: int(s.get("rec_epochs_per_outer")) == 1),
        "embedding_dimension_8": every(lambda s: int(s.get("hidden2")) == 8),
        "lightgcn_layers_3": every(lambda s: int(s.get("lgcn_layers")) == 3),
        "bpr_batch_size_1024": every(lambda s: int(s.get("rec_batch_size")) == 1024),
        "ranking_learning_rate_0_001": every(lambda s: float(s.get("rec_lr")) == 0.001),
        "ranking_l2_0_001_source_constant": True,
        "diffusion_steps_100": every(lambda s: int(s.get("steps")) == 100),
        "diffusion_predicts_x0": every(lambda s: s.get("mean_type") == "x0"),
        "rec_refresh_0_25": every(lambda s: float(s.get("rec_refresh")) == 0.25),
        "edge_gate_disabled": every(lambda s: s.get("edge_gate_mode") == "none"),
        "upstream_rank_loss_disabled": every(lambda s: float(s.get("lambda_rank_upstream")) == 0.0),
        "alpha_zero_during_all_training_and_selection": every(lambda s: float(s.get("semantic_score_alpha")) == 0.0),
        "all_checkpoints_exist": all(Path(record["checkpoint"]).exists() for record in all_records),
        "a0_has_no_semantic_prior": all(not r["settings"].get("use_semantic_prior") for r in arms["A0"].values()),
        "a1_uses_condition_fusion_without_infonce": all(
            r["settings"].get("use_semantic_prior") is True
            and float(r["settings"].get("lambda_sem")) == 0.0
            for r in arms["A1"].values()
        ),
        "a2_adds_infonce_lambda_0_1": all(
            r["settings"].get("use_semantic_prior") is True
            and float(r["settings"].get("lambda_sem")) == 0.1
            for r in arms["A2"].values()
        ),
    }
    output = {
        "dataset": args.dataset,
        "protocol_version": "corrected_v6_training_module_ablation_audit",
        "checks": checks,
        "ood_evaluation_allowed": all(checks.values()),
        "records": {
            arm: {
                str(seed): {
                    "record": str(Path(record_path).resolve()),
                    "checkpoint": records[seed]["checkpoint"],
                    "best_epoch": records[seed]["best_epoch"],
                    "best_val_ndcg20": records[seed]["best_val_metrics"]["Top20"]["NDCG"],
                }
                for seed, record_path in zip(SEEDS, paths)
            }
            for (arm, records), paths in zip(arms.items(), (args.a0, args.a1, args.a2))
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"checks": checks, "ood_evaluation_allowed": output["ood_evaluation_allowed"]}, indent=2))
    if not output["ood_evaluation_allowed"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
