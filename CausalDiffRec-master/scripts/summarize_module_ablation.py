#!/usr/bin/env python
"""Summarize paired A0/A1/A2/A3 and semantic-shuffle OOD results."""
from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np


SEEDS = (1024, 2048, 3072, 4096, 5120)
METRICS = ("Recall", "NDCG")


def load_seeded(pattern: str) -> dict[int, dict]:
    records = [json.loads(Path(path).read_text(encoding="utf-8")) for path in glob.glob(pattern)]
    by_seed = {int(record["settings"]["seed"]): record for record in records}
    if tuple(sorted(by_seed)) != SEEDS:
        raise RuntimeError(f"Expected seeds {SEEDS} for {pattern}, got {tuple(sorted(by_seed))}")
    return by_seed


def metric(record: dict, field: str, name: str) -> float:
    return float(record[field]["Top20"][name])


def summary(values) -> dict:
    x = np.asarray(values, dtype=float)
    return {
        "values": [float(v) for v in x],
        "mean": float(x.mean()),
        "sample_std": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
    }


def sign_p(values) -> tuple[int, int, float]:
    values = [value for value in values if value != 0]
    n = len(values)
    positive = sum(value > 0 for value in values)
    p = sum(math.comb(n, k) for k in range(positive, n + 1)) / (2 ** n) if n else 1.0
    return positive, n, float(p)


def paired(left, right) -> dict:
    delta = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    positive, non_ties, p = sign_p(delta.tolist())
    mean = float(delta.mean())
    std = float(delta.std(ddof=1)) if len(delta) > 1 else 0.0
    half = 2.7764451051977987 * std / math.sqrt(len(delta)) if len(delta) > 1 else 0.0
    baseline = float(np.mean(right))
    return {
        "paired_differences": [float(v) for v in delta],
        "mean_difference": mean,
        "sample_std_difference": std,
        "relative_gain_percent": (100.0 * mean / baseline) if baseline else None,
        "paired_t_95ci": [mean - half, mean + half],
        "positive_seed_count": positive,
        "non_tied_seed_count": non_ties,
        "one_sided_exact_sign_p": p,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--root", default=".")
    parser.add_argument("--out", required=True)
    parser.add_argument("--markdown", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    records = root / "experiments" / "records"
    reports = root / "experiments" / "reports"
    alpha_report = json.loads((reports / f"{args.dataset}_v6_module_ablation_alpha_validation.json").read_text(encoding="utf-8"))
    alpha = float(alpha_report["summary"]["selected_alpha"])

    a0 = load_seeded(str(records / f"{args.dataset}_v6_module_ablation_ood_a0_seed*.json"))
    a1 = load_seeded(str(records / f"{args.dataset}_v6_module_ablation_ood_a1_seed*.json"))
    a3 = load_seeded(str(records / f"{args.dataset}_v6_module_ablation_ood_a3_real_seed*.json"))
    shuffle_paths = glob.glob(str(records / f"{args.dataset}_v6_module_ablation_ood_a3_shuffle*_seed*.json"))
    shuffles = [json.loads(Path(path).read_text(encoding="utf-8")) for path in shuffle_paths]
    if len(shuffles) != 25:
        raise RuntimeError(f"Expected 25 shuffled evaluations, got {len(shuffles)}")

    arms = {name: {metric_name: [] for metric_name in METRICS} for name in ("A0", "A1", "A2", "A3")}
    for seed in SEEDS:
        for name in METRICS:
            arms["A0"][name].append(metric(a0[seed], "best_metrics", name))
            arms["A1"][name].append(metric(a1[seed], "best_metrics", name))
            arms["A2"][name].append(metric(a3[seed], "baseline_metrics", name))
            arms["A3"][name].append(metric(a3[seed], "best_metrics", name))

    arm_summary = {
        arm: {f"{name}@20": summary(values) for name, values in metrics.items()}
        for arm, metrics in arms.items()
    }
    comparisons = {}
    for name in METRICS:
        key = f"{name}@20"
        comparisons[f"A1_minus_A0_{key}"] = paired(arms["A1"][name], arms["A0"][name])
        comparisons[f"A2_minus_A1_{key}"] = paired(arms["A2"][name], arms["A1"][name])
        comparisons[f"A3_minus_A2_{key}"] = paired(arms["A3"][name], arms["A2"][name])

    external = {}
    for label, pattern in {
        "CausalDiffRec": records / f"{args.dataset}_v5_fair_bpr25_ood_causaldiffrec_seed*.json",
        "LightGCN": records / f"{args.dataset}_v5_fair_bpr25_ood_lightgcn_seed*.json",
    }.items():
        try:
            source = load_seeded(str(pattern))
        except RuntimeError:
            continue
        external[label] = {}
        for name in METRICS:
            values = [metric(source[seed], "best_metrics", name) for seed in SEEDS]
            external[label][f"{name}@20"] = summary(values)
            comparisons[f"A3_minus_{label}_{name}@20"] = paired(arms["A3"][name], values)

    shuffled_by_seed = {seed: [] for seed in SEEDS}
    for record in shuffles:
        seed = int(record["settings"]["seed"])
        shuffled_by_seed[seed].append(metric(record, "best_metrics", "NDCG"))
    shuffle_seed_means = [float(np.mean(shuffled_by_seed[seed])) for seed in SEEDS]
    all_true = [value for value in arms["A3"]["NDCG"] for _ in range(5)]
    all_shuffle = [value for seed in SEEDS for value in shuffled_by_seed[seed]]
    shuffle_control = {
        "per_model_seed_shuffle_ndcg20": {str(seed): summary(shuffled_by_seed[seed]) for seed in SEEDS},
        "five_shuffle_mean_by_model_seed": shuffle_seed_means,
        "overall_shuffle_ndcg20": summary(all_shuffle),
        "real_ndcg20": summary(arms["A3"]["NDCG"]),
        "real_exceeds_shuffle_count": sum(real > shuffled for real, shuffled in zip(all_true, all_shuffle)),
        "comparisons": len(all_shuffle),
        "real_exceeds_shuffle_fraction": float(np.mean(np.asarray(all_true) > np.asarray(all_shuffle))),
        "paired_real_minus_five_shuffle_mean": paired(arms["A3"]["NDCG"], shuffle_seed_means),
    }

    same_checkpoint = True
    for seed in SEEDS:
        source_path = Path(a3[seed]["source_record"])
        source = json.loads(source_path.read_text(encoding="utf-8"))
        same_checkpoint = same_checkpoint and (
            int(source["settings"]["seed"]) == seed
            and float(source["settings"]["lambda_sem"]) == 0.1
            and float(source["settings"]["semantic_score_alpha"]) == 0.0
            and source["checkpoint"] == a3[seed]["checkpoint"]
            and float(a3[seed]["settings"]["alpha"]) == alpha
        )
    output = {
        "dataset": args.dataset,
        "protocol_version": "corrected_v6_training_module_ablation",
        "definitions": {
            "A0": "Stage-I backbone; no semantic prior; alpha=0",
            "A1": "A0 + ConditionFusion semantic prior; lambda_sem=0; alpha=0",
            "A2": "A1 + InfoNCE lambda_sem=0.1; alpha=0",
            "A3": f"same A2 checkpoint; score fusion alpha={alpha}",
        },
        "selected_alpha": alpha,
        "alpha_validation": alpha_report,
        "a3_reuses_a2_checkpoint": same_checkpoint,
        "metrics": arm_summary,
        "external_baselines": external,
        "comparisons": comparisons,
        "five_shuffle_control": shuffle_control,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        f"# {args.dataset} 训练阶段模块消融", "",
        f"验证集冻结的 α：**{alpha:g}**；A3复用A2检查点：**{same_checkpoint}**。", "",
        "| 组别 | Recall@20 | NDCG@20 |", "|---|---:|---:|",
    ]
    for arm in ("A0", "A1", "A2", "A3"):
        recall = arm_summary[arm]["Recall@20"]
        ndcg = arm_summary[arm]["NDCG@20"]
        lines.append(f"| {arm} | {recall['mean']:.6f} ± {recall['sample_std']:.6f} | {ndcg['mean']:.6f} ± {ndcg['sample_std']:.6f} |")
    lines.extend(["", "## NDCG@20 配对贡献", "", "| 比较 | 均值差 | 相对提升 | 95%区间 | 正向种子 | 单边符号检验p |", "|---|---:|---:|---:|---:|---:|"])
    for label in ("A1_minus_A0_NDCG@20", "A2_minus_A1_NDCG@20", "A3_minus_A2_NDCG@20", "A3_minus_CausalDiffRec_NDCG@20", "A3_minus_LightGCN_NDCG@20"):
        if label not in comparisons:
            continue
        row = comparisons[label]
        ci = row["paired_t_95ci"]
        lines.append(f"| {label} | {row['mean_difference']:.6f} | {row['relative_gain_percent']:.2f}% | [{ci[0]:.6f}, {ci[1]:.6f}] | {row['positive_seed_count']}/5 | {row['one_sided_exact_sign_p']:.5f} |")
    lines.extend(["", "## 真实语义与五次打乱", "", f"真实语义超过打乱语义：**{shuffle_control['real_exceeds_shuffle_count']}/{shuffle_control['comparisons']}**（{shuffle_control['real_exceeds_shuffle_fraction']:.1%}）。"])
    Path(args.markdown).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "selected_alpha": alpha}, indent=2))


if __name__ == "__main__":
    main()
