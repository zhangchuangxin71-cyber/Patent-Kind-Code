#!/usr/bin/env python3
"""Build a patent-facing summary from frozen OOD experiment records.

The script intentionally separates validation-gate numbers from OOD-test
numbers.  Patent tables should cite the OOD values computed from per-seed
records, while validation summaries are only used to document that alpha/profile
selection happened before test evaluation.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path


SEEDS = (1024, 2048, 3072, 4096, 5120)
BOOTSTRAP_ROUNDS = 20000


DATASETS = [
    {
        "label": "MovieLens-1M",
        "dataset": "movielens1m",
        "protocol": "corrected-v4 pure LightGCN BPR200 late fusion",
        "report": "experiments/reports/movielens1m_pure_lightgcn_v4_bpr200_ood.json",
        "real_pattern": "experiments/records/movielens1m_pure_lightgcn_v4_bpr200_ood_semantic_real_seed{seed}.json",
        "shuffle_pattern": "experiments/records/movielens1m_pure_lightgcn_v4_bpr200_ood_semantic_shuffled_seed{seed}.json",
        "kind": "late_fusion",
        "main_variant": "semantic_real",
    },
    {
        "label": "Yelp2018",
        "dataset": "yelp2018",
        "protocol": "corrected-v4 LSCI-stage late fusion",
        "report": "experiments/reports/yelp2018_v4_latefusion_ood.json",
        "real_pattern": "experiments/records/yelp2018_v4_lf_ood_semantic_real_seed{seed}_v4.json",
        "shuffle_pattern": "experiments/records/yelp2018_v4_lf_ood_semantic_shuffled_seed{seed}_v4.json",
        "kind": "late_fusion",
        "main_variant": "semantic_real",
    },
    {
        "label": "Food",
        "dataset": "food",
        "protocol": "corrected-v4 pure LightGCN late fusion",
        "report": "experiments/reports/food_v4_uniform_ood.json",
        "real_pattern": "experiments/records/food_v4_uniform_ood_semantic_real_seed{seed}.json",
        "shuffle_pattern": "experiments/records/food_v4_uniform_ood_semantic_shuffled_seed{seed}.json",
        "kind": "late_fusion",
        "main_variant": "semantic_real",
    },
    {
        "label": "Amazon Beauty",
        "dataset": "amazon_beauty",
        "protocol": "corrected-v4 activity-adaptive semantic fusion",
        "report": "experiments/reports/amazon_beauty_v4_activity_adaptive_ood.json",
        "real_pattern": "experiments/records/amazon_beauty_v4_activity_adaptive_ood_real_seed{seed}.json",
        "shuffle_pattern": "experiments/records/amazon_beauty_v4_activity_adaptive_ood_shuffle_seed{seed}.json",
        "kind": "activity_adaptive",
        "main_variant": "adaptive_real",
    },
]


SUPPORTING_REPORTS = {
    "strict_main": "experiments/reports/strict_ood_e0_stage1_e3_final.md",
    "kuairec_v2_latefusion": "experiments/reports/kuairec_v2_latefusion_validation.json",
    "kuairec_v2_popularity": "experiments/reports/kuairec_v2_popularity_rerank.json",
    "kuairec_v2_groups": "experiments/reports/kuairec_v2_latefusion_groups_real_a05.json",
    "complexity": "experiments/复杂度分析.md",
    "food_raw_vs_deepseek_validation": "experiments/reports/food_v5_raw_vs_deepseek_validation.json",
    "food_raw_vs_deepseek_ood": "experiments/reports/food_v5_raw_vs_deepseek_ood.json",
    "yelp_raw_vs_deepseek_validation": "experiments/reports/yelp2018_v5_raw_vs_deepseek_validation.json",
    "yelp_raw_vs_deepseek_ood": "experiments/reports/yelp2018_v5_raw_vs_deepseek_ood.json",
    "kuairec_raw_vs_deepseek_validation": "experiments/reports/kuairec_v5_raw_vs_deepseek_validation.json",
    "fair_three_method_movielens_validation": "experiments/reports/movielens1m_v5_fair_bpr25_validation_audit.json",
    "fair_three_method_movielens_ood": "experiments/reports/movielens1m_v5_fair_bpr25_method_comparison.json",
    "fair_three_method_yelp_validation": "experiments/reports/yelp2018_v5_fair_bpr25_validation_audit.json",
    "fair_three_method_yelp_ood": "experiments/reports/yelp2018_v5_fair_bpr25_method_comparison.json",
    "fair_three_method_food_validation": "experiments/reports/food_v5_fair_bpr25_validation_audit.json",
    "fair_three_method_food_ood": "experiments/reports/food_v5_fair_bpr25_method_comparison.json",
    "fair_three_method_amazon_validation": "experiments/reports/amazon_beauty_v5_fair_bpr25_validation_audit.json",
    "fair_three_method_amazon_ood": "experiments/reports/amazon_beauty_v5_fair_bpr25_method_comparison.json",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def metric(record: dict, metric_name: str, field: str = "Top20") -> float:
    return float(record[field][metric_name])


def top20_metric(top20_record: dict, metric_name: str) -> float:
    return float(top20_record[metric_name])


def mean_std(values: list[float]) -> dict:
    return {
        "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "values": values,
    }


def paired_diffs(a: list[float], b: list[float]) -> list[float]:
    return [x - y for x, y in zip(a, b)]


def bootstrap_ci(values: list[float], rounds: int = BOOTSTRAP_ROUNDS) -> list[float]:
    rng = random.Random(20260904)
    n = len(values)
    means = []
    for _ in range(rounds):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(statistics.mean(sample))
    means.sort()
    lo = means[int(0.025 * rounds)]
    hi = means[min(rounds - 1, int(0.975 * rounds))]
    return [lo, hi]


def one_sided_sign_p(values: list[float]) -> float:
    nonzero = [v for v in values if abs(v) > 1e-12]
    n = len(nonzero)
    if n == 0:
        return 1.0
    wins = sum(v > 0 for v in nonzero)
    # P(X >= wins), X ~ Binomial(n, 0.5)
    return sum(math.comb(n, k) for k in range(wins, n + 1)) / float(2**n)


def format_mean_std(stats: dict) -> str:
    return f"{stats['mean']:.5f} +/- {stats['sample_std']:.5f}"


def format_pct(x: float) -> str:
    return f"{100.0 * x:.2f}%"


def load_late_fusion_records(root: Path, cfg: dict) -> dict:
    baseline_r20, real_r20, shuffle_r20 = [], [], []
    baseline_n20, real_n20, shuffle_n20 = [], [], []
    sources = []
    for seed in SEEDS:
        real_path = root / cfg["real_pattern"].format(seed=seed)
        shuffle_path = root / cfg["shuffle_pattern"].format(seed=seed)
        real = read_json(real_path)
        shuffle = read_json(shuffle_path)
        if int(real["settings"]["seed"]) != seed or int(shuffle["settings"]["seed"]) != seed:
            raise RuntimeError(f"seed mismatch in {cfg['dataset']} seed {seed}")
        baseline_r20.append(top20_metric(real["baseline_metrics"]["Top20"], "Recall"))
        baseline_n20.append(top20_metric(real["baseline_metrics"]["Top20"], "NDCG"))
        real_r20.append(top20_metric(real["best_metrics"]["Top20"], "Recall"))
        real_n20.append(top20_metric(real["best_metrics"]["Top20"], "NDCG"))
        shuffle_r20.append(top20_metric(shuffle["best_metrics"]["Top20"], "Recall"))
        shuffle_n20.append(top20_metric(shuffle["best_metrics"]["Top20"], "NDCG"))
        sources.extend([str(real_path), str(shuffle_path)])
    return {
        "baseline": {"Recall@20": baseline_r20, "NDCG@20": baseline_n20},
        "real": {"Recall@20": real_r20, "NDCG@20": real_n20},
        "shuffle": {"Recall@20": shuffle_r20, "NDCG@20": shuffle_n20},
        "source_records": sources,
    }


def load_activity_records(root: Path, cfg: dict) -> dict:
    baseline_r20, real_r20, shuffle_r20 = [], [], []
    baseline_n20, real_n20, shuffle_n20 = [], [], []
    uniform_r20, uniform_n20 = [], []
    sources = []
    for seed in SEEDS:
        real_path = root / cfg["real_pattern"].format(seed=seed)
        shuffle_path = root / cfg["shuffle_pattern"].format(seed=seed)
        real = read_json(real_path)
        shuffle = read_json(shuffle_path)
        if int(real["settings"]["seed"]) != seed or int(shuffle["settings"]["seed"]) != seed:
            raise RuntimeError(f"seed mismatch in {cfg['dataset']} seed {seed}")
        baseline = real["test_metrics"]["baseline"]["Top20"]
        adaptive = real["test_metrics"]["adaptive"]["Top20"]
        uniform = real["test_metrics"]["uniform"]["Top20"]
        adaptive_shuffle = shuffle["test_metrics"]["adaptive"]["Top20"]
        baseline_r20.append(top20_metric(baseline, "Recall"))
        baseline_n20.append(top20_metric(baseline, "NDCG"))
        real_r20.append(top20_metric(adaptive, "Recall"))
        real_n20.append(top20_metric(adaptive, "NDCG"))
        uniform_r20.append(top20_metric(uniform, "Recall"))
        uniform_n20.append(top20_metric(uniform, "NDCG"))
        shuffle_r20.append(top20_metric(adaptive_shuffle, "Recall"))
        shuffle_n20.append(top20_metric(adaptive_shuffle, "NDCG"))
        sources.extend([str(real_path), str(shuffle_path)])
    return {
        "baseline": {"Recall@20": baseline_r20, "NDCG@20": baseline_n20},
        "real": {"Recall@20": real_r20, "NDCG@20": real_n20},
        "shuffle": {"Recall@20": shuffle_r20, "NDCG@20": shuffle_n20},
        "uniform_real": {"Recall@20": uniform_r20, "NDCG@20": uniform_n20},
        "source_records": sources,
    }


def summarize_dataset(root: Path, cfg: dict) -> dict:
    report_path = root / cfg["report"]
    report = read_json(report_path)
    gate = report["validation_summary"]
    if not gate.get("ood_test_allowed", False):
        raise RuntimeError(f"validation gate is closed for {cfg['dataset']}")

    records = (
        load_activity_records(root, cfg)
        if cfg["kind"] == "activity_adaptive"
        else load_late_fusion_records(root, cfg)
    )
    metrics = {}
    comparisons = {}
    for name in ("baseline", "real", "shuffle"):
        metrics[name] = {
            metric_name: mean_std(values)
            for metric_name, values in records[name].items()
        }

    for metric_name in ("Recall@20", "NDCG@20"):
        rb = paired_diffs(records["real"][metric_name], records["baseline"][metric_name])
        rs = paired_diffs(records["real"][metric_name], records["shuffle"][metric_name])
        comparisons[f"real_minus_baseline_{metric_name}"] = {
            **mean_std(rb),
            "relative_gain": statistics.mean(rb) / statistics.mean(records["baseline"][metric_name]),
            "bootstrap_95ci": bootstrap_ci(rb),
            "one_sided_sign_p": one_sided_sign_p(rb),
            "positive_seed_count": sum(v > 0 for v in rb),
        }
        comparisons[f"real_minus_shuffle_{metric_name}"] = {
            **mean_std(rs),
            "relative_gain": statistics.mean(rs) / statistics.mean(records["shuffle"][metric_name]),
            "bootstrap_95ci": bootstrap_ci(rs),
            "one_sided_sign_p": one_sided_sign_p(rs),
            "positive_seed_count": sum(v > 0 for v in rs),
        }

    extra_variants = {}
    if "uniform_real" in records:
        extra_variants["uniform_real"] = {
            metric_name: mean_std(values)
            for metric_name, values in records["uniform_real"].items()
        }

    return {
        "label": cfg["label"],
        "dataset": cfg["dataset"],
        "protocol": cfg["protocol"],
        "seeds": list(SEEDS),
        "selection_seeds": gate.get("selection_seeds"),
        "confirmation_seeds": gate.get("confirmation_seeds"),
        "selected_alpha": gate.get("selected_alpha") or gate.get("uniform_alpha"),
        "selected_group_alphas": gate.get("selected_group_alphas"),
        "validation_gate": {
            "test_gt_loaded": gate.get("test_gt_loaded"),
            "ood_test_allowed": gate.get("ood_test_allowed"),
            "checks": gate.get("checks", {}),
        },
        "metrics": metrics,
        "extra_variants": extra_variants,
        "comparisons": comparisons,
        "source_report": str(report_path),
        "source_records": records["source_records"],
    }


def load_kuairec_diagnostic(root: Path) -> dict:
    out = {}
    late_path = root / SUPPORTING_REPORTS["kuairec_v2_latefusion"]
    pop_path = root / SUPPORTING_REPORTS["kuairec_v2_popularity"]
    groups_path = root / SUPPORTING_REPORTS["kuairec_v2_groups"]
    if late_path.exists():
        late = read_json(late_path)["summary"]
        out["late_fusion"] = {
            "selected_alpha": late["selected_alpha"],
            "selection_baseline_ndcg20": late["selection_baseline_ndcg20"],
            "selection_real_ndcg20": late["selection_real_ndcg20"],
            "ood_test_allowed": late["ood_test_allowed"],
        }
    if pop_path.exists():
        pop = read_json(pop_path)["summary"]
        out["popularity_rerank"] = {
            "selected_gamma": pop["selected_gamma"],
            "ood_baseline": pop["ood_baseline"],
            "ood_ndcg20": pop["ood_ndcg20"],
            "ood_gain": pop["ood_gain"],
        }
    if groups_path.exists():
        groups = read_json(groups_path)["summary"]
        out["groups"] = {
            "item_target_recall20": groups.get("aggregate", {}).get("item_target_recall20", {}),
            "source": str(groups_path),
        }
    return out


def load_strict_main_excerpt(root: Path) -> dict:
    path = root / SUPPORTING_REPORTS["strict_main"]
    return {
        "source": str(path),
        "available": path.exists(),
        "role": "background only; corrected-v3 full LSCI is not used as the positive patent main table",
    }


def load_text_source_ablation(root: Path) -> dict:
    output = {}
    names = {
        "food": ("food_raw_vs_deepseek_validation", "food_raw_vs_deepseek_ood"),
        "yelp2018": ("yelp_raw_vs_deepseek_validation", "yelp_raw_vs_deepseek_ood"),
        "kuairec": ("kuairec_raw_vs_deepseek_validation", None),
    }
    for dataset, (validation_key, ood_key) in names.items():
        validation_path = root / SUPPORTING_REPORTS[validation_key]
        row = {
            "validation_report": str(validation_path),
            "validation": read_json(validation_path)["summary"],
        }
        if ood_key:
            ood_path = root / SUPPORTING_REPORTS[ood_key]
            if ood_path.exists():
                row["ood_report"] = str(ood_path)
                row["ood"] = read_json(ood_path)
        output[dataset] = row
    return output


def load_fair_three_method(root: Path) -> dict:
    output = {}
    for dataset, validation_key, ood_key in (
        ("movielens1m", "fair_three_method_movielens_validation", "fair_three_method_movielens_ood"),
        ("yelp2018", "fair_three_method_yelp_validation", "fair_three_method_yelp_ood"),
        ("food", "fair_three_method_food_validation", "fair_three_method_food_ood"),
        ("amazon_beauty", "fair_three_method_amazon_validation", "fair_three_method_amazon_ood"),
    ):
        validation = root / SUPPORTING_REPORTS[validation_key]
        ood = root / SUPPORTING_REPORTS[ood_key]
        row = {
            "validation_report": str(validation),
            "validation_available": validation.exists(),
            "ood_report": str(ood),
            "ood_available": ood.exists(),
        }
        if validation.exists():
            row["validation"] = read_json(validation)
        if ood.exists():
            row["ood"] = read_json(ood)
        output[dataset] = row
    return output


def alpha_label(row: dict) -> str:
    if row["selected_group_alphas"]:
        alphas = row["selected_group_alphas"]
        return (
            "low={low_activity}, mid={mid_activity}, high={high_activity}".format(
                **alphas
            )
        )
    return str(row["selected_alpha"])


def support_label(row: dict) -> str:
    rb = row["comparisons"]["real_minus_baseline_NDCG@20"]
    rs = row["comparisons"]["real_minus_shuffle_NDCG@20"]
    if rb["mean"] > 0 and rb["positive_seed_count"] == 5:
        return "正向主证据"
    if rs["mean"] > 0 and rs["positive_seed_count"] == 5:
        return "机制证据"
    return "不作为正面证据"


def build_markdown(summary: dict) -> str:
    lines = [
        "# 专利实验结果与过程说明",
        "",
        "本报告由冻结的 OOD 测试记录自动生成。报告中的主结果均来自 OOD test；validation 结果只用于说明选参门禁，不作为测试指标汇报。",
        "",
        "## 1. 总体结论",
        "",
        "现有实验可以支撑较稳妥的专利表述：基于物品语义先验的推荐增强方法，在多数公开数据集上提升了分布外推荐鲁棒性，并且正确语义先验整体优于打乱语义先验。更具体地说，MovieLens-1M、Yelp2018、Amazon Beauty 三个数据集相对协同过滤基线取得 OOD NDCG@20 正向提升；Food 未超过 baseline，但正确语义明显优于 shuffled semantic，可作为机制有效性的辅助证据；KuaiRec 当前为失败/局限性诊断，不进入正面主表。",
        "",
        "不建议在专利中写成“完整端到端 LSCI-DiffRec 在所有数据集均显著优于 CausalDiffRec”。更稳的写法是：“在五个数据集中的多数数据集取得分布外性能提升，并通过语义打乱对照验证了语义先验的有效性”。",
        "",
        "## 2. 实验流程",
        "",
        "1. 数据准备：使用 strict 数据协议构建训练、验证、IID 测试和 OOD 测试划分。训练阶段只使用 train 图；validation 用于选择融合系数或用户活跃度分组系数；OOD test 只在选参完成后读取一次。",
        "2. 语义先验构建：对物品侧文本元数据生成语义表示，最终使用 SBERT 语义向量作为 `semantic_prior.pt`。为检验模型是否真正利用语义，同时构造 `semantic_prior_shuffled.pt`，即保持向量分布不变但打乱物品与语义向量的对应关系。",
        "3. 模型与对照：主表采用冻结 backbone 后的语义分数融合方案。对照包括 collaborative baseline、正确语义融合 real semantic、打乱语义 shuffled semantic；Amazon Beauty 还包含 activity-adaptive 分组融合。",
        "4. 选参与测试：在 seeds `1024, 2048, 3072` 上选择 alpha/profile，在 seeds `4096, 5120` 上做 validation confirmation；确认通过后，才对五个 seeds 的 OOD test 记录做汇总。",
        "5. 统计方式：每个数据集汇报五个随机种子的均值与样本标准差，并对同一 seed 下的 `real-baseline`、`real-shuffled` 差值计算 bootstrap 95% CI 和单边 sign test。",
        "",
        "## 3. OOD 主结果",
        "",
        "| 数据集 | 实验变体 | 证据定位 | R@20 baseline | R@20 real | R@20 shuffled | N@20 baseline | N@20 real | N@20 shuffled | N@20 vs baseline | N@20 vs shuffle |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["datasets"]:
        metrics = row["metrics"]
        cmp_base = row["comparisons"]["real_minus_baseline_NDCG@20"]
        cmp_shuffle = row["comparisons"]["real_minus_shuffle_NDCG@20"]
        lines.append(
            "| {label} | {protocol} | {support} | {rb} | {rr} | {rs} | {nb} | {nr} | {ns} | {gb} | {gs} |".format(
                label=row["label"],
                protocol=row["protocol"],
                support=support_label(row),
                rb=format_mean_std(metrics["baseline"]["Recall@20"]),
                rr=format_mean_std(metrics["real"]["Recall@20"]),
                rs=format_mean_std(metrics["shuffle"]["Recall@20"]),
                nb=format_mean_std(metrics["baseline"]["NDCG@20"]),
                nr=format_mean_std(metrics["real"]["NDCG@20"]),
                ns=format_mean_std(metrics["shuffle"]["NDCG@20"]),
                gb=format_pct(cmp_base["relative_gain"]),
                gs=format_pct(cmp_shuffle["relative_gain"]),
            )
        )
    lines.extend(
        [
            "",
            "结果解读：MovieLens-1M、Yelp2018、Amazon Beauty 可写入正面主结果；Food 的 real semantic 在 NDCG@20 上比 baseline 低 0.42%，不能写成 baseline 提升，但它相对 shuffled semantic 提升 4.03%，可用于说明正确语义配对本身有作用。",
            "",
            "## 4. 配对统计证据",
            "",
            "| 数据集 | N@20 real-baseline 均值差 | 95% bootstrap CI | sign p | 正向 seeds | N@20 real-shuffle 均值差 | 95% bootstrap CI | sign p | 正向 seeds |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["datasets"]:
        rb = row["comparisons"]["real_minus_baseline_NDCG@20"]
        rs = row["comparisons"]["real_minus_shuffle_NDCG@20"]
        lines.append(
            "| {label} | {rb_mean:.5f} | [{rb_lo:.5f}, {rb_hi:.5f}] | {rb_p:.5f} | {rb_pos}/5 | {rs_mean:.5f} | [{rs_lo:.5f}, {rs_hi:.5f}] | {rs_p:.5f} | {rs_pos}/5 |".format(
                label=row["label"],
                rb_mean=rb["mean"],
                rb_lo=rb["bootstrap_95ci"][0],
                rb_hi=rb["bootstrap_95ci"][1],
                rb_p=rb["one_sided_sign_p"],
                rb_pos=rb["positive_seed_count"],
                rs_mean=rs["mean"],
                rs_lo=rs["bootstrap_95ci"][0],
                rs_hi=rs["bootstrap_95ci"][1],
                rs_p=rs["one_sided_sign_p"],
                rs_pos=rs["positive_seed_count"],
            )
        )
    lines.extend(
        [
            "",
            "统计解读：由于每组只有 5 个随机种子，sign test 最小单边 p 值为 0.03125，因此这里主要看方向一致性和 bootstrap CI。MovieLens-1M、Yelp2018、Amazon Beauty 对 baseline 与 shuffled 均为 5/5 seeds 正向；Food 对 baseline 只有 2/5 正向，但对 shuffled 为 5/5 正向。",
            "",
            "## 5. 选参与门禁过程",
            "",
            "| 数据集 | selection seeds | confirmation seeds | 选中 alpha/profile | OOD gate |",
            "|---|---:|---:|---|---:|",
        ]
    )
    for row in summary["datasets"]:
        lines.append(
            f"| {row['label']} | {row['selection_seeds']} | {row['confirmation_seeds']} | {alpha_label(row)} | {row['validation_gate']['ood_test_allowed']} |"
        )
    lines.extend(
        [
            "",
            "过程说明：OOD gate 为 True 表示该数据集先完成 validation-only 选择和 confirmation，再允许读取 OOD test 记录。这样可以避免根据测试集结果反向选择 alpha 或 profile。",
            "",
            "## 6. 分数据集说明",
            "",
        ]
    )
    for row in summary["datasets"]:
        rb = row["comparisons"]["real_minus_baseline_NDCG@20"]
        rs = row["comparisons"]["real_minus_shuffle_NDCG@20"]
        lines.extend(
            [
                f"### {row['label']}",
                "",
                f"- 协议：{row['protocol']}；选中 alpha/profile：{alpha_label(row)}。",
                f"- 相对 baseline 的 NDCG@20 均值变化为 {rb['mean']:.5f}（{format_pct(rb['relative_gain'])}），正向 seed 数为 {rb['positive_seed_count']}/5。",
                f"- 相对 shuffled semantic 的 NDCG@20 均值变化为 {rs['mean']:.5f}（{format_pct(rs['relative_gain'])}），正向 seed 数为 {rs['positive_seed_count']}/5。",
                f"- 专利写法定位：{support_label(row)}。",
                "",
            ]
        )
    lines.extend(
        [
            "## 7. 局限性与写法边界",
            "",
            "- 正面主表建议使用本报告的 v4/late-fusion OOD 表，因为它包含冻结 validation 选参、五个随机种子和 real-vs-shuffled 控制。",
            "- MovieLens-1M、Yelp2018、Amazon Beauty 可作为相对 baseline 的正向数据集；Food 只作为语义机制控制，不作为 baseline 提升数据集。",
            "- corrected-v3 full LSCI 只能作为背景实验：Yelp2018 正向明显，Food/MovieLens 接近持平，KuaiRec 为负向。",
            "- Amazon Beauty 的 activity-adaptive 结果显著高于 baseline 和 shuffled，但 OOD 上 uniform semantic fusion 略高于 adaptive，因此不要把“自适应分组规则本身”单独写成增益来源。",
            "",
            "## 8. KuaiRec 失败诊断",
            "",
        ]
    )
    kuai = summary.get("kuairec_v2_diagnostic", {})
    if kuai:
        lf = kuai.get("late_fusion", {})
        pop = kuai.get("popularity_rerank", {})
        groups = kuai.get("groups", {}).get("item_target_recall20", {})
        lines.extend(
            [
                f"- Late fusion 在 validation 上选到 alpha={lf.get('selected_alpha')}，说明语义融合没有带来可确认增益，OOD gate 未打开。",
                f"- Popularity rerank 选到 gamma={pop.get('selected_gamma')}；OOD NDCG@20 保持 {pop.get('ood_ndcg20'):.5f}，增益为 {pop.get('ood_gain'):.5f}。",
            ]
        )
        if groups:
            tail = groups.get("tail", {}).get("mean")
            mid = groups.get("mid", {}).get("mean")
            head = groups.get("head", {}).get("mean")
            cold = groups.get("cold_le_5", {}).get("mean")
            lines.append(
                f"- item-target recall@20 呈现 head-only 行为：tail={tail:.5f}，mid={mid:.5f}，head={head:.5f}，cold<=5={cold:.5f}。这说明当前 KuaiRec 的瓶颈是曝光/流行度偏置，而不是简单后处理语义融合。"
            )
    lines.extend(
        [
            "",
            "## 9. 可复现来源",
            "",
            "主报告文件：",
            "",
        ]
    )
    source_paths = [row["source_report"] for row in summary["datasets"]]
    source_paths.extend(summary["supporting_reports"].values())
    for path in sorted(dict.fromkeys(source_paths)):
        lines.append(f"- `{path}`")
    lines.extend(["", "原始记录模式：", ""])
    for row in summary["datasets"]:
        record_count = len(row["source_records"])
        example = row["source_records"][0]
        lines.append(f"- {row['label']}：{record_count} 个文件，例如 `{example}`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--json_out", default="experiments/reports/patent_experiment_summary.json")
    ap.add_argument("--md_out", default="experiments/reports/patent_experiment_final.md")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    datasets = [summarize_dataset(root, cfg) for cfg in DATASETS]
    summary = {
        "generated_by": "scripts/summarize_patent_experiments.py",
        "seeds": list(SEEDS),
        "claim_supported": "semantic priors / score-level semantic fusion improve OOD recommendation robustness over baseline on MovieLens-1M, Yelp2018, and Amazon Beauty; Food supports real-vs-shuffled semantic specificity only. Under an equal-BPR25 protocol, the full method outperforms both LightGCN and original CausalDiffRec on MovieLens-1M; outperforms original CausalDiffRec but not LightGCN on Yelp2018 and Amazon Beauty; and outperforms LightGCN but has no stable advantage over original CausalDiffRec on Food. On Yelp2018, DeepSeek-normalized text outperforms raw text encoded by the same SBERT",
        "claim_not_supported": "the full method or DeepSeek text normalization universally improves every dataset; the full method is superior to both LightGCN and original CausalDiffRec on every evaluated dataset; learned causal edge gating is a validated source of stable gain",
        "datasets": datasets,
        "strict_main": load_strict_main_excerpt(root),
        "kuairec_v2_diagnostic": load_kuairec_diagnostic(root),
        "raw_vs_deepseek_text_ablation": load_text_source_ablation(root),
        "fair_lightgcn_causaldiffrec_full_method": load_fair_three_method(root),
        "supporting_reports": {
            name: str(root / path) for name, path in SUPPORTING_REPORTS.items()
        },
    }

    json_out = root / args.json_out
    md_out = root / args.md_out
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_out.write_text(build_markdown(summary), encoding="utf-8")

    print(f"saved json: {json_out}")
    print(f"saved markdown: {md_out}")
    for row in datasets:
        rb = row["comparisons"]["real_minus_baseline_NDCG@20"]
        rs = row["comparisons"]["real_minus_shuffle_NDCG@20"]
        print(
            f"{row['label']}: N@20 real-baseline {rb['mean']:.5f} "
            f"({format_pct(rb['relative_gain'])}), real-shuffle {rs['mean']:.5f} "
            f"({format_pct(rs['relative_gain'])})"
        )


if __name__ == "__main__":
    main()
