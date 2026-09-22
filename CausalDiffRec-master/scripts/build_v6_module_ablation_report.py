#!/usr/bin/env python
"""Build the combined patent-facing v6 module-ablation report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--out", default="experiments/reports/v6_module_ablation_final.json")
    parser.add_argument("--markdown", default="experiments/reports/v6_module_ablation_final.md")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    reports = root / "experiments" / "reports"
    datasets = {}
    for dataset in ("yelp2018", "movielens1m"):
        datasets[dataset] = json.loads(
            (reports / f"{dataset}_v6_module_ablation_summary.json").read_text(encoding="utf-8")
        )
    data_audit = json.loads((reports / "strict_data_leakage_audit.json").read_text(encoding="utf-8"))
    reproducibility = json.loads((reports / "reproducibility_evidence.json").read_text(encoding="utf-8"))
    output = {
        "protocol_version": "corrected_v6_patent_training_module_ablation_final",
        "scope": (
            "Yelp2018 and MovieLens-1M A0/A1/A2 training, A3 frozen-score evaluation, "
            "and B0/B1/B2 fixed-inference-form checkpoint comparison"
        ),
        "datasets": datasets,
        "data_leakage_audit": data_audit,
        "reproducibility_evidence": reproducibility,
        "claim_rule": (
            "Attribute ConditionFusion only from A1-A0, InfoNCE only from A2-A1, and score fusion "
            "only from A3-A2. Report negative or mixed directions without suppressing them."
        ),
        "patent_evidence_boundary": {
            "supported": [
                "Frozen semantic score fusion improves over the same A2 checkpoint without fusion on both datasets.",
                "Correct item-semantic alignment improves over five-permutation means for all five model seeds.",
                "A3 improves over CausalDiffRec on both datasets and over LightGCN on MovieLens-1M.",
                "Yelp supports a DeepSeek-rewrite implementation example; it does not establish universal LLM gains.",
            ],
            "not_supported": [
                "ConditionFusion independently improves recommendation quality.",
                "InfoNCE independently improves recommendation quality.",
                "ConditionFusion and InfoNCE jointly improve the final inference form.",
                "A3 outperforms LightGCN on every dataset.",
                "LLM rewriting outperforms raw text on every dataset.",
                "Learned edge gating is effective under the v6 experiment, where it is disabled.",
            ],
        },
    }
    (root / args.out).write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# corrected-v6 训练阶段模块严格消融终报", "",
        "A0/A1/A2 均使用五个模型种子、固定数据划分、25 个外层轮次和每轮一次完整 BPR；"
        "A3不重新训练，严格复用A2检查点，仅改变冻结后的评分融合权重。", "",
    ]
    for dataset, report in datasets.items():
        lines.extend([
            f"## {dataset}", "",
            f"验证集冻结 α：**{report['selected_alpha']:g}**；A3复用A2：**{report['a3_reuses_a2_checkpoint']}**。", "",
            "| 组别 | Recall@20 | NDCG@20 |", "|---|---:|---:|",
        ])
        for arm in ("A0", "A1", "A2", "A3"):
            recall = report["metrics"][arm]["Recall@20"]
            ndcg = report["metrics"][arm]["NDCG@20"]
            lines.append(
                f"| {arm} | {recall['mean']:.6f} ± {recall['sample_std']:.6f} | "
                f"{ndcg['mean']:.6f} ± {ndcg['sample_std']:.6f} |"
            )
        lines.extend(["", "| NDCG@20比较 | 均值差 | 相对提升 | 正/负/并列 | 符号检验p |", "|---|---:|---:|---:|---:|"])
        for key in (
            "A1_minus_A0_NDCG@20", "A2_minus_A1_NDCG@20", "A3_minus_A2_NDCG@20",
            "A3_minus_CausalDiffRec_NDCG@20", "A3_minus_LightGCN_NDCG@20",
        ):
            if key not in report["comparisons"]:
                continue
            row = report["comparisons"][key]
            lines.append(
                f"| {key} | {row['mean_difference']:+.6f} | {row['relative_gain_percent']:+.2f}% | "
                f"{row['positive_seed_count']}/{row['negative_seed_count']}/{row['tied_seed_count']} | "
                f"{row['one_sided_exact_sign_p']:.5f} |"
            )
        lines.extend(["", "### 固定最终推理形式后的检查点比较", "", f"检查点与冻结 α 一致性审计：**{report['inference_form_checkpoint_audit']['all_checks_pass']}**。", "", "| 组别 | Recall@20 | NDCG@20 |", "|---|---:|---:|"])
        for arm in ("B0", "B1", "B2"):
            recall = report["inference_form_metrics"][arm]["Recall@20"]
            ndcg = report["inference_form_metrics"][arm]["NDCG@20"]
            lines.append(
                f"| {arm} | {recall['mean']:.6f} ± {recall['sample_std']:.6f} | "
                f"{ndcg['mean']:.6f} ± {ndcg['sample_std']:.6f} |"
            )
        lines.extend(["", "| NDCG@20比较 | 均值差 | 相对提升 | 正/负/并列 | 符号检验p |", "|---|---:|---:|---:|---:|"])
        for key in ("B1_minus_B0_NDCG@20", "B2_minus_B1_NDCG@20", "B2_minus_B0_NDCG@20"):
            row = report["inference_form_comparisons"][key]
            lines.append(
                f"| {key} | {row['mean_difference']:+.6f} | {row['relative_gain_percent']:+.2f}% | "
                f"{row['positive_seed_count']}/{row['negative_seed_count']}/{row['tied_seed_count']} | "
                f"{row['one_sided_exact_sign_p']:.5f} |"
            )
        control = report["five_shuffle_control"]
        aggregate = control["paired_real_minus_five_shuffle_mean"]
        lines.extend([
            "",
            f"真实语义相对每个模型种子的5次打乱均值：**{aggregate['positive_seed_count']}正/"
            f"{aggregate['negative_seed_count']}负/{aggregate['tied_seed_count']}并列，p={aggregate['one_sided_exact_sign_p']:.5f}**。",
            f"未聚合的 {control['real_exceeds_shuffle_count']}/{control['comparisons']} 仅作描述，不视为25个独立样本。", "",
        ])
    lines.extend([
        "## 可支持与不可支持的技术效果", "",
        "可支持：", "",
        "- 冻结后的语义评分融合在两个数据集上均优于同一A2检查点的无评分融合结果。",
        "- 按五个模型种子聚合后，正确对齐的物品语义均优于五次置乱的均值。",
        "- 完整A3在两个数据集上均优于CausalDiffRec，并在MovieLens-1M上优于LightGCN。",
        "- Yelp可作为DeepSeek文本改写的正向实施例，但不支持LLM改写在所有数据集上均有增益。", "",
        "不可支持：", "",
        "- ConditionFusion、InfoNCE或二者联合训练能独立提高最终推理性能。B0/B1/B2对照未给出稳定正向证据。",
        "- 完整方法在所有数据集上均优于LightGCN。Yelp上A3低于LightGCN。",
        "- 可学习边门控的有效性。v6主实验中边门控关闭。", "",
        "## 审计结论", "",
        f"- 数据结构与无泄漏检查全部通过：**{data_audit['all_structural_checks_pass']}**。",
        "- Food 的配方文本不是按交互时点截断的语义快照，不能描述成完全时间严格的文本快照。",
        "- DeepSeek和SBERT均为离线成本；在线评分不调用大语言模型，也不执行扩散采样。",
        "- 第三方数据本体按来源许可下载重建，公开仓库提供哈希、审计、代码、结果和精简评价检查点。",
    ])
    (root / args.markdown).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(root / args.markdown)


if __name__ == "__main__":
    main()
