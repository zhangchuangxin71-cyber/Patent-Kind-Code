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
        "scope": "Yelp2018 and MovieLens-1M A0/A1/A2 training plus A3 frozen-score evaluation",
        "datasets": datasets,
        "data_leakage_audit": data_audit,
        "reproducibility_evidence": reproducibility,
        "claim_rule": (
            "Attribute ConditionFusion only from A1-A0, InfoNCE only from A2-A1, and score fusion "
            "only from A3-A2. Report negative or mixed directions without suppressing them."
        ),
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
        lines.extend(["", "| NDCG@20比较 | 均值差 | 相对提升 | 正向种子 | 符号检验p |", "|---|---:|---:|---:|---:|"])
        for key in (
            "A1_minus_A0_NDCG@20", "A2_minus_A1_NDCG@20", "A3_minus_A2_NDCG@20",
            "A3_minus_CausalDiffRec_NDCG@20", "A3_minus_LightGCN_NDCG@20",
        ):
            if key not in report["comparisons"]:
                continue
            row = report["comparisons"][key]
            lines.append(
                f"| {key} | {row['mean_difference']:+.6f} | {row['relative_gain_percent']:+.2f}% | "
                f"{row['positive_seed_count']}/5 | {row['one_sided_exact_sign_p']:.5f} |"
            )
        control = report["five_shuffle_control"]
        lines.extend([
            "",
            f"真实语义超过五份打乱语义的比例：**{control['real_exceeds_shuffle_count']}/"
            f"{control['comparisons']} ({control['real_exceeds_shuffle_fraction']:.1%})**。", "",
        ])
    lines.extend([
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
