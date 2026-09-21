#!/usr/bin/env python
"""Build the Chinese patent addendum for the two corrected-v5 experiments."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with (ROOT / path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ms(value):
    return f"{value['mean']:.6f} ± {value['sample_std']:.6f}"


def build():
    text = {}
    for dataset in ("food", "yelp2018"):
        text[dataset] = {
            "validation": read(f"experiments/reports/{dataset}_v5_raw_vs_deepseek_validation.json"),
            "ood": read(f"experiments/reports/{dataset}_v5_raw_vs_deepseek_ood.json"),
        }
    kuai = read("experiments/reports/kuairec_v5_raw_vs_deepseek_validation.json")
    fair_audit = read("experiments/reports/movielens1m_v5_fair_bpr25_validation_audit.json")
    fair = read("experiments/reports/movielens1m_v5_fair_bpr25_method_comparison.json")

    y = text["yelp2018"]["ood"]["aggregate_top20"]["NDCG"]
    f = text["food"]["ood"]["aggregate_top20"]["NDCG"]
    m = fair["aggregate_top20"]["NDCG"]
    yv = text["yelp2018"]["validation"]["summary"]
    fv = text["food"]["validation"]["summary"]
    kv = kuai["summary"]

    priors = [
        ("Food", "data_strict/processed/food/v1_strict/features/semantic_prior.pt",
         "data_strict/processed/food/v1_strict/features/semantic_prior_raw_sbert.pt"),
        ("Yelp2018", "data_strict/processed/yelp2018/v1_strict/features/semantic_prior.pt",
         "data_strict/processed/yelp2018/v1_strict/features/semantic_prior_raw_sbert.pt"),
        ("KuaiRec", "data_strict/processed/kuairec/v1_strict/features/semantic_prior.pt",
         "data_strict/processed/kuairec/v1_strict/features/semantic_prior_raw_sbert.pt"),
    ]

    lines = [
        "# 新增专利对比实验详细报告",
        "",
        "生成日期：2026-09-05",
        "",
        "本报告专门补齐两个原有证据缺口：（1）原始文本与 DeepSeek 优化文本的独立贡献；（2）LightGCN、原始 CausalDiffRec 与完整方法在统一协议下的比较。所有负结果与适用边界均保留，不根据 OOD 测试结果反向调参。",
        "",
        "## 一、原始文本与 DeepSeek 优化文本对照",
        "",
        "### 1. 实验要回答的问题",
        "",
        "已有 real-vs-shuffled 实验只能证明“正确物品语义对应有用”，不能分离 DeepSeek 改写和 SBERT 编码的贡献。本实验直接比较：",
        "",
        "- A 组：原始物品文本 -> 同一 SBERT -> 384 维 raw prior；",
        "- B 组：原始物品文本 -> DeepSeek 规范化摘要 -> 同一 SBERT -> 384 维 LLM prior。",
        "",
        "### 2. 保持不变与实际改变的因素",
        "",
        "| 类别 | 具体设置 |",
        "|---|---|",
        "| 保持不变 | strict 数据划分、每个 seed 的同一冻结 checkpoint、全候选物品集、训练物品屏蔽、Top-20 评价函数 |",
        "| 保持不变 | SBERT `sentence-transformers/all-MiniLM-L6-v2`、384 维、L2 归一化、100% 覆盖率 |",
        "| 保持不变 | 用 train 交互的物品向量均值构造用户语义画像；协同分数和语义分数均按用户 z-score |",
        "| 唯一有意改变 | SBERT 的输入是原始拼接文本，还是 DeepSeek 改写的规范化摘要 |",
        "",
        "### 3. 语义文件完整性",
        "",
        "| 数据集 | LLM prior SHA-256 | raw prior SHA-256 |",
        "|---|---|---|",
    ]
    for label, llm_path, raw_path in priors:
        lines.append(f"| {label} | `{sha256(llm_path)}` | `{sha256(raw_path)}` |")
    lines.extend([
        "",
        "三组 prior 的物品数分别为 Food 6,309、Yelp2018 12,185、KuaiRec 10,612，两个文本来源的向量形状、SBERT 型号与 semantic mask 逐项一致。生成 raw prior 时写入独立文件 `semantic_prior_raw_sbert.pt`，没有覆盖原 LLM prior。",
        "",
        "### 4. validation-only 选参与门禁",
        "",
        "`alpha` 候选范围为 0、0.025、0.05、0.075、0.1、0.15、0.2、0.25、0.5、0.75、1、1.25、1.5、2。种子 1024/2048/3072 负责选参，4096/5120 负责独立确认。此阶段 `load_test_gt=False`。",
        "",
        "| 数据集 | LLM 选中 alpha | raw 独立选中 alpha | selection LLM/raw（共享 alpha） | confirmation LLM/raw（共享 alpha） | 是否允许 OOD |",
        "|---|---:|---:|---:|---:|---|",
        f"| Food | {fv['selected_alpha']['shared_llm_selected']} | {fv['selected_alpha']['raw_independently_selected']} | {fv['selection']['llm_shared']:.6f}/{fv['selection']['raw_shared']:.6f} | {fv['confirmation']['llm_shared']:.6f}/{fv['confirmation']['raw_shared']:.6f} | {fv['ood_test_allowed']} |",
        f"| Yelp2018 | {yv['selected_alpha']['shared_llm_selected']} | {yv['selected_alpha']['raw_independently_selected']} | {yv['selection']['llm_shared']:.6f}/{yv['selection']['raw_shared']:.6f} | {yv['confirmation']['llm_shared']:.6f}/{yv['confirmation']['raw_shared']:.6f} | {yv['ood_test_allowed']} |",
        f"| KuaiRec v2 | {kv['selected_alpha']['shared_llm_selected']} | {kv['selected_alpha']['raw_independently_selected']} | {kv['selection']['llm_shared']:.6f}/{kv['selection']['raw_shared']:.6f} | {kv['confirmation']['llm_shared']:.6f}/{kv['confirmation']['raw_shared']:.6f} | {kv['ood_test_allowed']} |",
        "",
        "KuaiRec 的最优权重为 0，因此根据门禁不读取该项消融的 OOD 标签。",
        "",
        "### 5. 一次性 OOD 结果",
        "",
        "主分析将 LLM 选定的同一 alpha 同时用于两组，以最大程度隔离“文本来源”这一变量；辅助分析给两组相同的搜索范围，但各自使用自己在 validation 选定的 alpha。",
        "",
        "| 数据集 | baseline | raw（共享 alpha） | raw（自选 alpha） | DeepSeek | DeepSeek-raw（共享） | DeepSeek-raw（各自选参） |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| Yelp2018 | {y['collaborative_baseline']['mean']:.6f} | {y['raw_text_sbert_shared_alpha']['mean']:.6f} | {y['raw_text_sbert_own_validation_alpha']['mean']:.6f} | **{y['deepseek_text_sbert']['mean']:.6f}** | **{y['deepseek_minus_raw_shared_alpha']['mean']:+.6f}** | **{y['deepseek_minus_raw_each_own_alpha']['mean']:+.6f}** |",
        f"| Food | {f['collaborative_baseline']['mean']:.6f} | {f['raw_text_sbert_shared_alpha']['mean']:.6f} | {f['raw_text_sbert_own_validation_alpha']['mean']:.6f} | {f['deepseek_text_sbert']['mean']:.6f} | {f['deepseek_minus_raw_shared_alpha']['mean']:+.6f} | {f['deepseek_minus_raw_each_own_alpha']['mean']:+.6f} |",
        "",
        f"Yelp2018 共享 alpha 主对比中，DeepSeek 相对 raw 提升 {y['deepseek_vs_raw_shared_alpha_relative_percent']:.2f}%，{y['deepseek_minus_raw_shared_alpha']['positive_seeds']}/5 种子胜出；配对差的描述性 95% t 区间为 [{y['deepseek_minus_raw_shared_alpha']['paired_delta_t95_ci_approx'][0]:.6f}, {y['deepseek_minus_raw_shared_alpha']['paired_delta_t95_ci_approx'][1]:.6f}]，单侧精确符号检验 p={y['deepseek_minus_raw_shared_alpha']['one_sided_exact_sign_p_positive']:.5f}。",
        "",
        "结论边界：Yelp2018 支持 DeepSeek 改写的独立正向贡献；Food 的 OOD 方向反转，必须作为负结果保留；KuaiRec 未通过 validation 门禁。不得写“DeepSeek 在所有数据集都更好”。",
        "",
        "## 二、LightGCN、CausalDiffRec 与完整方法公平比较",
        "",
        "### 1. 三种方法的准确定义",
        "",
        "- LightGCN：仅使用协同交互图和 BPR 排序损失，不使用扩散、因果环境或语义。",
        "- 原始 CausalDiffRec（E0）：VGAE + 扩散生成 + 原图编辑/环境建模 + 持久 LightGCN 排序头，不使用语义先验。",
        "- 完整方法：因果扩散/LSCI backbone + `ConditionFusion` 语义条件 + `lambda_sem=0.1` 的 InfoNCE + 仅基于 train 交互的用户语义画像分数融合，`alpha=0.75`。",
        "",
        "learned edge gate 在早期消融中未稳定超过 random gate，因此本次“完整方法”指已得到证据支持的可实施组合，`edge_gate_mode=none`；不把未得到支持的 learned gate 包装成已验证优势。",
        "",
        "### 2. 统一的实验条件",
        "",
        "| 条件 | 三种方法的共享设置 |",
        "|---|---|",
        "| 数据 | MovieLens-1M `v1_strict`，同一 train/validation/OOD test |",
        "| 候选集 | 全物品候选，屏蔽 train 已交互物品 |",
        "| 评价 | Recall@20、NDCG@20，同一评价函数 |",
        "| 选模型 | 只按 validation NDCG@20 保存 checkpoint；训练记录 `best_metrics=None` |",
        "| 随机种子 | 1024、2048、3072、4096、5120 |",
        "| BPR 预算 | 每种方法 25 次完整 BPR pass |",
        "| 排序参数 | 8 维嵌入、3 层 LightGCN、batch=1024、lr=0.001、L2=0.001 |",
        "",
        f"公平性审计的 {len(fair_audit['checks'])} 项检查全部为 true，`ood_test_allowed={fair_audit['ood_test_allowed']}`。“等预算”指排序学习的完整 BPR pass 数相同；CausalDiffRec 和完整方法仍执行各自架构所需的表示学习目标，这属于方法本身。",
        "",
        "### 3. 冻结后的一次性 OOD 结果",
        "",
        "| 方法 | Recall@20（均值±样本标准差） | NDCG@20（均值±样本标准差） |",
        "|---|---:|---:|",
        f"| LightGCN | {ms(fair['aggregate_top20']['Recall']['lightgcn'])} | {ms(m['lightgcn'])} |",
        f"| 原始 CausalDiffRec | {ms(fair['aggregate_top20']['Recall']['causaldiffrec'])} | {ms(m['causaldiffrec'])} |",
        f"| 完整方法 | {ms(fair['aggregate_top20']['Recall']['full_method'])} | {ms(m['full_method'])} |",
        "",
        f"完整方法相对原始 CausalDiffRec 的 NDCG@20 差为 {m['full_minus_causaldiffrec']['mean']:+.6f}（{m['full_vs_causaldiffrec_relative_percent']:+.2f}%），正向种子 {m['full_minus_causaldiffrec']['positive_seeds']}/5；相对 LightGCN 的差为 {m['full_minus_lightgcn']['mean']:+.6f}（{m['full_vs_lightgcn_relative_percent']:+.2f}%），正向种子 {m['full_minus_lightgcn']['positive_seeds']}/5。",
        "",
        "专利中必须根据上表实际方向描述：可以声称的优势只限于配对差为正的对照；如果完整方法低于 LightGCN，则必须明确保留该结果，不得用不同训练预算的其他表替换。",
        "",
        "### 4. 预算口径的额外说明",
        "",
        "25-pass 表的目的是与 CausalDiffRec 原实验的 25 个 outer epochs 对齐，同时让每个 outer epoch 的排序头完整训练一遍。项目另有 MovieLens LightGCN BPR200 的较长预算曲线；不得把 BPR200 分数与本表 BPR25 分数并列宣称公平优势。",
        "",
        "## 三、可复现命令与证据",
        "",
        "```bash",
        "bash scripts/run_raw_sbert_prior_build.sh",
        "bash scripts/run_text_source_validation.sh",
        "bash scripts/run_text_source_ood.sh food data_strict/processed/food/v1_strict experiments/reports/food_v5_raw_vs_deepseek_validation.json 0",
        "bash scripts/run_text_source_ood.sh yelp2018 data_strict/processed/yelp2018/v1_strict experiments/reports/yelp2018_v5_raw_vs_deepseek_validation.json 0",
        "bash scripts/run_movielens_v5_fair_bpr25_training.sh 0",
        "bash scripts/run_movielens_v5_fair_bpr25_ood.sh 0",
        "```",
        "",
        "核心机器可读证据：",
        "",
        "- `experiments/reports/yelp2018_v5_raw_vs_deepseek_ood.json`；",
        "- `experiments/reports/food_v5_raw_vs_deepseek_ood.json`；",
        "- `experiments/reports/kuairec_v5_raw_vs_deepseek_validation.json`；",
        "- `experiments/reports/movielens1m_v5_fair_bpr25_validation_audit.json`；",
        "- `experiments/reports/movielens1m_v5_fair_bpr25_method_comparison.json`。",
        "",
        "## 四、给专利撰写 AI 的结论边界",
        "",
        "1. 可以把 Yelp2018 的 raw-vs-DeepSeek 结果作为大语言模型文本规范化的独立贡献实施例。",
        "2. 必须同时说明 Food 方向不一致、KuaiRec 未通过门禁；不得概括为普适性提升。",
        "3. 三方法表的“完整方法”已明确启用语义条件、InfoNCE 和分数融合，但不启用未获支持的 learned edge gate。",
        "4. 只根据统一 BPR25 表的实际配对方向陈述相对 LightGCN 或 CausalDiffRec 的优势，不得引用不同数据划分或不同预算的外部分数填表。",
        "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build(), encoding="utf-8")
    print("Saved", out)


if __name__ == "__main__":
    main()
