# 专利实验结果与过程说明

本报告由冻结的 OOD 测试记录自动生成。报告中的主结果均来自 OOD test；validation 结果只用于说明选参门禁，不作为测试指标汇报。

## 1. 总体结论

现有实验可以支撑较稳妥的专利表述：基于物品语义先验的推荐增强方法，在多数公开数据集上提升了分布外推荐鲁棒性，并且正确语义先验整体优于打乱语义先验。更具体地说，MovieLens-1M、Yelp2018、Amazon Beauty 三个数据集相对协同过滤基线取得 OOD NDCG@20 正向提升；Food 未超过 baseline，但正确语义明显优于 shuffled semantic，可作为机制有效性的辅助证据；KuaiRec 当前为失败/局限性诊断，不进入正面主表。

不建议在专利中写成“完整端到端 LSCI-DiffRec 在所有数据集均显著优于 CausalDiffRec”。更稳的写法是：“在五个数据集中的多数数据集取得分布外性能提升，并通过语义打乱对照验证了语义先验的有效性”。

## 2. 实验流程

1. 数据准备：使用 strict 数据协议构建训练、验证、IID 测试和 OOD 测试划分。训练阶段只使用 train 图；validation 用于选择融合系数或用户活跃度分组系数；OOD test 只在选参完成后读取一次。
2. 语义先验构建：对物品侧文本元数据生成语义表示，最终使用 SBERT 语义向量作为 `semantic_prior.pt`。为检验模型是否真正利用语义，同时构造 `semantic_prior_shuffled.pt`，即保持向量分布不变但打乱物品与语义向量的对应关系。
3. 模型与对照：主表采用冻结 backbone 后的语义分数融合方案。对照包括 collaborative baseline、正确语义融合 real semantic、打乱语义 shuffled semantic；Amazon Beauty 还包含 activity-adaptive 分组融合。
4. 选参与测试：在 seeds `1024, 2048, 3072` 上选择 alpha/profile，在 seeds `4096, 5120` 上做 validation confirmation；确认通过后，才对五个 seeds 的 OOD test 记录做汇总。
5. 统计方式：每个数据集汇报五个随机种子的均值与样本标准差，并对同一 seed 下的 `real-baseline`、`real-shuffled` 差值计算 bootstrap 95% CI 和单边 sign test。

## 3. OOD 主结果

| 数据集 | 实验变体 | 证据定位 | R@20 baseline | R@20 real | R@20 shuffled | N@20 baseline | N@20 real | N@20 shuffled | N@20 vs baseline | N@20 vs shuffle |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MovieLens-1M | corrected-v4 pure LightGCN BPR200 late fusion | 正向主证据 | 0.06928 +/- 0.00218 | 0.07165 +/- 0.00261 | 0.06720 +/- 0.00239 | 0.06831 +/- 0.00191 | 0.07037 +/- 0.00201 | 0.06700 +/- 0.00185 | 3.02% | 5.04% |
| Yelp2018 | corrected-v4 LSCI-stage late fusion | 正向主证据 | 0.01731 +/- 0.00146 | 0.01862 +/- 0.00106 | 0.01694 +/- 0.00136 | 0.01345 +/- 0.00115 | 0.01452 +/- 0.00080 | 0.01314 +/- 0.00115 | 8.00% | 10.52% |
| Food | corrected-v4 pure LightGCN late fusion | 机制证据 | 0.04690 +/- 0.00065 | 0.04679 +/- 0.00036 | 0.04431 +/- 0.00080 | 0.02780 +/- 0.00017 | 0.02768 +/- 0.00014 | 0.02661 +/- 0.00026 | -0.42% | 4.03% |
| Amazon Beauty | corrected-v4 activity-adaptive semantic fusion | 正向主证据 | 0.03847 +/- 0.00117 | 0.07576 +/- 0.00159 | 0.03189 +/- 0.00104 | 0.01824 +/- 0.00109 | 0.03834 +/- 0.00117 | 0.01466 +/- 0.00051 | 110.24% | 161.56% |

结果解读：MovieLens-1M、Yelp2018、Amazon Beauty 可写入正面主结果；Food 的 real semantic 在 NDCG@20 上比 baseline 低 0.42%，不能写成 baseline 提升，但它相对 shuffled semantic 提升 4.03%，可用于说明正确语义配对本身有作用。

## 4. 配对统计证据

| 数据集 | N@20 real-baseline 均值差 | 95% bootstrap CI | sign p | 正向 seeds | N@20 real-shuffle 均值差 | 95% bootstrap CI | sign p | 正向 seeds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MovieLens-1M | 0.00206 | [0.00187, 0.00231] | 0.03125 | 5/5 | 0.00338 | [0.00316, 0.00362] | 0.03125 | 5/5 |
| Yelp2018 | 0.00108 | [0.00074, 0.00142] | 0.03125 | 5/5 | 0.00138 | [0.00103, 0.00173] | 0.03125 | 5/5 |
| Food | -0.00012 | [-0.00032, 0.00005] | 0.81250 | 2/5 | 0.00107 | [0.00094, 0.00122] | 0.03125 | 5/5 |
| Amazon Beauty | 0.02010 | [0.01933, 0.02110] | 0.03125 | 5/5 | 0.02368 | [0.02274, 0.02462] | 0.03125 | 5/5 |

统计解读：由于每组只有 5 个随机种子，sign test 最小单边 p 值为 0.03125，因此这里主要看方向一致性和 bootstrap CI。MovieLens-1M、Yelp2018、Amazon Beauty 对 baseline 与 shuffled 均为 5/5 seeds 正向；Food 对 baseline 只有 2/5 正向，但对 shuffled 为 5/5 正向。

## 5. 选参与门禁过程

| 数据集 | selection seeds | confirmation seeds | 选中 alpha/profile | OOD gate |
|---|---:|---:|---|---:|
| MovieLens-1M | [1024, 2048, 3072] | [4096, 5120] | 0.1 | True |
| Yelp2018 | [1024, 2048, 3072] | [4096, 5120] | 0.25 | True |
| Food | [1024, 2048, 3072] | [4096, 5120] | 0.25 | True |
| Amazon Beauty | [1024, 2048, 3072] | [4096, 5120] | low=1.0, mid=1.0, high=0.5 | True |

过程说明：OOD gate 为 True 表示该数据集先完成 validation-only 选择和 confirmation，再允许读取 OOD test 记录。这样可以避免根据测试集结果反向选择 alpha 或 profile。

## 6. 分数据集说明

### MovieLens-1M

- 协议：corrected-v4 pure LightGCN BPR200 late fusion；选中 alpha/profile：0.1。
- 相对 baseline 的 NDCG@20 均值变化为 0.00206（3.02%），正向 seed 数为 5/5。
- 相对 shuffled semantic 的 NDCG@20 均值变化为 0.00338（5.04%），正向 seed 数为 5/5。
- 专利写法定位：正向主证据。

### Yelp2018

- 协议：corrected-v4 LSCI-stage late fusion；选中 alpha/profile：0.25。
- 相对 baseline 的 NDCG@20 均值变化为 0.00108（8.00%），正向 seed 数为 5/5。
- 相对 shuffled semantic 的 NDCG@20 均值变化为 0.00138（10.52%），正向 seed 数为 5/5。
- 专利写法定位：正向主证据。

### Food

- 协议：corrected-v4 pure LightGCN late fusion；选中 alpha/profile：0.25。
- 相对 baseline 的 NDCG@20 均值变化为 -0.00012（-0.42%），正向 seed 数为 2/5。
- 相对 shuffled semantic 的 NDCG@20 均值变化为 0.00107（4.03%），正向 seed 数为 5/5。
- 专利写法定位：机制证据。

### Amazon Beauty

- 协议：corrected-v4 activity-adaptive semantic fusion；选中 alpha/profile：low=1.0, mid=1.0, high=0.5。
- 相对 baseline 的 NDCG@20 均值变化为 0.02010（110.24%），正向 seed 数为 5/5。
- 相对 shuffled semantic 的 NDCG@20 均值变化为 0.02368（161.56%），正向 seed 数为 5/5。
- 专利写法定位：正向主证据。

## 7. 局限性与写法边界

- 正面主表建议使用本报告的 v4/late-fusion OOD 表，因为它包含冻结 validation 选参、五个随机种子和 real-vs-shuffled 控制。
- MovieLens-1M、Yelp2018、Amazon Beauty 可作为相对 baseline 的正向数据集；Food 只作为语义机制控制，不作为 baseline 提升数据集。
- corrected-v3 full LSCI 只能作为背景实验：Yelp2018 正向明显，Food/MovieLens 接近持平，KuaiRec 为负向。
- Amazon Beauty 的 activity-adaptive 结果显著高于 baseline 和 shuffled，但 OOD 上 uniform semantic fusion 略高于 adaptive，因此不要把“自适应分组规则本身”单独写成增益来源。

## 8. KuaiRec 失败诊断

- Late fusion 在 validation 上选到 alpha=0.0，说明语义融合没有带来可确认增益，OOD gate 未打开。
- Popularity rerank 选到 gamma=0.0；OOD NDCG@20 保持 0.06807，增益为 0.00000。
- item-target recall@20 呈现 head-only 行为：tail=0.00000，mid=0.00000，head=0.09354，cold<=5=0.00000。这说明当前 KuaiRec 的瓶颈是曝光/流行度偏置，而不是简单后处理语义融合。

## 9. 可复现来源

主报告文件：

- `/media/p520/F2FAF016FAEFD53F/ZCX-2026.5.27/专利/CausalDiffRec-master/experiments/reports/amazon_beauty_v4_activity_adaptive_ood.json`
- `/media/p520/F2FAF016FAEFD53F/ZCX-2026.5.27/专利/CausalDiffRec-master/experiments/reports/amazon_beauty_v5_fair_bpr25_method_comparison.json`
- `/media/p520/F2FAF016FAEFD53F/ZCX-2026.5.27/专利/CausalDiffRec-master/experiments/reports/amazon_beauty_v5_fair_bpr25_validation_audit.json`
- `/media/p520/F2FAF016FAEFD53F/ZCX-2026.5.27/专利/CausalDiffRec-master/experiments/reports/food_v4_uniform_ood.json`
- `/media/p520/F2FAF016FAEFD53F/ZCX-2026.5.27/专利/CausalDiffRec-master/experiments/reports/food_v5_fair_bpr25_method_comparison.json`
- `/media/p520/F2FAF016FAEFD53F/ZCX-2026.5.27/专利/CausalDiffRec-master/experiments/reports/food_v5_fair_bpr25_validation_audit.json`
- `/media/p520/F2FAF016FAEFD53F/ZCX-2026.5.27/专利/CausalDiffRec-master/experiments/reports/food_v5_raw_vs_deepseek_ood.json`
- `/media/p520/F2FAF016FAEFD53F/ZCX-2026.5.27/专利/CausalDiffRec-master/experiments/reports/food_v5_raw_vs_deepseek_validation.json`
- `/media/p520/F2FAF016FAEFD53F/ZCX-2026.5.27/专利/CausalDiffRec-master/experiments/reports/kuairec_v2_latefusion_groups_real_a05.json`
- `/media/p520/F2FAF016FAEFD53F/ZCX-2026.5.27/专利/CausalDiffRec-master/experiments/reports/kuairec_v2_latefusion_validation.json`
- `/media/p520/F2FAF016FAEFD53F/ZCX-2026.5.27/专利/CausalDiffRec-master/experiments/reports/kuairec_v2_popularity_rerank.json`
- `/media/p520/F2FAF016FAEFD53F/ZCX-2026.5.27/专利/CausalDiffRec-master/experiments/reports/kuairec_v5_raw_vs_deepseek_validation.json`
- `/media/p520/F2FAF016FAEFD53F/ZCX-2026.5.27/专利/CausalDiffRec-master/experiments/reports/movielens1m_pure_lightgcn_v4_bpr200_ood.json`
- `/media/p520/F2FAF016FAEFD53F/ZCX-2026.5.27/专利/CausalDiffRec-master/experiments/reports/movielens1m_v5_fair_bpr25_method_comparison.json`
- `/media/p520/F2FAF016FAEFD53F/ZCX-2026.5.27/专利/CausalDiffRec-master/experiments/reports/movielens1m_v5_fair_bpr25_validation_audit.json`
- `/media/p520/F2FAF016FAEFD53F/ZCX-2026.5.27/专利/CausalDiffRec-master/experiments/reports/strict_ood_e0_stage1_e3_final.md`
- `/media/p520/F2FAF016FAEFD53F/ZCX-2026.5.27/专利/CausalDiffRec-master/experiments/reports/yelp2018_v4_latefusion_ood.json`
- `/media/p520/F2FAF016FAEFD53F/ZCX-2026.5.27/专利/CausalDiffRec-master/experiments/reports/yelp2018_v5_fair_bpr25_method_comparison.json`
- `/media/p520/F2FAF016FAEFD53F/ZCX-2026.5.27/专利/CausalDiffRec-master/experiments/reports/yelp2018_v5_fair_bpr25_validation_audit.json`
- `/media/p520/F2FAF016FAEFD53F/ZCX-2026.5.27/专利/CausalDiffRec-master/experiments/reports/yelp2018_v5_raw_vs_deepseek_ood.json`
- `/media/p520/F2FAF016FAEFD53F/ZCX-2026.5.27/专利/CausalDiffRec-master/experiments/reports/yelp2018_v5_raw_vs_deepseek_validation.json`
- `/media/p520/F2FAF016FAEFD53F/ZCX-2026.5.27/专利/CausalDiffRec-master/experiments/复杂度分析.md`

原始记录模式：

- MovieLens-1M：10 个文件，例如 `/media/p520/F2FAF016FAEFD53F/ZCX-2026.5.27/专利/CausalDiffRec-master/experiments/records/movielens1m_pure_lightgcn_v4_bpr200_ood_semantic_real_seed1024.json`
- Yelp2018：10 个文件，例如 `/media/p520/F2FAF016FAEFD53F/ZCX-2026.5.27/专利/CausalDiffRec-master/experiments/records/yelp2018_v4_lf_ood_semantic_real_seed1024_v4.json`
- Food：10 个文件，例如 `/media/p520/F2FAF016FAEFD53F/ZCX-2026.5.27/专利/CausalDiffRec-master/experiments/records/food_v4_uniform_ood_semantic_real_seed1024.json`
- Amazon Beauty：10 个文件，例如 `/media/p520/F2FAF016FAEFD53F/ZCX-2026.5.27/专利/CausalDiffRec-master/experiments/records/amazon_beauty_v4_activity_adaptive_ood_real_seed1024.json`
