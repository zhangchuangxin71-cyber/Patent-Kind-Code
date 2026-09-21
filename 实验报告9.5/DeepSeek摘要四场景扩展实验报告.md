# DeepSeek 摘要四场景扩展实验报告

生成日期：2026-09-08

## 实验协议

在 `v1_strict` 数据划分下，对 LightGCN、无语义 CausalDiffRec 和完整语义增强方法进行比较。每种方法使用五个随机种子（1024、2048、3072、4096、5120），训练预算为 25 次完整 BPR 遍历；模型按 validation NDCG@20 选取，随后进行一次冻结 OOD 评估。语义先验由 DeepSeek 摘要后使用同一 SBERT（`all-MiniLM-L6-v2`）编码得到。

## OOD 结果

指标为均值 ± 样本标准差。

| 数据集 | 方法 | Recall@20 | NDCG@20 |
|---|---|---:|---:|
| Yelp2018 | LightGCN | 0.021620 ± 0.000343 | 0.016900 ± 0.000244 |
|  | CausalDiffRec | 0.016542 ± 0.000991 | 0.012828 ± 0.000742 |
|  | DeepSeek 完整组合（α=0.5） | 0.018690 ± 0.000342 | 0.014654 ± 0.000244 |
| Food | LightGCN | 0.047580 ± 0.000083 | 0.027838 ± 0.000144 |
|  | CausalDiffRec | 0.049176 ± 0.000867 | 0.028374 ± 0.000305 |
|  | DeepSeek 完整组合（α=1.0） | 0.049314 ± 0.000441 | 0.028546 ± 0.000126 |
| Amazon Beauty | LightGCN | 0.028586 ± 0.000558 | 0.012526 ± 0.000788 |
|  | CausalDiffRec | 0.011702 ± 0.002240 | 0.004962 ± 0.001123 |
|  | DeepSeek 完整组合（α=2.0） | **0.063830 ± 0.000185** | **0.030380 ± 0.000240** |
| MovieLens-1M | LightGCN | 0.028842 ± 0.000174 | 0.027902 ± 0.000020 |
|  | CausalDiffRec | 0.028596 ± 0.000163 | 0.027822 ± 0.000072 |
|  | DeepSeek 完整组合（α=0.5） | **0.032636 ± 0.000303** | **0.032026 ± 0.000326** |

## 验证集选参

各数据集 α 只使用 validation 选择，未使用 OOD 标签：Yelp2018=0.5，Food=1.0，Amazon Beauty=2.0，MovieLens-1M=0.5。四个数据集的 confirmation seeds 均满足语义融合优于 α=0 基线。

## 结论

语义增强方法在四个场景均明显优于无语义 CausalDiffRec；在 Amazon Beauty 和 MovieLens-1M 上超过 LightGCN，在 Food 上略高于 LightGCN，在 Yelp2018 上仍低于 LightGCN。该结果支持“语义先验对 OOD、长尾和冷启动具有补偿作用”，但不支持“语义增强在所有数据集上普遍超过 LightGCN”。

## 机器可读结果

- Amazon Beauty：`CausalDiffRec-master/experiments/reports/amazon_beauty_v5_deepseek_alpha2_ood.json`
- Yelp2018/Food：`CausalDiffRec-master/experiments/reports/yelp_food_deepseek_alpha_ood_comparison.json`
- MovieLens-1M：`CausalDiffRec-master/experiments/reports/movielens1m_deepseek_alpha_validation.json`
