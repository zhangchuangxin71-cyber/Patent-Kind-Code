# Strict OOD E0 / Stage-I / E3 及语义控制最终汇总（corrected-v3）

生成日期：2026-08-31；主实验完成日期：2026-08-20；Yelp / Food 语义控制完成日期：2026-08-30 / 2026-08-31。

## 可引用协议

- 数据：各数据集 `v1_strict`；评测：OOD。
- 种子：`1024, 2048, 3072, 4096, 5120`；每一格均为五种子均值 ± 样本标准差。
- 训练：25 epochs；checkpoint 仅按 validation `NDCG@20` 选择，之后读取一次 OOD test。
- E0 为 CausalDiffRec；Stage-I 为 `edge_gate_mode=none`、无语义；E3 为 Stage-I 加 SBERT 先验、语义融合与 `lambda_sem=0.1`。
- 本文件及 `experiments/summaries/*_strict_ood_ablation.json` 是 strict 主结果的唯一终版来源。较早说明中的 strict 数字保留为历史试验，不能与本表混用。

## 主结果

| 数据集 | 方法 | R@10 | N@10 | R@20 | N@20 |
|---|---|---:|---:|---:|---:|
| Food | E0 | 0.0278 ± 0.0006 | 0.0208 ± 0.0003 | 0.0489 ± 0.0014 | **0.0282 ± 0.0004** |
| Food | Stage-I | **0.0278 ± 0.0006** | **0.0208 ± 0.0002** | 0.0479 ± 0.0007 | 0.0278 ± 0.0003 |
| Food | E3 | 0.0276 ± 0.0007 | 0.0208 ± 0.0004 | 0.0483 ± 0.0015 | 0.0280 ± 0.0006 |
| Yelp2018 | E0 | 0.0049 ± 0.0006 | 0.0054 ± 0.0007 | 0.0094 ± 0.0011 | 0.0072 ± 0.0009 |
| Yelp2018 | Stage-I | 0.0090 ± 0.0002 | 0.0101 ± 0.0002 | 0.0173 ± 0.0006 | 0.0134 ± 0.0004 |
| Yelp2018 | E3 | **0.0094 ± 0.0007** | **0.0105 ± 0.0008** | **0.0179 ± 0.0012** | **0.0138 ± 0.0011** |
| KuaiRec | E0 | **0.0799 ± 0.0024** | **0.5279 ± 0.0090** | **0.1424 ± 0.0009** | **0.4854 ± 0.0012** |
| KuaiRec | Stage-I | 0.0385 ± 0.0103 | 0.2830 ± 0.0527 | 0.0607 ± 0.0118 | 0.2526 ± 0.0441 |
| KuaiRec | E3 | 0.0472 ± 0.0061 | 0.3172 ± 0.0428 | 0.0702 ± 0.0088 | 0.2806 ± 0.0405 |
| MovieLens-1M | E0 | 0.0157 ± 0.0001 | 0.0234 ± 0.0001 | 0.0288 ± 0.0002 | 0.0278 ± 0.0001 |
| MovieLens-1M | Stage-I | 0.0156 ± 0.0000 | 0.0234 ± 0.0001 | 0.0288 ± 0.0001 | 0.0278 ± 0.0001 |
| MovieLens-1M | E3 | **0.0158 ± 0.0003** | **0.0236 ± 0.0002** | **0.0288 ± 0.0003** | **0.0279 ± 0.0002** |

## 终版解读

- **Yelp2018**：E3 的 N@20 为 `0.01380`，高于 E0 的 `0.00716`（约 +92.9%）；其中 Stage-I 已贡献主要增益，E3 在其上再提升约 3.3%。
- **Food**：E0、Stage-I、E3 与 E4 均在误差范围内接近；E3 不构成稳定优势。
- **KuaiRec**：E3 相比 Stage-I 有所恢复，但仍显著低于 E0；不可宣称该数据集上的改进。
- **MovieLens-1M**：E3 相比 E0 的 N@20 仅约 +0.4%，应表述为基本持平。

## Yelp2018 机制控制（corrected-v3，2026-08-30）

为区分 InfoNCE、正确语义配对和额外模块容量，追加了两条预先固定的 25 epoch × 5-seed 控制。它们与上表 Yelp E3 使用相同的 strict 数据、validation 选模和一次性 OOD test；不依据本轮 OOD 结果再选参数。

| 方法 | 唯一变化 | R@20 | N@20 |
|---|---|---:|---:|
| E0 | 无 LSCI | 0.00940 ± 0.00110 | 0.00716 ± 0.00086 |
| Stage-I | 无语义 | 0.01729 ± 0.00056 | 0.01336 ± 0.00039 |
| E4 | 语义融合开，`lambda_sem=0` | 0.01713 ± 0.00065 | 0.01326 ± 0.00051 |
| E3 | 完整、正确 SBERT 语义 | **0.01787 ± 0.00125** | **0.01380 ± 0.00105** |
| E3-shuffled | 仅将语义向量的物品行随机置换 | 0.01688 ± 0.00125 | 0.01309 ± 0.00091 |

- E3 相比 E4 的 N@20 均值增量为 `+0.00054`（约 +4.1%）。
- E3 相比 shuffled-E3 的 N@20 均值增量为 `+0.00071`（约 +5.5%），但有一个种子方向相反。
- **结论：** 两个均值方向均支持完整且正确配对的语义，但五种子方差较大，尚不能将其表述为稳健、独立的语义机制证据。当前最严谨的说法是：Yelp 的最终 E3 优于 E0，而 Stage-I 贡献主要增益；语义和 InfoNCE 的额外贡献为正向但证据有限。

控制原始记录：`experiments/records/yelp2018_lsci_strict_ood_e4_lamsem0_corrected_v3_yelp_e4_lamsem0_formal_seed*_v3.json` 与 `experiments/records/yelp2018_lsci_strict_ood_e4_shuffle_corrected_v3_yelp_e3_shuffle_formal_seed*_v3.json`。

## Food InfoNCE 控制（corrected-v3，2026-08-31）

Food 使用与主表相同的 strict OOD、5-seed、25 epoch、validation 选模和一次性 OOD test 协议。E4 保留 SBERT 先验与语义融合，仅设 `lambda_sem=0`；因此 E3 与 E4 的唯一设计差异是 InfoNCE 对齐损失。

| 方法 | R@20 | N@20 |
|---|---:|---:|
| E0 | **0.04894 ± 0.00142** | **0.02816 ± 0.00044** |
| Stage-I | 0.04788 ± 0.00073 | 0.02782 ± 0.00029 |
| E4（融合开，`lambda_sem=0`） | 0.04819 ± 0.00129 | 0.02790 ± 0.00053 |
| E3（完整 SBERT + InfoNCE） | 0.04829 ± 0.00154 | 0.02802 ± 0.00064 |

- E3 相比 E4 的 N@20 均值仅 `+0.00012`（约 +0.4%），明显小于两者的跨种子标准差。
- **结论：** Food 上没有可辨识的 InfoNCE 稳定增益；该控制与主表一致，支持将 Food 表述为各方法基本持平，而非语义改进案例。

控制原始记录：`experiments/records/food_lsci_strict_ood_e4_lamsem0_corrected_v3_food_e4_lamsem0_formal_seed*_v3.json`。

## 生成与可复现

汇总程序为 `scripts/aggregate_strict_ablation.py`（E3 / E4 根据显式 `method` 独立归类，避免与 Stage-I 的 `edge_gate_mode=none` 混合）。重新生成：

```bash
for ds in food yelp2018 kuairec movielens1m; do
  python3 scripts/aggregate_strict_ablation.py --dataset "$ds"
done
```

E3 原始记录命名为 `experiments/records/<dataset>_lsci_strict_ood_e3_strict_e3_semantic_seed*_strict_ood_e3_seed*_v3.json`。
