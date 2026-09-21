# Strict OOD 消融实验最终报告

协议：四个 strict 数据集，OOD 测试，25 epochs，5 seeds（1024、2048、3072、4096、5120）。模型选择只使用 validation，表中指标为最佳 validation epoch 对应的 one-shot test 结果。

## NDCG@20（均值 ± 标准差）

| 数据集 | E0 | Stage-I none | Random gate | Soft gate |
|---|---:|---:|---:|---:|
| Food | 0.02816 ± 0.00044 | 0.02782 ± 0.00029 | 0.02792 ± 0.00038 | 0.02786 ± 0.00020 |
| Yelp2018 | 0.00716 ± 0.00086 | 0.01336 ± 0.00039 | 0.01353 ± 0.00030 | 0.01360 ± 0.00054 |
| KuaiRec | 0.48540 ± 0.00121 | 0.25265 ± 0.04406 | 0.24307 ± 0.02703 | 0.25615 ± 0.02289 |
| MovieLens-1M | 0.02783 ± 0.00010 | 0.02784 ± 0.00006 | 0.02789 ± 0.00017 | 0.02797 ± 0.00021 |

## 结论

- Stage-I 相对 E0 的收益具有数据集依赖性：Yelp2018 明显提升，Food 和 MovieLens 基本持平，KuaiRec 的 test 口径显示 E0 更高且波动更小。
- soft gate 与 random gate 的差距很小，当前证据不足以证明评分器学到了稳定的因果信号。
- 生产默认保持 `--edge_gate_mode none`；soft/random 仅作为消融配置。
- Douban 因 strict 数据源仍标记为 BLOCKED，未纳入本报告。

原始记录：`experiments/records/`；机器汇总：`experiments/summaries/*_strict_ood_ablation.json`。
