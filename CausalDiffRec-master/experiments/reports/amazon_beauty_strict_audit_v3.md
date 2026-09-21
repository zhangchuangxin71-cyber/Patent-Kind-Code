# Amazon Beauty strict-v3 数据审计

审计日期：2026-08-17

## 结论

`amazon_beauty/v1_strict` 可用于方法验证，但它是经过节点上限抽样的 method-validation 数据，不是原论文表 1 数据，不能表述为原论文复现。

未发现划分交互重叠、验证/测试边进入编码图、ID 映射错位或语义先验维度错位。新的 LightGCN 实验只加载 validation GT，checkpoint 选择不读取 OOD test。

## 数据核验

- 用户 10,553；物品 6,086；交互 94,148。
- train / val / iid_test / ood_test：52,723 / 7,531 / 15,065 / 18,829。
- 四个划分内部均无重复 pair，四个划分之间两两交集均为 0。
- `train.bin`、`eval_context_iid.bin`、`eval_context_ood.bin` 的用户到物品边集合均与 train 的 52,723 个 pair 完全一致。
- 用户和物品映射连续，四个 parquet 中 raw ID 到内部 ID 与映射 JSON 完全一致。
- val 有 60 个 train 未见用户、69 个 train 未见物品，分别涉及 90 和 88 条验证交互；这些节点的纯协同表示没有训练信号。
- OOD 是 popularity-uniform 构造，train 与 OOD 的物品频率 Gini 分别约为 0.587 和 0.235。

## 语义与基础参照

- 语义先验为 `6086 × 384`，metadata 记录文本覆盖率 1.0。
- item metadata 未使用 reviewText，避免把用户反馈文本写入物品描述造成直接泄漏。
- train 流行度排序在 val 上的 NDCG@20 为 0.0193544，标准用户 Hit Rate@20 为 0.0641679。该结果只作为训练失效检查，不是最终强基线。

## 实验协议

1. 五种子纯 LightGCN，固定 200 epoch，不使用早停。
2. seeds 1024/2048/3072 选择语义融合 alpha，4096/5120 独立确认。
3. 必须满足两个确认种子均提升、确认均值相对提升至少 1%、真实语义优于 shuffled semantic，才允许一次性读取 OOD test。
