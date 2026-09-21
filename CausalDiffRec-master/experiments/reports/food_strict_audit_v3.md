# Food strict-v3 数据审计

审计日期：2026-08-18

## 结论

Food `v1_strict` 可用于第三个强基线语义实验。用户数 7,809、物品数 6,309、总交互数 216,407 与项目记录的论文目标一致；划分采用逐用户时间 OOD，数据状态为 `rule-equivalent`。

## 完整性核验

- train / val / iid_test / ood_test：117,713 / 13,854 / 41,453 / 43,387。
- 四个划分内部均无重复 pair，划分之间两两交集为 0。
- `train.bin`、`eval_context_iid.bin`、`eval_context_ood.bin` 的用户到物品边集合均与 train 完全一致。
- 用户、物品映射连续，四个 parquet 的 raw ID 映射与 JSON 完全一致。
- val 中没有 train 未见用户或 train 未见物品。
- 语义先验为 `6309 × 384`，所有行有限且非零；metadata 文本覆盖率为 1.0。
- shuffled semantic 保持原向量行集合，仅有 1 行随机留在原位置。

## 基础参照与协议

- train 流行度排序在 val 上的 NDCG@20 为 0.0232543，标准用户 Hit Rate@20 为 0.108251。
- 五种子纯 LightGCN 固定训练 200 epoch。
- 1024/2048/3072 选择 alpha，4096/5120 独立确认。
- 仅当两个确认种子均提升、确认均值相对提升至少 1%、真实语义优于 shuffled semantic 时，才读取一次 OOD test。
