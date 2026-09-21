# KuaiRec strict-v3 数据与评估审计

审计日期：2026-08-17

## 结论

当前 `v1_strict` 数据可用于新的验证集实验。没有发现交互划分重叠、验证/测试边进入编码图、ID 映射错位、越界或语义先验行错位。新的强基线必须只用验证集选择 checkpoint；本轮训练已设置 `load_test_gt=False`。

KuaiRec 的验证集来自 big matrix 的全局随机 10% 划分，而 OOD 测试集来自 small matrix。这能复现“曝光环境变化”的既定材料，但没有独立的 OOD validation，因此超参数选择只能依据 IID validation。该限制应在论文中披露。

## 数据与图核验

- 用户数：7,176；物品数：10,612；总节点数：17,788。
- train / val / iid_test / ood_test 交互数分别为 592,489 / 84,641 / 169,284 / 217,229。
- 四个划分的 `(user, item)` 两两交集均为 0，且每个划分内部无重复 pair。
- `train.bin` 有 1,184,978 条双向边，恰好对应 592,489 条 train 交互。
- `eval_context_iid.bin` 和 `eval_context_ood.bin` 的用户到物品边集合均与 train 完全相同：无缺失、无额外边。
- 用户、物品映射均从 0 连续编号；四个 parquet 中 raw ID 到内部 ID 的映射与 JSON 映射逐行一致。
- val 的 6,944 个用户全部在 train 中出现；val 有 231 个物品未在 train 中出现，共涉及 240 条交互。
- OOD 中有 1 个冷启动用户和 8 个冷启动物品。全集比 manifest 中记录的论文目标统计各多 1 个用户和 1 个物品，来源是映射时纳入了仅在 OOD 中出现的节点。

## 语义先验核验

- `semantic_prior.pt` 的物品矩阵为 `10612 × 384`，10,612 行均为有限、非零向量，metadata 覆盖率为 1.0。
- 物品 metadata 共 10,612 行，ID 唯一；title、description、categories 均为非空。
- shuffled prior 形状一致且保持原向量行的多重集合；仅有 1 行偶然留在原位置，符合随机排列对照的预期。

以上核验只能证明文件对齐和数值完整，不能证明文本的语义质量。真实语义必须通过 real prior、shuffled prior 与协同基线的验证对照判断。

## 指标口径解释

项目中的 `Hit Ratio` 不是常见的“至少命中一个物品的用户比例”，而是：

`所有用户命中的正样本总数 / 所有用户的正样本总数`

它实际是微平均 recall。代码中的 `Recall` 才是逐用户 recall 的宏平均。旧 OOD 集每用户正样本数的中位数为 95、均值约 153.95，因此旧结果 `Precision@20 ≈ 0.405` 表示每用户平均命中约 8.1 个，而 `Hit Ratio@20 ≈ 0.0526` 等于约 `8.1 / 153.95`；两者并不矛盾。

为避免误读，后续论文应把现有 `Hit Ratio` 标成 `Micro Recall`，如需标准 HR，再额外报告“至少命中一个的用户比例”。历史表格可保留原字段，但必须注明定义。

## 基础 sanity reference

只按 train 物品流行度排序，并为每个用户屏蔽 train 已见物品，在 val 上得到：

- NDCG@20：0.0575775
- Precision@20：0.0281898
- 宏平均 Recall@20：0.0691305
- 标准用户 Hit Rate@20：0.3777362

该结果仅作为训练是否失效的 sanity reference，不是最终论文基线。

## 当前实验决策

1. seed 1024 的纯 LightGCN 固定训练 200 epoch，不启用早停，避免早期短暂下降导致错误终止。
2. 单种子完成后，只有在协议字段正确、200 epoch 完整结束、NDCG 有限且超过流行度 sanity reference 时，才自动扩展到其余四个种子。
3. 五种子完成后，用 1024/2048/3072 选择 late-fusion alpha，用 4096/5120 独立确认。
4. 只有当两个确认种子都提升、确认均值相对提升至少 1%、且真实语义优于 shuffled semantic 时，才允许一次性读取 OOD test；否则保持 test 未打开。

## 尚存限制

- big matrix 的 train/val/iid_test 是按全局交互随机切分，而非逐用户切分；少量 val 物品在 train 中完全不可见。
- 当前数据验证脚本只用边数近似检查图合同；本次人工审计做了精确边集合检查，但建议以后把精确检查固化到 validator。
- 当前指标名称容易误导，应补充标准 HR，同时明确 micro/macro averaging。
- 文本覆盖率 100% 不等于文本有判别力；必须依赖真实/打乱语义对照得出结论。
