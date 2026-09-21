# 专利撰写 AI 实验交接稿（唯一推荐导出文件）

生成日期：2026-09-06

用途：将本文件完整交给另一套 AI，用于撰写专利中的“技术方案、实施例、实验设置、对照实验、实验结果及有益效果”。本文件已经区分终版证据、辅助证据、失败实验和过期开发结果。

> 给专利撰写 AI 的强制说明：实验事实以本文件及其列出的终版来源为准。不要从 `experiments/LSCI_全部实验汇总说明.md` 或 `experiments/LSCI_实验对照表.md` 抄取早期 strict 数字；这些文件含有开发阶段结果，与 corrected-v3/v4/v5 终版存在差异。

## 1. 可用于专利的核心结论

本项目当前最稳妥的专利实验结论是：

> 在协同推荐模型输出的基础上，引入由物品文本构建的语义先验，根据用户训练期交互历史形成用户语义画像，并对协同分数与语义匹配分数进行标准化融合，可以提升多数测试数据集上的分布外推荐效果。真实物品—语义对应关系总体优于保持向量分布不变但打乱物品对应关系的语义对照，说明改进并非仅来自额外分数通道或随机扰动。

实验涉及五个数据集：MovieLens-1M、Yelp2018、Food、Amazon Beauty 和 KuaiRec。

- 相对协同 baseline，MovieLens-1M、Yelp2018、Amazon Beauty 的 OOD NDCG@20 提升，即 5 个数据集中的 3 个有效，符合“多数数据集有效”的表述。
- Food 未超过 baseline，但真实语义显著优于打乱语义，可作为语义机制证据。
- KuaiRec 未通过 validation 门禁，属于失败与适用边界，不作为正向实施效果。
- corrected-v5 统一 BPR25 比较中，MovieLens-1M 完整方法的 OOD NDCG@20 为 0.035068，相对原始 CausalDiffRec 提升 26.04%，相对 LightGCN 提升 25.68%，均为 5/5 种子胜出。
- 同一 BPR25 协议在 Yelp2018 与 Amazon Beauty 上显示：完整方法相对 CausalDiffRec 分别提升 19.91% 与 45.33%（均 5/5），但分别低于 LightGCN 8.98% 与 42.41%；Food 上相对 LightGCN 提升 2.94%（5/5），但相对 CausalDiffRec 的 +0.99% 不稳定（5 个 seed 配对符号检验 `p=0.18750`）。
- corrected-v5 文本来源对照中，Yelp2018 的 DeepSeek+SBERT 相对 raw+SBERT 提升 6.56%（共享 alpha）或 7.35%（两路各自同等选参），两种口径都是 5/5 种子胜出；Food 为负向，KuaiRec 未通过门禁。
- 因此可以写“在多数数据集上提升”，不能写“所有数据集都提升”或“完整 LSCI 在所有数据集上显著优于原方法”。

建议把专利技术重点放在一条完整技术链：由大模型规范化文本得到物品语义先验；通过 `ConditionFusion` 将该先验注入扩散训练、以 InfoNCE 对齐；再用训练期交互构建用户语义画像并标准化融合语义/协同分数，权重在实验协议中冻结。三方法表属于组合实施例，只能支持整体效果，不能把增益拆分归因于某一模块；也不要把“完整方法必须在所有数据集优于 CausalDiffRec/LightGCN”或未获稳定支持的 learned causal edge gate 作为唯一核心增益来源。

## 2. 技术方案实际是怎样实现的

### 2.1 数据输入与无泄漏约束

标准 strict 数据协议将交互划分为 train、validation、IID test 和 OOD test；KuaiRec v2 exposure 是例外，只保留 train、validation 与 OOD test。所有协议下，训练模型和构造用户语义画像时都只使用 train 交互：

- train：学习协同模型参数，并构造每个用户的语义画像；
- validation：选择 checkpoint、统一融合系数 `alpha` 或分组融合系数；
- IID test：标准 strict 协议中保留的同分布评测划分，不作为本文正面 OOD 主表来源；
- OOD test：只在 validation 选参和 confirmation 均通过后读取，作为最终主结果。

评测推荐列表前，用户在 train 中已经交互过的物品会被屏蔽，防止把历史物品重新推荐并计入指标。

### 2.2 物品语义先验

对每个物品，从 `metadata/items.jsonl` 读取标题、类别和描述等文本字段，拼成物品文本。语义先验的构建流程为：

1. 对配置为 `llm` 的数据集，先用 DeepSeek 将物品文本改写成保留类别、属性和意图的短英文摘要；结果写入缓存，训练和评测时不再调用外部 API。
2. 使用 `sentence-transformers/all-MiniLM-L6-v2` 对文本编码。
3. 对编码向量进行归一化，得到每个物品的 384 维向量，并保存为 `features/semantic_prior.pt`。
4. 无文本物品用语义 mask 标记，防止空文本被当作有效语义。

终版记录中的语义先验信息如下：

| 数据集 | 文本处理方式 | 编码器 | 物品数 | 维度 | 覆盖率 |
|---|---|---|---:|---:|---:|
| MovieLens-1M | raw metadata | SBERT all-MiniLM-L6-v2 | 2380 | 384 | 100% |
| Yelp2018 | DeepSeek 改写后编码 | SBERT all-MiniLM-L6-v2 | 12185 | 384 | 100% |
| Food | DeepSeek 改写后编码 | SBERT all-MiniLM-L6-v2 | 6309 | 384 | 100% |
| Amazon Beauty | raw metadata | SBERT all-MiniLM-L6-v2 | 6086 | 384 | 100% |
| KuaiRec v2 | 复用 v1 的 DeepSeek 改写缓存与 SBERT 先验 | SBERT all-MiniLM-L6-v2 | 10612 | 384 | 100% |

### 2.3 用户语义画像

设物品 $i$ 的归一化语义向量为 $e_i$，用户 $u$ 在 train 中交互的物品集合为 $I_u^{train}$。用户语义画像由训练历史物品语义的均值构成，再进行 L2 归一化：

\[
p_u = \operatorname{Normalize}\left(\frac{1}{|I_u^{train}|}\sum_{i\in I_u^{train}} e_i\right).
\]

用户与候选物品之间的语义分数为：

\[
s_{sem}(u,i)=p_u^\top e_i.
\]

这里不使用 validation 或 test 中的用户—物品关系构造画像，所以语义路径不会泄漏测试答案。

### 2.4 标准化语义分数融合

协同模型为每个用户产生候选物品分数 $s_{cf}(u,i)$。由于协同分数与语义分数尺度不同，代码先按用户分别做 z-score 标准化，再融合：

\[
s(u,i)=z_u(s_{cf}(u,i))+\alpha\,z_u(s_{sem}(u,i)).
\]

其中 $z_u(\cdot)$ 表示在同一用户的全部候选物品上减均值、除以标准差；`alpha` 控制语义分数权重。

- `alpha=0`：collaborative baseline；
- `alpha>0` 且使用真实 prior：real semantic；
- `alpha>0` 且使用打乱 prior：shuffled semantic。

主实验复用同一 seed 的同一冻结 checkpoint，只改变最终计分时使用的语义 prior 或 `alpha`，因此 real、shuffled 与 baseline 之间是配对比较，不存在重新训练造成的混杂。

### 2.5 打乱语义对照

`semantic_prior_shuffled.pt` 保留原语义向量集合、向量维度和数值分布，但随机置换物品行，使物品 ID 与语义向量错误对应。该对照只改变“语义是否与正确物品匹配”，其他计算流程与 real semantic 完全相同。

如果真实语义优于 shuffled，说明效果来自正确语义关系，而不仅是增加一个分数通道、向量范数或随机重排。

### 2.6 用户活跃度自适应融合

根据用户在 train 中的交互次数（degree）划分低、中、高活跃用户。在 validation selection seeds 上分别为三个组选择 `alpha`，然后冻结分位数阈值和三个权重。

其计分公式为：

\[
s(u,i)=z_u(s_{cf}(u,i))+\alpha_{g(u)}z_u(s_{sem}(u,i)),
\]

其中 $g(u)$ 是用户活跃度组。分组只由 train degree 决定。

### 2.7 指标与统计口径

- Recall@20：前 20 个推荐中覆盖真实相关物品的比例，越高越好。
- NDCG@20：同时考虑前 20 个推荐是否命中以及命中位置，相关物品排得越靠前得分越高，越高越好。
- 正式结果使用 seeds `1024, 2048, 3072, 4096, 5120`，汇报五次结果的均值与样本标准差。
- real、baseline、shuffled 复用同一 seed 的 checkpoint，因此统计时计算同 seed 配对差值，而不是把两组独立处理。
- bootstrap 95% CI 对五个 seed 的配对差值重采样；单边 sign test 检验正向 seed 是否多于随机方向。五个差值全部为正时，单边 sign test 的最小 p 值是 0.03125。

## 3. 所有主要实验分别怎样做、改变了什么

| 编号 | 实验 | 数据集 | 保持不变 | 唯一或主要改变 | 目的 | 专利定位 |
|---|---|---|---|---|---|---|
| P0 | strict 数据协议与泄漏审计 | Food、Yelp、KuaiRec、MovieLens、Amazon | 原始交互与元数据来源 | 将数据按严格协议拆成 train/val/IID/OOD，并检查交互重叠 | 保证结果不是测试泄漏 | 方法可信度说明 |
| P1 | legacy E0 复现 | Food、Yelp、KuaiRec、Douban | 原仓库旧数据和原模型 | 不加入新模块，仅更换随机种子重复 5 次 | 验证旧工程可运行 | 背景，不进终版主表 |
| P2 | corrected-v3 E0 / Stage-I / gate 消融 | Food、Yelp、KuaiRec、MovieLens | strict 数据、25 epochs、5 seeds、validation 选模 | 在 CausalDiffRec E0、LSCI `none`、随机环境 gate、learned soft gate 间切换 | 判断图环境/gate 是否稳定有效 | 辅助与失败边界 |
| P3 | corrected-v3 E3 完整语义训练 | Food、Yelp、KuaiRec、MovieLens | 与 P2 相同的 strict/OOD/5-seed 协议 | 在 Stage-I backbone 上加入 SBERT prior、语义融合和 `lambda_sem=0.1` InfoNCE | 检查端到端语义训练效果 | 背景；仅 Yelp 明显正向 |
| P4 | E4：关闭 InfoNCE | Yelp、Food | 与 E3 相同的模型、先验、融合与训练协议 | 只将 `lambda_sem` 从 0.1 改为 0 | 分离 InfoNCE 的独立贡献 | 机制辅助证据 |
| P5 | E3-shuffled | Yelp，早期另有 Food/Kuai 单 seed 诊断 | E3 训练设置及向量集合不变 | 只打乱物品 ID 与语义向量的对应 | 验证是否真正利用正确语义 | v3 辅助；v4 已升级为 5-seed 正式控制 |
| P6 | 强制语义门控诊断 | Food、Yelp、KuaiRec，单 seed | 数据、训练轮数和主体模型不变 | 强制 gate=0.3，并把 `lambda_sem` 提到 1.0 | 判断失败是否因为模型“没有使用足够语义” | 仅诊断，不作为专利正面统计 |
| P7 | corrected-v4 统一 late fusion | MovieLens、Yelp、Food | 每个 seed 的冻结协同 checkpoint、候选集、评测函数均不变 | 在 `alpha=0`、真实语义、打乱语义之间切换 | 直接验证语义分数融合 | 专利主实验 |
| P8 | activity-adaptive late fusion | MovieLens、Amazon Beauty | 冻结 checkpoint、语义画像和融合公式不变 | 把统一 `alpha` 改为由 train degree 决定的三组 `alpha` | 检查不同活跃度用户是否需要不同语义权重 | MovieLens 支持；Amazon 仅作可用实现例 |
| P9 | 相同 BPR 预算控制 | MovieLens | 五个 seeds、200 次完整 BPR pass、OOD 门禁 | 对纯 LightGCN 与 LSCI-stage 初始化/刷新方式进行公平训练预算比较 | 排除训练步数不一致造成的假增益 | 决定终版选用纯 LightGCN 主结果 |
| P10 | BPR budget curve | MovieLens、Yelp | 数据、模型结构和 seeds | 只改变完整 BPR passes 数量 | 检查 backbone 是否训练不足 | 训练诊断，不作为方法增益 |
| P11 | KuaiRec v2 exposure late fusion | KuaiRec | 冻结 LightGCN、5 seeds、validation-only 选参 | 扫描 `alpha=0...2`，比较 real/shuffle/baseline | 检查曝光协议下语义融合是否有效 | 失败/适用边界 |
| P12 | KuaiRec popularity rerank | KuaiRec | 同一冻结推荐分数 | 从物品分数中减去 `gamma*log(1+train popularity)`，validation 选 gamma | 判断失败是否能由简单流行度惩罚修复 | 失败诊断 |
| P13 | KuaiRec 分组命中诊断 | KuaiRec | 同一 checkpoint 与 alpha | 仅按用户活跃度和物品流行度分组统计指标 | 定位尾部和冷物品问题 | 失败原因解释 |
| P14 | 复杂度分析 | Food、Yelp、KuaiRec、Douban 等 | 算法实现 | 统计理论时间/空间复杂度、墙钟和显存 | 说明工程可实施性 | 可写入实施例，精确新方法耗时仍建议补测 |
| P15 | 原始文本 vs DeepSeek 优化文本 | Food、Yelp、KuaiRec | 同一数据划分、checkpoint、SBERT、向量维度、候选集、评价和 alpha 网格 | 只改变 SBERT 的输入是原始文本还是 DeepSeek 规范化摘要 | 隔离大语言模型优化文本的独立贡献 | Yelp 正向实施例；Food/Kuai 为边界 |
| P16 | LightGCN / CausalDiffRec / 完整方法统一 BPR25 比较 | MovieLens、Yelp、Food、Amazon Beauty | 同一 strict 划分、候选集、指标、选模规则、5 seeds、25 次 BPR pass 和排序超参 | 切换协同基线、原始因果扩散、含语义条件/InfoNCE/分数融合的完整方法 | 直接检验本次五 seed 的三方法配对方向 | MovieLens 两对照均正向；Yelp/Amazon 仅相对 CausalDiffRec 正向；Food 仅相对 LightGCN 5/5 正向 |

## 4. corrected-v3 端到端模型消融

### 4.1 方法定义

- E0：原始 CausalDiffRec。
- Stage-I / `none`：LSCI 训练骨架，`edge_gate_mode=none`，不加入物品语义。这里不要沿用旧文档中的“只切图”简称；在 corrected-v3 终版里，以 `edge_gate_mode=none` 为准。
- Random gate：使用 pair-symmetric 随机环境。
- Soft gate：先随机 warm-up，再使用学习得到的 pair-symmetric 边权。
- E3：Stage-I 加正确 SBERT 语义先验、语义融合与 `lambda_sem=0.1` 的 InfoNCE 对齐损失。
- E4：保留 E3 的 SBERT 先验和语义融合，只令 `lambda_sem=0`。
- E3-shuffled：保留 E3 设置，只随机置换物品语义向量行。

### 4.2 gate 消融结果

下表为 strict OOD NDCG@20，5 seeds，25 epochs，checkpoint 仅按 validation NDCG@20 选择：

| 数据集 | E0 | Stage-I none | Random gate | Soft gate | 解读 |
|---|---:|---:|---:|---:|---|
| Food | 0.02816±0.00044 | 0.02782±0.00029 | 0.02792±0.00038 | 0.02786±0.00020 | 四者接近 |
| Yelp2018 | 0.00716±0.00086 | 0.01336±0.00039 | 0.01353±0.00030 | 0.01360±0.00054 | LSCI 骨架有效，但 soft 与 random 很接近 |
| KuaiRec | 0.48540±0.00121 | 0.25265±0.04406 | 0.24307±0.02703 | 0.25615±0.02289 | 各 LSCI arm 均低于 E0 |
| MovieLens-1M | 0.02783±0.00010 | 0.02784±0.00006 | 0.02789±0.00017 | 0.02797±0.00021 | 各方法近似持平 |

该实验改变的是 edge/environment arm。结果不足以证明 learned soft gate 比随机 gate 稳定更好，因此专利不能把“learned causal gate 已在多数据集验证有效”写成实验结论。

### 4.3 E0 / Stage-I / E3 结果

| 数据集 | 方法 | R@20 | NDCG@20 | 该实验改变了什么 | 结论 |
|---|---|---:|---:|---|---|
| Food | E0 | 0.04894±0.00142 | 0.02816±0.00044 | 原始方法 | 对照 |
| Food | Stage-I | 0.04788±0.00073 | 0.02782±0.00029 | 换为 LSCI none 骨架，无语义 | 与 E0 接近 |
| Food | E3 | 0.04829±0.00154 | 0.02802±0.00064 | Stage-I + SBERT + 融合 + InfoNCE | 未稳定超过 E0 |
| Yelp2018 | E0 | 0.00940±0.00110 | 0.00716±0.00086 | 原始方法 | 对照 |
| Yelp2018 | Stage-I | 0.01729±0.00056 | 0.01336±0.00039 | LSCI none 骨架，无语义 | 贡献主要增益 |
| Yelp2018 | E3 | 0.01787±0.00125 | 0.01380±0.00105 | 加 SBERT、融合、InfoNCE | 比 E0 高，较 Stage-I 约 +3.3% |
| KuaiRec | E0 | 0.14242±0.00093 | 0.48540±0.00121 | 原始方法 | 最优 |
| KuaiRec | Stage-I | 0.06073±0.01177 | 0.25265±0.04406 | LSCI none 骨架 | 明显下降 |
| KuaiRec | E3 | 0.07025±0.00880 | 0.28059±0.04049 | 加 SBERT、融合、InfoNCE | 比 Stage-I 恢复，但仍低于 E0 |
| MovieLens-1M | E0 | 0.02878±0.00021 | 0.02783±0.00010 | 原始方法 | 对照 |
| MovieLens-1M | Stage-I | 0.02876±0.00008 | 0.02784±0.00006 | LSCI none 骨架 | 基本持平 |
| MovieLens-1M | E3 | 0.02885±0.00026 | 0.02794±0.00022 | 加 SBERT、融合、InfoNCE | 约 +0.4%，基本持平 |

### 4.4 InfoNCE 和正确语义配对控制

Yelp corrected-v3：

| 方法 | 相对 E3 改变 | R@20 | NDCG@20 |
|---|---|---:|---:|
| Stage-I | 去掉全部语义部分 | 0.01729±0.00056 | 0.01336±0.00039 |
| E4 | 保留先验与融合，只将 `lambda_sem=0.1` 改为 0 | 0.01713±0.00065 | 0.01326±0.00051 |
| E3 | 正确 SBERT + `lambda_sem=0.1` | 0.01787±0.00125 | 0.01380±0.00105 |
| E3-shuffled | 只打乱物品—语义行对应 | 0.01688±0.00125 | 0.01309±0.00091 |

E3 比 E4 的 NDCG@20 高 0.00054，E3 比 shuffled 高 0.00071，但 shuffled 比较中有一个 seed 方向相反。该结果支持语义和 InfoNCE 的正向趋势，但不能单独宣称稳定显著。

Food corrected-v3：E3 NDCG@20 为 0.02802±0.00064，E4 为 0.02790±0.00053，只差约 +0.4%，小于跨 seed 波动，不能认为 Food 上 InfoNCE 有稳定独立增益。

## 5. corrected-v4 专利主实验

### 5.1 统一实验步骤

四个正式数据集上的 v4 主实验均按以下顺序执行：

1. 对每个 seed 训练或载入协同推荐 checkpoint。
2. 用 train 交互构造用户语义画像；载入 real 和 shuffled 两套 prior。
3. 在 selection seeds `1024, 2048, 3072` 的 validation NDCG@20 上选 `alpha` 或分组 profile。
4. 在 confirmation seeds `4096, 5120` 上验证：真实语义需超过 baseline 和 shuffled，并满足预设门禁。
5. 选参期间设置 `load_test_gt=False`，不读取 OOD 标签。
6. 门禁通过后，固定所有参数，对 5 个 seeds 各读取一次 OOD test。
7. 对同一 seed 的 real-baseline 与 real-shuffled 做配对差值，汇报均值、样本标准差、bootstrap 95% CI 和单边 sign test。

### 5.2 每个数据集具体改变

| 数据集 | 冻结 backbone | 对照设置 | 选中的语义设置 | 相比 baseline 实际改变 |
|---|---|---|---|---|
| MovieLens-1M | 3 层 pure LightGCN，统一 200 次 BPR pass | 同 checkpoint，`alpha=0` | uniform `alpha=0.1` | 只在最终分数中加入真实语义匹配分数 |
| Yelp2018 | corrected-v4 LSCI Stage-I checkpoint | 同 checkpoint，`alpha=0` | uniform `alpha=0.25` | 只加入真实语义 late fusion |
| Food | pure LightGCN checkpoint | 同 checkpoint，`alpha=0` | uniform `alpha=0.25` | 只加入真实语义 late fusion |
| Amazon Beauty | LightGCN checkpoint | 同 checkpoint，`alpha=0` | low=1.0、mid=1.0、high=0.5 | 依据 train degree 为用户选择已冻结的语义权重 |

### 5.3 OOD 终版结果

| 数据集 | R@20 baseline | R@20 real | R@20 shuffled | N@20 baseline | N@20 real | N@20 shuffled | real vs baseline | real vs shuffled |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MovieLens-1M | 0.06928±0.00218 | 0.07165±0.00261 | 0.06720±0.00239 | 0.06831±0.00191 | 0.07037±0.00201 | 0.06700±0.00185 | +3.02% | +5.04% |
| Yelp2018 | 0.01731±0.00146 | 0.01862±0.00106 | 0.01694±0.00136 | 0.01345±0.00115 | 0.01452±0.00080 | 0.01314±0.00115 | +8.00% | +10.52% |
| Food | 0.04690±0.00065 | 0.04679±0.00036 | 0.04431±0.00080 | 0.02780±0.00017 | 0.02768±0.00014 | 0.02661±0.00026 | -0.42% | +4.03% |
| Amazon Beauty | 0.03847±0.00117 | 0.07576±0.00159 | 0.03189±0.00104 | 0.01824±0.00109 | 0.03834±0.00117 | 0.01466±0.00051 | +110.24% | +161.56% |

注意：MovieLens 表中的 R@20 real/shuffled 对应 uniform `alpha=0.1`；Amazon 表中的 real/shuffled 对应 activity-adaptive profile。

### 5.4 配对统计

| 数据集 | real-baseline N@20 均值差 | bootstrap 95% CI | 单边 sign p | 正向 seeds | real-shuffled N@20 均值差 | bootstrap 95% CI | 单边 sign p | 正向 seeds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MovieLens-1M | +0.00206 | [0.00187, 0.00231] | 0.03125 | 5/5 | +0.00338 | [0.00316, 0.00362] | 0.03125 | 5/5 |
| Yelp2018 | +0.00108 | [0.00074, 0.00142] | 0.03125 | 5/5 | +0.00138 | [0.00103, 0.00173] | 0.03125 | 5/5 |
| Food | -0.00012 | [-0.00032, 0.00005] | 0.81250 | 2/5 | +0.00107 | [0.00094, 0.00122] | 0.03125 | 5/5 |
| Amazon Beauty | +0.02010 | [0.01933, 0.02110] | 0.03125 | 5/5 | +0.02368 | [0.02274, 0.02462] | 0.03125 | 5/5 |

只有 5 个配对 seeds 时，单边 sign test 的最小可能 p 值就是 0.03125。MovieLens、Yelp、Amazon 对 baseline 和 shuffled 都是 5/5 正向；Food 对 baseline 不成立，但对 shuffled 是 5/5 正向。

## 6. corrected-v5 新增关键对照

### 6.1 原始文本 vs DeepSeek 优化文本

本实验的 A 组为“原始物品文本 → 同一 SBERT → raw prior”，B 组为“同一原始文本 → DeepSeek 规范化摘要 → 同一 SBERT → LLM prior”。两组保持 strict 数据划分、每 seed 的冻结 checkpoint、全候选集、历史物品屏蔽、Top-20 评价、`all-MiniLM-L6-v2`、384 维、L2 归一化、100% 覆盖率、用户语义画像和 z-score 融合完全相同；唯一有意改变是 SBERT 的输入文本来源。

alpha 网格为 `0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2`；`1024/2048/3072` 用于 selection，`4096/5120` 用于 confirmation，此阶段 `load_test_gt=False`。主对比共享 LLM 在 validation 选定的 alpha；辅助对比给两路完全相同的网格并各自选最优 alpha。

| 数据集 | baseline N@20 | raw 共享 alpha | raw 自选 alpha | DeepSeek | DeepSeek-raw 共享 | DeepSeek-raw 各自选参 |
|---|---:|---:|---:|---:|---:|---:|
| Yelp2018 | 0.013448 | 0.013630 | 0.013530 | **0.014524** | **+6.56%** | **+7.35%** |
| Food | 0.027800 | **0.027946** | 0.027800 | 0.027684 | -0.94% | -0.42% |

Yelp 共享 alpha 对比为 5/5 种子正向，配对差 95% t 区间 `[0.000403, 0.001385]`，单侧精确符号检验 `p=0.03125`；各自选参时也为 5/5 正向。Food 在 OOD 上方向反转；KuaiRec 的两路最优 alpha 均为 0，按门禁不读取该项 OOD 标签。

### 6.2 LightGCN、原始 CausalDiffRec 与完整方法

三个对照的准确定义为：

- LightGCN：只使用协同交互图与 BPR 损失；
- 原始 CausalDiffRec（E0）：VGAE + 扩散生成 + 图编辑/环境建模 + LightGCN 排序头，无语义；
- 完整方法：因果扩散/LSCI backbone + `ConditionFusion` + `lambda_sem=0.1` InfoNCE + train-only 用户语义画像分数融合（`alpha=0.75`）。未开启缺少稳定证据的 learned edge gate，即 `edge_gate_mode=none`。

三组在 MovieLens-1M、Yelp2018、Food、Amazon Beauty 上共享同一 corrected-v5 BPR25 协议：对应数据集的 strict 划分、全物品 OOD 候选、train 物品屏蔽、validation NDCG@20 选模、5 seeds、25 次完整 BPR pass、8 维嵌入、3 层 LightGCN、batch 1024、lr 0.001 与 L2 0.001。每个数据集的 16 项直接公平性/无泄漏审计均为 true 后，才对五个种子做一次性冻结 OOD 评测。Yelp2018、Food、Amazon Beauty 的 `alpha=0.75` 为固定组合参数，未提供外部语义权重 confirmation 文件；不得写成已完成独立 validation confirmation。

| 方法 | Recall@20 | NDCG@20 |
|---|---:|---:|
| LightGCN | 0.028842±0.000174 | 0.027902±0.000020 |
| 原始 CausalDiffRec | 0.028596±0.000163 | 0.027822±0.000072 |
| 完整方法 | **0.037086±0.000277** | **0.035068±0.000094** |

MovieLens 上，完整方法相对 CausalDiffRec 的 NDCG@20 为 `+0.007246/+26.04%`，相对 LightGCN 为 `+0.007166/+25.68%`；两组均是 5/5 种子正向，单侧精确符号检验 `p=0.03125`。

同协议扩展结果如下（均为五个 OOD seed 的均值±样本标准差）：

| 数据集 | LightGCN N@20 | CausalDiffRec N@20 | 完整方法 N@20 | 完整方法相对 CausalDiffRec | 完整方法相对 LightGCN | 可支持结论 |
|---|---:|---:|---:|---:|---:|---|
| MovieLens-1M | 0.027902±0.000020 | 0.027822±0.000072 | **0.035068±0.000094** | **+26.04%，5/5，p=0.03125** | **+25.68%，5/5，p=0.03125** | 同时优于两对照 |
| Yelp2018 | **0.016900±0.000244** | 0.012828±0.000742 | 0.015382±0.000175 | **+19.91%，5/5，p=0.03125** | -8.98%，0/5，p=1 | 仅本次 5 seeds 相对 CausalDiffRec 正向 |
| Food | 0.027838±0.000144 | 0.028374±0.000305 | **0.028656±0.000163** | +0.99%，4/5，p=0.1875 | **+2.94%，5/5，p=0.03125** | 仅本次 5 seeds 相对 LightGCN 正向 |
| Amazon Beauty | **0.012526±0.000788** | 0.004964±0.001126 | 0.007214±0.000321 | **+45.33%，5/5，p=0.03125** | -42.41%，0/5，p=1 | 仅本次 5 seeds 相对 CausalDiffRec 正向 |

因此，四数据集实验不能写成“完整方法在所有数据集同时优于 LightGCN 和 CausalDiffRec”。准确结论是：在本次五 seed、该数据划分与 BPR25 预算下，MovieLens 上相对两个对照均为正向；Yelp 与 Amazon Beauty 上仅相对原始 CausalDiffRec 为正向；Food 上仅相对 LightGCN 为 5/5 正向，而相对 CausalDiffRec 尚不确定。`p=0.03125` 是本次五 seed 的单侧精确符号检验，不能外推为跨场景稳定优势。“统一预算”指完整 BPR pass 数相同；因果扩散方法仍需执行其架构固有的表示学习目标。

由于三方法表同时改变因果扩散/LSCI 训练结构、`ConditionFusion`、InfoNCE 与最终语义分数融合，它仅支持组合方法的整体效果；禁止把该表的差异单独归因于因果干预、InfoNCE、语义条件或分数融合中的任一模块。

## 7. activity-adaptive 实验的准确解释

### 7.1 MovieLens-1M

train degree 的分位阈值为 `q1=23`、`q3=79`：

- 低活跃：`degree<=23`，选 `alpha=0.2`；
- 中活跃：`23<degree<=79`，选 `alpha=0.1`；
- 高活跃：`degree>79`，选 `alpha=0`。

OOD NDCG@20：baseline 0.06831，uniform real 0.07037，adaptive real 0.07112，adaptive shuffled 0.06615。adaptive 比 uniform 高 0.000746，5/5 seeds 均为正向。该结果可以支持“根据用户训练期活跃度采用不同语义权重”的一个实施例。

### 7.2 Amazon Beauty

train degree 阈值为 `q1=3`、`q3=7`：

- 低活跃：`degree<=3`，选 `alpha=1.0`；
- 中活跃：`3<degree<=7`，选 `alpha=1.0`；
- 高活跃：`degree>7`，选 `alpha=0.5`。

OOD NDCG@20：baseline 0.01824，uniform real 0.03890，adaptive real 0.03834，adaptive shuffled 0.01466。真实自适应语义远高于 baseline 和 shuffled，但 adaptive 比 uniform 低 0.000556。

因此 Amazon 可以证明“语义融合有效”，却不能证明“自适应分组必然优于统一权重”。专利中可以把分组融合写成可选实施方式，不能把 Amazon 的巨大提升全部归因于自适应规则。

## 8. MovieLens BPR25 与 BPR200 结果的不同用途

项目现有两组合法但回答不同问题的 MovieLens 结果。corrected-v5 BPR25 表把 LightGCN、原始 CausalDiffRec 和完整方法的排序学习预算统一为 25 次完整 BPR pass，用于证明完整方法对现有方法的优势；该表中 `alpha=0.75`，完整方法 NDCG@20 0.035068，相对 CausalDiffRec 为 +26.04%。

corrected-v4 BPR200 表则是纯 LightGCN 长训练预算上的冻结分数融合实验，validation 选定 `alpha=0.1`，baseline/real/shuffled NDCG@20 分别为 0.06831/0.07037/0.06700，用于证明长预算协同 backbone 上语义 late fusion 仍然有效。

两表可分别报告，但不能将 BPR200 LightGCN 绝对分数与 BPR25 完整方法绝对分数放入同一张“公平优势表”，也不能把 +3.02% 和 +26.04% 当成同一实验。

## 9. KuaiRec 失败实验具体做法与结论

### 9.1 v2 exposure 数据协议

KuaiRec v2 exposure 包含 7176 个用户、10612 个物品：train 846414 条、validation 173781 条、OOD test 43448 条。该版本用于缓解旧协议中 validation 与 OOD 来自不同曝光空间的问题。

### 9.2 late fusion 扫描

保持五个冻结 LightGCN checkpoint 不变，在 validation 上扫描 `alpha={0,0.1,0.25,0.5,0.75,1,1.25,1.5,2}`。唯一变化是语义融合权重和 real/shuffled prior。

selection seeds 和 confirmation seeds 均选择 `alpha=0`。因此语义通道没有通过门禁，OOD 测试不应作为正向语义结果。这个结果表明：在当前 KuaiRec 曝光结构下，简单语义 late fusion 无法改善推荐。

### 9.3 popularity rerank

在协同分数中加入流行度惩罚：

\[
s'(u,i)=s_{cf}(u,i)-\gamma\log(1+degree_{train}(i)).
\]

validation 扫描 `gamma={0,0.001,0.0025,0.005,0.01,0.02,0.05,0.1}`，最终选择 `gamma=0`。OOD NDCG@20 为 0.06807，与 baseline 完全相同，说明简单流行度惩罚也不能解决问题。

### 9.4 分组诊断

固定 `alpha=0.5`，不重新训练，只按 train item popularity 统计 target recall@20：tail=0、mid=0、head=0.09354、cold degree<=5 为 0。该结果说明当前模型几乎只命中头部物品，瓶颈更像候选覆盖/曝光偏置，而不是语义权重调小或简单重排问题。

KuaiRec 应写入“适用边界或失败分析”，不能进入正向主表。

## 10. 哪些实验能证明什么

| 证据 | 能支持 | 不能支持 |
|---|---|---|
| 3/5 数据集 real 高于 baseline | 多数数据集上 OOD 效果提升 | 所有数据集都提升 |
| 4 个正式 v4 数据集 real 高于 shuffled，均为 5/5 seeds | 正确语义对应关系对结果有作用 | 任意文本编码都有效 |
| MovieLens activity-adaptive 高于 uniform | 活跃度分组可以作为有效实施方式 | 所有数据集上 adaptive 都更优 |
| Amazon real 大幅高于 baseline/shuffled | Amazon 上语义融合效果强 | 巨大提升全部来自 adaptive 分组 |
| Food v4 real 高于 shuffled、未高于 baseline | 正确语义比错误语义好 | v4 冻结分数融合实验中 Food 优于协同 baseline |
| Food v5 完整方法高于 LightGCN、相对 CausalDiffRec 不稳定 | 组合方法在该 BPR25 条件下可超过 LightGCN | Food 上所有实施方式均优于协同 baseline 或 CausalDiffRec |
| Yelp v3 E3 高于 E0 | 完整模型在 Yelp 上正向 | InfoNCE 是全部增益来源 |
| MovieLens BPR25 完整方法高于 LightGCN/CausalDiffRec，均 5/5 seeds | 统一排序预算下完整方法在 MovieLens 实施例中优于两个对照 | 其他数据集或 BPR200 下必然存在同等幅度 |
| Yelp BPR25 完整方法高于 CausalDiffRec、低于 LightGCN | 本次 Yelp 五 seed 中组合方法相对 CausalDiffRec 为正向 | 完整方法必然优于 LightGCN |
| Food BPR25 完整方法高于 LightGCN、相对 CausalDiffRec 不稳定 | 本次 Food 五 seed 中组合方法相对 LightGCN 为正向 | 完整方法必然优于 CausalDiffRec |
| Amazon Beauty BPR25 完整方法高于 CausalDiffRec、低于 LightGCN | 本次 Amazon Beauty 五 seed 中组合方法相对 CausalDiffRec 为正向 | 完整方法必然优于 LightGCN |
| Yelp DeepSeek+SBERT 高于 raw+SBERT，两种选参口径均 5/5 seeds | DeepSeek 规范化文本在 Yelp 有独立正向贡献 | DeepSeek 对所有数据集都有效 |
| soft gate 与 random gate 接近 | 目前 gate 证据有限 | learned causal gate 已被稳定验证 |
| KuaiRec alpha=0、gamma=0 | 当前方法存在曝光/头部偏置边界 | KuaiRec 上语义增强有效 |

## 11. 专利中建议采用的写法

可直接改写为专利语言的实验结论：

> 在五个公开推荐数据集上进行了严格评测或适用边界诊断，其中四个 v4 语义分数融合数据集在不读取分布外测试标签的条件下，仅利用验证集选择并确认语义融合权重，随后完成冻结的分布外测试。结果显示，该 v4 实施方式在 MovieLens-1M、Yelp2018 和 Amazon Beauty 上均提高了 NDCG@20；在 Food 上虽然未超过协同基线，但正确物品语义显著优于打乱物品—语义对应关系的对照；KuaiRec 未通过验证门禁，作为方法适用边界。由此表明，利用训练期交互形成用户语义画像，并将其与物品语义先验的匹配分数融合到协同推荐分数中，在多数本次评测条件下有助于提升分布外推荐性能。完整因果扩散组合方法的三方法比较应另按数据集和对照方向陈述，不作为上述多数数据集结论的外推。

建议在权利要求或实施方式中保留以下可选结构：

1. 物品文本可直接编码，也可先由大语言模型压缩/规范化后编码。
2. 用户语义画像仅由训练期交互物品的语义向量聚合得到。
3. 两路分数先按用户标准化，再进行加权融合。
4. 融合权重仅在验证数据上选择并冻结。
5. 融合权重可为全局统一值，也可依据训练期用户活跃度分组设置。
6. 可构造打乱物品—语义对应关系的对照，用于验证正确语义关系的作用。

## 12. 禁止出现的表述

- 禁止写“所有数据集均提升”或“每个数据集均显著提升”。
- 禁止笼统写 Food 优于或不优于 collaborative baseline：v4 冻结语义分数融合中 Food 未超过协同 baseline、只优于 shuffled semantic；v5 BPR25 组合方法中 Food NDCG@20 高于 LightGCN、相对 CausalDiffRec 尚不稳定。
- 禁止把 KuaiRec 写成正向结果。
- 禁止把 validation 指标当成 OOD test 指标。
- 禁止把 legacy、corrected-v3、corrected-v4 的绝对分数放在同一公平对比表中。
- 禁止把 MovieLens BPR25 完整方法比较中的 +26.04% 与 BPR200 长预算 late-fusion 中的 +3.02% 混为同一实验或同一训练预算。
- 禁止写完整方法在四个 BPR25 数据集均优于 LightGCN 与 CausalDiffRec；Yelp、Amazon Beauty 低于 LightGCN，Food 相对 CausalDiffRec 的差异未达稳定证据。
- 禁止声称 DeepSeek 在所有数据集优于原始文本；Yelp 正向、Food 负向、KuaiRec 未通过门禁。
- 禁止声称 Amazon activity-adaptive 优于 uniform；Amazon 上 uniform 略高。
- 禁止声称 learned causal gate 稳定优于 random gate；现有结果不支持。
- 禁止把仅运行单 seed 的 force-gate 诊断写成统计显著的正式实验。
- 不要使用旧汇总中“Yelp 提升 6～7 倍”“KuaiRec 基本持平”等过期数字；corrected-v3 终版已经替换这些结果。

## 13. 仍可补充但不影响当前专利主结论的实验

当前结果已经可以支撑一组边界清楚的专利实施例。原始文本-vs-DeepSeek 和 LightGCN/CausalDiffRec/完整方法两项优先缺口已完成，不再列为待做项。若希望让实验部分更完整，可继续补充：

1. 在相同硬件和批大小下，记录 baseline、uniform fusion、adaptive fusion 的训练时间、推理时间、峰值显存和参数量。late fusion 不增加 backbone 训练参数，但会增加语义编码存储和打分开销，应给出实测值。
2. 汇报 IID 与 OOD 两种测试，以显示方法没有只针对 OOD 调整；选参仍只能使用 validation。
3. 增加至少一种其他推荐 backbone，证明语义融合不依赖 LightGCN。
4. 增加用户活跃度、物品流行度和冷启动物品的分组结果，特别是在 MovieLens、Yelp 和 Amazon 上。
5. 如果专利主张 learned causal gate，则必须另做 learned-vs-random 的强机制实验；在补充前，不应将其作为已验证的主要有益效果。

## 14. 唯一可引用来源与代码位置

终版汇总：

- v4 专利主报告：`experiments/reports/patent_experiment_final.md`
- v4 机器可读摘要：`experiments/reports/patent_experiment_summary.json`
- v3 终版 E0/Stage-I/E3：`experiments/reports/strict_ood_e0_stage1_e3_final.md`
- v3 gate 消融：`experiments/reports/strict_ood_ablation_final.md`
- MovieLens BPR200：`experiments/reports/movielens1m_pure_lightgcn_v4_bpr200_ood.json`
- Yelp v4：`experiments/reports/yelp2018_v4_latefusion_ood.json`
- Food v4：`experiments/reports/food_v4_uniform_ood.json`
- Amazon v4：`experiments/reports/amazon_beauty_v4_activity_adaptive_ood.json`
- MovieLens activity-adaptive：`experiments/reports/movielens1m_v4_activity_adaptive_ood.json`
- KuaiRec v2：`experiments/reports/kuairec_v2_latefusion_validation.json`
- KuaiRec popularity：`experiments/reports/kuairec_v2_popularity_rerank.json`
- KuaiRec 分组诊断：`experiments/reports/kuairec_v2_latefusion_groups_real_a05.json`
- 原文-vs-DeepSeek：`experiments/reports/yelp2018_v5_raw_vs_deepseek_ood.json`、`food_v5_raw_vs_deepseek_ood.json`、`kuairec_v5_raw_vs_deepseek_validation.json`
- 三方法公平性审计：`experiments/reports/movielens1m_v5_fair_bpr25_validation_audit.json`、`experiments/reports/yelp2018_v5_fair_bpr25_validation_audit.json`、`experiments/reports/food_v5_fair_bpr25_validation_audit.json`、`experiments/reports/amazon_beauty_v5_fair_bpr25_validation_audit.json`
- 三方法 OOD 结果：`experiments/reports/movielens1m_v5_fair_bpr25_method_comparison.json`、`experiments/reports/yelp2018_v5_fair_bpr25_method_comparison.json`、`experiments/reports/food_v5_fair_bpr25_method_comparison.json`、`experiments/reports/amazon_beauty_v5_fair_bpr25_method_comparison.json`
- 中文详细附加报告：`实验报告9.5/新增专利对比实验详细报告.md`
- 复杂度：`experiments/复杂度分析.md`

关键实现：

- 语义先验构建：`scripts/build_semantic_prior.py`
- 用户画像与融合公式：`utils/late_fusion.py`
- validation-only 统一权重选择：`scripts/validate_late_fusion.py`
- 冻结 OOD 融合评测：`scripts/evaluate_late_fusion.py`
- 活跃度分组权重选择：`scripts/validate_activity_adaptive_fusion.py`
- 活跃度分组 OOD 评测：`scripts/evaluate_activity_adaptive_fusion.py`
- 专利结果汇总：`scripts/summarize_patent_experiments.py`
- raw SBERT 先验生成：`scripts/run_raw_sbert_prior_build.sh`
- 文本来源 validation 消融：`scripts/validate_text_source_ablation.py`
- 文本来源 OOD 汇总：`scripts/summarize_text_source_ood.py`
- 三方法公平性审计：`scripts/audit_fair_method_validation.py`
- 三方法 OOD 汇总：`scripts/summarize_fair_method_ood.py`

四个完成冻结 OOD 测试的正式 v4 数据集，各有 5 个 real 与 5 个 shuffled OOD 原始记录，位于 `experiments/records/`。这些记录保存 seed、checkpoint、数据版本、alpha/profile、语义 prior 元信息、baseline 指标和融合指标，可用于复核表中每个数字。

## 15. 给另一套 AI 的写作任务

请根据本文件撰写专利，不要自行扩大实验结论。建议产出结构为：

1. 背景技术与现有协同推荐在分布变化下的问题；
2. 发明目的；
3. 物品文本语义先验构建；
4. 训练交互约束的用户语义画像构建；
5. 协同分数与语义分数的按用户标准化融合；
6. validation-only 权重确定和 OOD 冻结测试；
7. 可选的用户活跃度自适应融合；
8. 原始文本-vs-DeepSeek 实施例，明确 Yelp 正向及 Food/Kuai 边界；
9. LightGCN、CausalDiffRec 与完整方法的 MovieLens、Yelp2018、Food、Amazon Beauty BPR25 统一预算实施例；限定为本次五 seed：MovieLens 相对两对照均正向、Yelp/Amazon Beauty 仅相对 CausalDiffRec 正向、Food 仅相对 LightGCN 5/5 正向；
10. 四个正面/机制数据集的实施例和 KuaiRec 适用边界；
11. 有益效果，采用“多数数据集有效”的限定措辞；
12. 权利要求，覆盖统一权重与分组权重两种实现，但不要把未经验证的 learned causal gate 效果写成既定事实。
