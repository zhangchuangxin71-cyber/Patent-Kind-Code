# 四数据集严格重建与语义元数据准备设计

## 1. 目标与口径

为 Food、KuaiRec、Yelp2018、Douban 建立一套独立于现有 `dataset/` 的可复现数据流水线，同时满足：

1. 按 CausalDiffRec 论文 Appendix C 重建 OOD 划分；
2. 为 LSCI-DiffRec 准备节点级文本元数据；
3. 保证用户语义先验仅使用训练交互，不读取验证集或测试集；
4. 输出数据来源、版本、哈希、统计量和验证报告；
5. 现有 bin 与实验统一标记为 `legacy_bin`，不覆盖、不混用。

“严格重建”优先要求使用论文对应的数据快照。若历史快照不可取得，则按相同公开规则生成 `rule-equivalent` 版本，并在清单和实验报告中明确披露，不能宣称逐条等同论文数据。

## 2. 数据源

### 2.1 Food

- 交互：Food.com `RAW_interactions.csv`
- 物品元数据：`RAW_recipes.csv`
- 关键字段：`user_id`、`recipe_id`、`date`、`rating`、`review`、`name`、`tags`、`description`、`ingredients`、`steps`
- 来源：Kaggle `shuyangli94/food-com-recipes-and-user-interactions`（对应 ACL D19-1613）
- 许可：按 Kaggle/原作者条款；下载前记录 URL 与许可摘要到 manifest

### 2.2 KuaiRec

- 训练域：`big_matrix.csv`
- OOD 测试域：近全曝光的 `small_matrix.csv`
- 元数据：`item_categories.csv`、`item_daily_features.csv`、`kuairec_caption_category.csv`（若存在）
- 关键字段：`user_id`、`video_id`、`watch_ratio`、时间、类别、caption、静态视频属性
- 来源：https://kuairec.com / GitHub `chongminggao/KuaiRec`
- 许可：按官方发布条款记录到 manifest

### 2.3 Yelp2018

- 交互：Yelp Open Dataset 对应历史快照的 reviews
- 物品元数据：business records
- 关键字段：`user_id`、`business_id`、`stars`、`date`、`name`、`categories`、`attributes`
- 优先寻找与论文统计对应的 2018 快照；当前官方下载仅作为无法获得历史版时的 `rule-equivalent` 备选
- 许可：Yelp Dataset License；下载需用户确认同意后进行

### 2.4 Douban

- 交互：论文引用的 Douban 原始评分数据
- 物品元数据：电影标题、类型、简介等公开字段
- 论文在线脚注链接不完整，下载前必须确认来源、字段和许可；不能用无法追溯的镜像冒充官方快照
- 若无法确认：该数据集标记 `BLOCKED`，不进入 READY，不影响其他三个数据集推进

## 3. 目录与版本隔离

```text
data_strict/
├── raw/<dataset>/<source_version>/       # 原始文件，只读
├── interim/<dataset>/                    # 清洗和映射中间结果
├── processed/<dataset>/<version>/
│   ├── interactions/
│   │   ├── train.parquet
│   │   ├── val.parquet
│   │   ├── iid_test.parquet
│   │   └── ood_test.parquet
│   ├── metadata/
│   │   ├── users.jsonl
│   │   └── items.jsonl
│   ├── mappings/
│   │   ├── user_id_map.json
│   │   ├── item_id_map.json
│   │   └── node_id_map.json
│   ├── graphs/
│   │   ├── train.bin                 # 仅训练边；训练编码唯一允许图
│   │   ├── train_plus_val.bin         # 可选：仅用于验证阶段构图，禁止选参以外用途
│   │   ├── eval_context_iid.bin       # IID 评估上下文（见 §6）
│   │   └── eval_context_ood.bin       # OOD 评估上下文（见 §6）
│   ├── features/
│   │   └── node_feat.pt               # 仅用 train 拟合的节点特征
│   ├── manifest.json
│   ├── validation_report.json
│   └── READY                          # 仅当全部门禁通过时写入
└── cache/
```

原始大文件和生成数据不提交版本控制；代码、manifest 模板、统计报告和小型测试夹具可纳入项目。

## 4. 统一预处理规则

### 4.1 正反馈与迭代过滤

- Food、Yelp2018、Douban：保留 `rating >= 4`；
- KuaiRec：保留 `watch_ratio >= 2`；
- Food：用户至少 15 条正交互；
- Yelp2018、Douban：用户至少 25 条正交互；
- Food：**物品至少 15 条**正交互，且采用**顺序单次**过滤（先用户后物品）。实证：该规则精确复现论文规模 7809/6309/216407；若按物品≥50 做全迭代双部 k-core，Food.com 当前快照会收敛为空集；
- Yelp2018、Douban：物品至少 50 条正交互；默认使用迭代 k-core 直至收敛（若与论文规模偏差过大，再在 manifest 披露并改为与论文对齐的顺序规则）；
- KuaiRec：**不**对用户/物品施加与 Food/Yelp 相同的 k-core（论文仅规定 watch_ratio；额外过滤须写入 `extra_filters` 并说明与论文偏差）；
- 重复用户—物品交互：保留最新时间戳一条（无时间戳则保留评分更高者；仍冲突取首次出现），规则写入 manifest。

### 4.2 OOD 划分（划分契约）

固定全局 `split_seed = 1024`（可在 registry 覆盖，必须写入 manifest）。

每个交互记录至少包含：`raw_user_id`、`raw_item_id`、`timestamp`（可空）、`value`、`split ∈ {train,val,iid_test,ood_test}`。

#### Food：时间偏移

每个用户按时间降序排列，将最近 20% 正交互作为 OOD 测试（向下取整至少 1，若用户剩余不足则跳过该用户进入 OOD）；剩余交互按 `split_seed` 随机划分为训练、验证、IID 测试，比例 7:1:2。

#### Yelp2018 / Douban：流行度偏移

算法（确定性，写入 manifest）：

1. 计算过滤后物品频率 \(f_i\)；
2. 目标：抽取约 20% 交互进入 OOD，使 OOD 中物品出现次数尽可能接近均匀；
3. 实现：按物品分组，对每个物品按 \( \min(\mathrm{count}_i, \lfloor \gamma \cdot \mathrm{median\_count} \rfloor) \) 与配额约束采样，再全局微调到总交互比例 ≈20%（容差 ±1%）；
4. 输出训练域与 OOD 的物品度分布、Gini、JS 散度；验证要求 `gini_ood < gini_train` 且 `js_ood_vs_uniform < js_train_vs_uniform`；
5. 剩余交互按 7:1:2 划分 train/val/iid_test。

#### KuaiRec：曝光偏移 + 跨矩阵去重

1. `big_matrix` 正反馈过滤后按 7:1:2 → train/val/iid_test；
2. `small_matrix` 正反馈过滤后作为 ood_test **候选**；
3. **强制去重**：删除 small 中与 big 已出现的 `(user,item)` 完全相同的交互（官方说明 small 交互已从 big 排除；若发现残留，必须剔除并在报告中计数）；
4. 节点映射：以 **train∪val∪iid_test∪ood_test** 中出现的用户/物品并集建映射；仅出现在 OOD 的节点记为 `cold_start=true`；
5. 报告：OOD 冷启动用户/物品比例、与 big 重叠剔除条数。

### 4.3 防泄漏约束（硬规则）

- 用户 Prompt 历史 **只来自 train**；
- 物品 **静态** 文本（标题、类别、简介）可跨划分；
- **时变元数据禁止泄漏**：
  - KuaiRec `item_daily_features` 只能使用训练时间窗内、且不晚于该物品在 train 中最后交互日的统计；
  - Yelp/Food 评论文本、交互级 review **不得**写入物品静态描述（避免把测试评论当物品先验）；
  - 流行度、平均评分、共现统计 **只拟合 train**；
- 标准化器、词表、节点特征统计 **仅拟合 train**；
- 每个 `(user,item)` 正交互只能属于一个 split；
- **选模与早停**：只允许使用 `val`；`iid_test` 与 `ood_test` **禁止**参与任何超参/阈值/早停决策；
- **编码图契约**：模型训练时的图编码器输入 **只能是 train 图**（或明确声明的 train 诱导子图）。禁止把 ood_test / iid_test 边并入训练编码图。评估时使用 §6 规定的评估上下文图，且评估上下文不得回传梯度。

## 5. LSCI 文本元数据

统一物品记录格式：

```json
{
  "raw_item_id": "...",
  "item_id": 0,
  "title": "...",
  "categories": ["..."],
  "description": "...",
  "attributes": {"...": "..."},
  "text_available": true,
  "source_fields": ["..."],
  "cold_start": false
}
```

用户记录只保留映射与可选静态属性（不含测试行为画像）。离线 Prompt 构建器根据 `train.parquet` 取历史。

文本优先级：

- Food：名称 > 标签/食材 > 描述（**不含** steps 全文时截断至 N 字符，默认 512）；
- KuaiRec：caption > 类别 > 静态上传属性（**不含**日更播放量等时变特征原文）；
- Yelp2018：商户名称 > 类别 > 静态 attributes；
- Douban：标题 > 类型 > 简介。

缺失文本：`text_available=false`，保留节点，用 `semantic_mask` 处理。

## 6. 图契约

节点编号：用户 `0 .. n_user-1`，物品 `n_user .. n_user+n_item-1`，连续、与 mapping 一致。

图文件语义：

| 文件 | 边来源 | 用途 |
|------|--------|------|
| `train.bin` | 仅 train | **训练编码唯一允许图** |
| `train_plus_val.bin` | train∪val | 可选；仅验证阶段构图参考，不用于最终 OOD 报告训练 |
| `eval_context_iid.bin` | 见下 | IID 测试编码/检索上下文 |
| `eval_context_ood.bin` | 见下 | OOD 测试编码/检索上下文 |

**评估上下文边（默认，与 legacy CausalDiffRec 兼容并写明）**：

- `eval_context_ood`：用 **train 边** 构图，在 **ood_test 交互** 上做全排序评估（测试边不进入消息传递，只作为 ground-truth）；
- `eval_context_iid`：同理，ground-truth 为 iid_test。

若某实验需要“测试图也编码”（仅诊断），必须另存 `diag_*` 文件并在 run 记录中标记 `eval_leakage_mode=encode_test_graph`，**不得**进入正式表。

节点特征 `node_feat.pt`：维度与数据集 registry 一致；仅用 train 统计初始化（随机高斯或 train 侧属性编码）；全图节点共享同一特征表，冷启动节点用未知向量。

## 7. 处理组件

```text
scripts/data/
├── download_datasets.py
├── prepare_all.py
├── prepare_food.py
├── prepare_kuairec.py
├── prepare_yelp.py
├── prepare_douban.py
├── build_graphs.py
├── validate_dataset.py
├── common/
│   ├── io.py
│   ├── filters.py
│   ├── splits.py
│   ├── mapping.py
│   ├── graph_export.py
│   └── manifest.py
└── dataset_registry.yaml
```

## 8. 验证标准与 READY 门禁

`validate_dataset.py` 必须全部通过才写 `READY`：

1. 原始文件 SHA-256 与 schema 完整；许可字段非空；
2. 正反馈与阈值规则满足（KuaiRec 按其特殊规则）；
3. 四 split 交互两两无 `(user,item)` 重叠；
4. KuaiRec：big/small 重叠剔除计数已记录；冷启动比例已记录；
5. ID 连续；用户在前、物品在后；
6. parquet / mapping / metadata / graphs / features 节点数一致；
7. Prompt 样例抽样：历史边 ⊆ train；
8. 时变特征未混入 `items.jsonl` 的静态 description；
9. Food：每用户 OOD 时间 ≥ 其 train 最大时间；
10. Yelp/Douban：`gini_ood < gini_train` 且相对均匀指标改善；
11. 图契约：`train.bin` 边集 = train 交互；eval 图不含测试边作消息传递边；
12. 统计量与论文表对比；偏差写入 `discrepancy`（允许 `rule-equivalent`）；
13. `manifest.json` 含 `data_status ∈ {paper-snapshot, rule-equivalent}`、`split_seed`、`source_urls`、`license_notes`。

任一失败：删除或拒写 `READY`，退出码非 0。

## 9. 迁移与实验策略

- 现有 `dataset/` 与结果 = `legacy_bin`；
- 严格版训练必须读 `data_strict/processed/.../READY`；
- 记录字段强制：`data_version`、`source_version`、`manifest_sha256`、`data_status`；
- 选模只用 val；正式汇报 OOD（主）与可选 IID；
- 旧结果不与严格版混算均值；
- 后台 Yelp E1 可完成，归入 `legacy_bin`。

## 10. 风险与处置

- Yelp 历史快照不可得 → `rule-equivalent` + 版本披露；
- Douban 来源不清 → `BLOCKED`，不伪造；
- 需登录/许可 → 阻塞并请用户操作；
- 下载失败 → 断点续传 + 哈希隔离；
- 流行度采样非论文逐条 → 算法披露 + 分布指标。

## 11. 完成定义

四数据集各自（或 Douban 显式 BLOCKED）具备：可追溯源、可重复命令、四路 split、映射与图、静态文本、manifest/验证报告、`data_status`、通过门禁的 `READY`（或 `BLOCKED.json`）。
