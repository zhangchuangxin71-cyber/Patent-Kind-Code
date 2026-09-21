# LSCI-DiffRec 实验与实现设计

## 目标

在 CausalDiffRec Baseline 上实现 LSCI-DiffRec，按论文 RQ1–RQ5 完成可复现实验；Baseline `train.py` 保持可独立运行。

## 已确认决策

- LLM：DeepSeek `deepseek-chat`；代理 `http://127.0.0.1:7897`
- Key：环境变量 `DEEPSEEK_API_KEY`（由用户自行填入）
- 语义编码：LLM 文本 → 本地 Sentence-BERT → `semantic_prior.pt`
- 推进路径：分阶段 E0→E1→E2→E3→E4→E5
- 不做：LLM 图编辑；改前向扩散；MSE 语义对齐

## 阶段

### E0 Baseline 冻结
整理已有 5-seed CausalDiffRec 结果为对照表。

### E1 Stage I（无 LLM）
- `CausalEdgeScorer` 边分
- \(G_c/G_v\) 按 `causal_keep_ratio` 划分
- 环境生成器仅编辑 \(G_v\)
- \(L_{inv}\) + budget
- 入口：`train_lsci.py`（`--use_semantic_prior false`）

### E2 语义先验
- 文本元数据准备
- `build_semantic_prior.py` 离线调用 DeepSeek + SBERT 缓存

### E3 完整 LSCI
- 门控融合 \(z_c,z_s\)
- InfoNCE \(L_{sem}\)
- 四数据集 × 5 seeds

### E4 / E5
消融与效率 / Token / 显存

## 文件清单

新增：`train_lsci.py`、`parameters_lsci.py`、`modules/causal_score.py`、`modules/invariant_loss.py`、`modules/condition_fusion.py`、`modules/semantic_loss.py`、`modules/generator_lsci.py`、`utils/semantic_prior.py`、`scripts/build_semantic_prior.py`、`scripts/prepare_text_meta.py`

Baseline 不破坏：原 `train.py` / `generator.py` / `parameters.py` 保持可用。

## 成功标准

- E1：相对 Baseline 不崩，至少一个数据集有稳定增益迹象
- E3：≥2 个 OOD 设置上优于 CausalDiffRec（5-seed）
- E4：去掉语义模块后指标下降
