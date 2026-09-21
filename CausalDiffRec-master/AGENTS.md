# Agent 操作约定（CausalDiffRec / LSCI）

与项目根 `.cursor/rules/no-connection-timeout.mdc` 一致，此处供人工与 Agent 快速查阅。

## 核心原则

**长任务不进对话、只进后台日志。**

- 训练 / 下载 / prepare：`nohup ... > logs/<name>.log 2>&1 &`
- 查进度：`tail -n 20 logs/<name>.log`（禁止长时间 Await）
- 禁止全盘 `find`；禁止打印 `kaggle.json` 等密钥内容

## 常用日志

| 任务 | 日志 |
|------|------|
| Food 下载 | `logs/download_food.log` |
| KuaiRec 下载 | `logs/download_kuairec_clean.log` |
| Food prepare | `logs/food_prepare.log` |
| KuaiRec prepare | `logs/kuairec_prepare2.log` |
| Yelp 下载 | `logs/download_yelp.log` |
| Yelp prepare | `logs/yelp_prepare.log` |
| Food strict 冒烟 | `logs/food_strict_eval_smoke.log` |
| Food E3 strict | `logs/food_lsci_e3_strict.log` |
| LLM 语义先验 | `logs/semantic_prior_llm.log` |
| LSCI multi-seed | `logs/lsci_*_multiseed_master.log` |

## 数据路径

- Legacy：`dataset/`（标记 `legacy_bin`，勿与 strict 混算）
- Strict：`data_strict/raw/` → `data_strict/processed/<ds>/v1_strict/`
- READY 门禁：`data_strict/processed/<ds>/v1_strict/READY`
