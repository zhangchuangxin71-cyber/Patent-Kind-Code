# LSCI E2 语义先验

## 依赖

- 文本：`data_strict/processed/<ds>/v1_strict/metadata/items.jsonl`（需 READY）
- 可选 LLM：`export DEEPSEEK_API_KEY=...`（代理默认 `127.0.0.1:7897`）
- 编码：优先 `sentence-transformers`；未安装时用确定性 `hash` 后端

## 构建

```bash
# 无 Key：raw 文本 → SBERT/hash
python scripts/build_semantic_prior.py -d food --mode raw

# 有 Key：DeepSeek 改写 → SBERT
export DEEPSEEK_API_KEY=...
python scripts/build_semantic_prior.py -d food --mode llm --backend sbert
```

输出：`data_strict/processed/<ds>/v1_strict/features/semantic_prior.pt`

## 训练（E3 启用）

```bash
python train_lsci.py -d food \
  --data_root data_strict/processed/food/v1_strict \
  --use_semantic_prior \
  --semantic_prior_path data_strict/processed/food/v1_strict/features/semantic_prior.pt
```

当前 E2 只负责离线先验；门控融合 + InfoNCE 在 E3（模块已脚手架：`condition_fusion.py` / `semantic_loss.py`）。
