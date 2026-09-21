# 数据与模型材料提供说明

本仓库公开代码、实验记录、审计报告、文件哈希和可复核的精简评价检查点。第三方原始数据及其直接派生副本不直接提交到公开 Git 仓库；请从官方来源下载，并使用 `scripts/data/` 下的固定种子脚本重建 `v1_strict`。

原因不是文件缺失，而是数据再分发条款：

- MovieLens-1M 官方 README 明确要求未经单独许可不得再分发数据；
- Yelp 数据需遵守 Yelp Dataset License，公开再分发前需取得相应授权；
- Food.com Kaggle 页面将数据文件版权标为原作者所有，未给出允许任意再分发的开放许可证；
- Amazon Beauty 数据由 Stanford/UCSD 页面提供，使用时应遵守其来源条款并引用原论文。

官方入口：

- Yelp Open Dataset: https://www.yelp.com/dataset
- MovieLens-1M: https://grouplens.org/datasets/movielens/1m/
- Food.com: https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions
- Amazon Reviews: https://snap.stanford.edu/data/amazon/productGraph/

## 可审计重建

每个本地 `v1_strict` 包的以下材料均由审计脚本记录存在性、大小和 SHA-256：

- `READY` 与 `manifest.json`；
- train/validation/IID/OOD parquet；
- train 与 OOD evaluation-context 图；
- 用户、物品映射；
- 物品元数据；
- 语义先验及打乱语义先验。

运行：

```bash
python scripts/audit_strict_protocol.py
```

审计输出位于：

- `experiments/reports/strict_data_leakage_audit.json`
- `experiments/reports/strict_data_leakage_audit.md`

如数据权利人已向使用者授予再分发许可，可依据审计 JSON 中的文件清单与哈希另行提供完整评价包；API 密钥始终不应纳入评价包。
