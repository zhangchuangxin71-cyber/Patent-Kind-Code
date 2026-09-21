# Progress Log

## 2026-07-23
- **Food READY**：精确对齐论文 7809/6309/216407
- **KuaiRec READY**
- **Yelp READY**：rule-equivalent 10520/12185/526173
- **训练接入 data_strict（完成）**
  - `utils/load_strict.py`：READY 门禁 + train.bin 编码 + parquet GT
  - `train_lsci.py`：`--data_root` / `--eval_split {ood,iid}`；legacy 默认不变
  - 修复 BPR 物品下标需 `+ n_user`
  - 冒烟：`scripts/smoke_strict_eval.py` → Food OOD 评估链路 **SMOKE_OK**
  - 完整 GPU 训练：当前 GPU0 被 PID 149908 占 ~16GB，dense VGAE 反向 OOM；空闲后示例：
    `python train_lsci.py -d food --data_root data_strict/processed/food/v1_strict --eval_split ood`
- 待办：Douban BLOCKED；GPU 空闲后跑 strict 正式实验

## 2026-07-22
- 用户确认：四数据集严格重建 + 文本元数据
- 规格写入并修订：`docs/superpowers/specs/2026-07-22-all-datasets-strict-rebuild-design.md`
- 脚手架：`scripts/data/*` + `data_strict/`
- KuaiRec 首次 zip 因 wget `-c` 拼坏（817MB=垃圾+正本）；已改干净重下
- deps：pyyaml/pyarrow 已就绪
- Food/Yelp 需手动；Douban BLOCKED
