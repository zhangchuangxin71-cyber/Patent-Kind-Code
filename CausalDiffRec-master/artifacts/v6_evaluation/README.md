# v6 精简评价检查点

每个文件仅保留冻结的 `rec_model` 状态，足以配合对应 portable record、严格数据包和 `scripts/evaluate_late_fusion.py` 复核 A0/A1/A2/A3。它们不含优化器、VGAE、扩散或 ConditionFusion 完整训练状态，不能用于续训。
