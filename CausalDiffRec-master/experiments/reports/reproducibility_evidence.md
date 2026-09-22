# 复现环境与工程证据

Git提交：`722a785e35880baf61979f19b9413c56340a92d6`

## 环境

- Python: `3.8.20`
- PyTorch/CUDA: `2.1.1+cu118` / `11.8`
- DGL/PyG: `2.1.0+cu118` / `2.5.3`
- GPU: `NVIDIA GeForce RTX 4090, 570.181, 24564 MiB`

## 离线/在线边界

DeepSeek 与 SBERT 均为离线处理；在线评价只加载语义向量和 LightGCN 状态，不调用大语言模型，也不执行扩散采样。各数据集的逐文件哈希、训练耗时、显存和评价耗时见 JSON。
