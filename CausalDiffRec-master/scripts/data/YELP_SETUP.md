# Yelp 数据准备说明

## 推荐方式（Kaggle，已有 `~/.kaggle/kaggle.json`）

1. 浏览器打开并 **Accept** 数据集条款：  
   https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset
2. 下载并 staging：

```bash
cd CausalDiffRec-master
nohup bash -c 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate zpp1 && python -u scripts/data/download_datasets.py --dataset yelp2018' > logs/download_yelp.log 2>&1 &
```

仅拉取 `review` + `business`（约 5GB+），不拉 `user.json`。

3. 预处理 + 校验：

```bash
nohup bash -c 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate zpp1 && python -u scripts/data/prepare_yelp.py && python -u scripts/data/validate_dataset.py --dataset yelp2018' > logs/yelp_prepare.log 2>&1 &
```

## 手动方式

将下列文件放到 `data_strict/raw/yelp2018/kaggle_yelp_dataset/`（或 `official/` / `pending/`）：

- `yelp_academic_dataset_review.json`
- `yelp_academic_dataset_business.json`

## 口径

- 状态：`rule-equivalent`（当前 Open Dataset ≠ 论文 2018 历史快照保证）
- 过滤默认：`rating>=4` + 迭代 k-core `user>=25` / `item>=50`
- 划分：流行度均匀 OOD（~20%）
- **评论文本不进**物品静态描述（防泄漏）
