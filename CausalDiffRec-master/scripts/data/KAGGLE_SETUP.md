# 放置 Kaggle API 密钥（Food 自动下载用）
#
# 1. 打开 https://www.kaggle.com/settings
# 2. API → Create New Token → 下载 kaggle.json
# 3. 执行：
#
#    mkdir -p ~/.kaggle
#    mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json   # 按你的下载路径调整
#    chmod 600 ~/.kaggle/kaggle.json
#
# 4. 告诉我「密钥已放好」，然后运行：
#    conda activate zpp1
#    pip install kaggle -q
#    python -u scripts/data/download_datasets.py --dataset food
#
# 需要文件：RAW_interactions.csv、RAW_recipes.csv
# 数据集：shuyangli94/food-com-recipes-and-user-interactions
