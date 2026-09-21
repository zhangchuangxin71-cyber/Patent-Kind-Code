# 严格数据划分与无泄漏审计

协议：`corrected_v6_strict_split_and_leakage_audit`

| 数据集 | train/val/IID/OOD 交互数 | JS散度 | 未见用户/物品 | 交互对零重叠 | 图仅含训练边 |
|---|---:|---:|---:|---:|---:|
| yelp2018 | 294657/42093/84189/105234 | 0.119806 | 0/1 | True | True |
| movielens1m | 309835/44262/88525/110655 | 0.188721 | 0/1 | True | True |
| food | 117713/13854/41453/43387 | 0.074284 | 135/2 | True | True |
| amazon_beauty | 52723/7531/15065/18829 | 0.241119 | 106/168 | True | True |

## 自动检查

- `user_profiles_use_train_interactions_only`: **True**
- `alpha_search_loader_sets_load_test_gt_false`: **True**
- `llm_input_uses_item_metadata_not_user_history`: **True**
- `candidate_scoring_masks_train_interactions`: **True**

## Food 时间协议说明

- 最后20%计数规则：**True**
- OOD时间不早于其余划分：**True**
- 严格晚于（同日并列会失败）：**False**
- Food interactions use per-user temporal OOD splitting, but recipe text comes from the available metadata dump and is not an interaction-time semantic snapshot.
