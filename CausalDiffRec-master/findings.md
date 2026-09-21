# Findings: 严格数据重建

## 规格审查已吸收的问题
1. **测试图参与编码** → 正式训练只用 `train.bin`；评估用 train 图 + test GT
2. **测试集选模** → 早停/阈值只看 val
3. **KuaiRec 跨矩阵重叠** → small 与 big 的 (u,i) 强制去重
4. **时变元数据泄漏** → daily features / 评论不进静态物品描述
5. **图契约 / READY 门禁** → 写入规格 §6 §8

## 论文目标规模
| Dataset | #U | #I | #Inter | Shift |
|---------|----|----|--------|-------|
| Food | 7809 | 6309 | 216407 | temporal |
| KuaiRec | 7175 | 10611 | 1153797 | exposure |
| Yelp2018 | 8090 | 13878 | 398216 | popularity |
| Douban | 8735 | 13143 | 354933 | popularity |

## Legacy vs 论文
- Douban legacy 规模接近；Yelp 用户物品接近但交互口径不同
- Food / KuaiRec legacy 明显小于论文 → 必须重建

## 数据源候选
- Food: Kaggle food-com-recipes-and-user-interactions
- KuaiRec: kuairec.com / chongminggao/KuaiRec
- Yelp: yelp.com/dataset（可能需手动同意许可）
- Douban: 脚注不完整 → 可能 BLOCKED

## 本地可搜路径
- DRGO: `/media/p520/4CDA9611DA95F782/file/ZCX/推荐系统/DRGO-master/dataset/`

## Food 过滤实证（2026-07-23）
- `rating>=4` + 去重后约 1,003,724 行
- **迭代双部 k-core (15,50)**：约 13 轮后收敛为空 → 不可用
- **顺序单次 user≥15 再 item≥15**：精确得到论文 7809/6309/216407
- 结论：Food 使用 `sequential_user_then_item`，`min_item_inter=15`；写入 registry 与规格
