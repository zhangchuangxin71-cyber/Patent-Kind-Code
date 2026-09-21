# Task Plan: 四数据集严格重建 + LSCI 元数据

## Goal
按论文 Appendix C 严格重建 Food / KuaiRec / Yelp2018 / Douban，并准备无测试泄漏的 LSCI 文本元数据；与 `legacy_bin` 隔离。

## Current Phase
Food / KuaiRec / Yelp 严格数据与主实验已完成；Douban 论文级严格对齐保持 BLOCKED

## Phases

### Phase 0: 规格
- [x] 用户确认严格重建口径
- [x] 设计文档落盘并修订审查问题（泄漏/图契约/KuaiRec/READY）
- 文档：`docs/superpowers/specs/2026-07-22-all-datasets-strict-rebuild-design.md`
- **Status:** complete

### Phase 1: 脚手架与原始数据
- [x] `data_strict/` 目录与 `scripts/data/` 公共库
- [x] `dataset_registry.yaml`
- [x] Food、KuaiRec、Yelp 原始数据定位与处理
- [ ] Douban 论文级原始来源确认
- **Status:** complete_except_douban

### Phase 2: 重建与验证
- [x] Food / KuaiRec / Yelp 严格过滤、OOD split、映射、图与文本
- [x] READY 门禁与训练图仅含 train 边的核验
- [ ] Douban 论文级严格重建
- **Status:** complete_except_douban

### Phase 3: 接入训练
- [x] `train.py` / `train_lsci.py` 读取 strict 数据与 READY 门禁
- [x] Food / Yelp / KuaiRec / MovieLens-1M 主实验完成并汇总
- [ ] Douban dense VGAE 的可扩展实现（若恢复该分支）
- **Status:** complete_except_douban

## Decisions
| Decision | Rationale |
|----------|-----------|
| 严格重建，不用双轨混算 | 用户选择；旧结果标 legacy_bin |
| 训练编码只用 train 图 | 防测试边泄漏进消息传递 |
| 选模只用 val | 防 OOD/IID 测试选模 |
| KuaiRec 强制 big/small 去重 | 防跨矩阵重叠 |
| 时变特征不进静态物品文本 | 防时间泄漏 |
| Douban 来源不清则 BLOCKED | 不伪造数据 |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| 规格审查子代理卡住 | 1 | 人工吸收 Issues Found 并修订规格 |
