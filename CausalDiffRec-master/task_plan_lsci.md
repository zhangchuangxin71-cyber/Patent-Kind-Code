# Task Plan: LSCI-DiffRec 实现与实验

## Goal
按批准设计完成 LSCI-DiffRec（E0–E5），并产出可复现均值±标准差结果。

## Current Phase
主实验与 Food / Yelp 语义消融均已归档；仅剩已阻塞的 Douban 分支与可选效率整理

## Phases

### E0: Baseline 冻结
- [x] **Status:** done

### E1: Stage I 无 LLM
- [x] Food / Yelp 5-seed（legacy_bin）
- **Status:** done

### E2: 语义先验
- [x] DeepSeek LLM 改写 + hash 编码（Food/KuaiRec/Yelp，覆盖率 100%）
- **Status:** done

### E3: 完整 LSCI（融合 + InfoNCE）
- [x] `ConditionFusion` / `SemanticInfoNCE` 接入 `train_lsci.py`
- [x] Food / Yelp2018 / KuaiRec / MovieLens-1M strict OOD：E3 25 epoch × 5 seeds
- [x] corrected-v3 E0 / Stage-I / E3 统一终版汇总：`experiments/reports/strict_ood_e0_stage1_e3_final.md`
- **Status:** complete

### E4–E5
- [x] Yelp corrected-v3 E4（`lambda_sem=0`）与 shuffled-E3：各 25 epoch × 5 seeds；机制证据为正向但有限
- [x] Food corrected-v3 E4（`lambda_sem=0`）：25 epoch × 5 seeds；E3 相比 E4 的 N@20 仅 +0.4%，无稳定增益
- [ ] Douban 严格复现（BLOCKED：数据来源与 dense VGAE 规模）
- **Status:** complete_except_blocked

## Notes
- legacy vs strict **分表**，不混算
- GPU 已空闲；E3 使用 `--use_semantic_prior --lambda_sem 0.1`
