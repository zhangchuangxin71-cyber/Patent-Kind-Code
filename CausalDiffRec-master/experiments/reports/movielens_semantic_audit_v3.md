# MovieLens-1M semantic audit and corrected-v3 late fusion

Date: 2026-08-13  
Protocol: strict train/validation/OOD-test separation; NDCG@20 validation selection  
Seeds: 1024, 2048, 3072, 4096, 5120

## Identity and data audit

- `item_id_map.json` contains 2,380 contiguous item IDs.
- `metadata/items.jsonl` contains exactly one row per ID, in the same identity mapping.
- Mapping mismatches in train, validation, IID-test, and OOD-test parquet files: zero.
- Seventeen dispersed metadata texts were re-encoded with the declared
  `all-MiniLM-L6-v2` model. All 17 retrieved their existing prior row at rank 1.
- Maximum absolute difference between regenerated and stored embeddings:
  `1.27e-7`.

Conclusion: no evidence of item/semantic row misalignment.

## Semantic signal audit (validation only)

- Semantic Top-10 neighbor genre-overlap rate: 88.75%.
- Random neighbor genre-overlap rate: 34.82%.
- Train-history semantic-profile NDCG@20: 0.02268.
- Shuffled-prior profile NDCG@20: 0.00502.
- Popularity profile NDCG@20: 0.10924.

The prior contains real preference-related content signal, but that signal is
much weaker than the collaborative/popularity component when used alone.

## Why conditional E3 failed

- Real and shuffled conditions each changed the diffusion output by about 21%
  relative L2, but real-vs-shuffled outputs had mean row cosine 0.99950.
- Final LightGCN item-to-matched-semantic cosine was 0.364706; random pairing
  was 0.364663.
- Final item-neighbor genre overlap was 35.23%, effectively the random level.
- The diffusion output is detached when it initializes LightGCN, after which
  BPR trains a separate recommendation representation.

Conclusion: the original conditional path reacts to the marginal distribution
of semantic inputs but loses item-identity semantics before final ranking.

## Late-fusion design and selection

The replacement uses:

`zscore(collaborative_score) + alpha * zscore(train_history_semantic_score)`

User semantic profiles are mean-pooled only from train interactions. Alpha was
selected on seeds 1024/2048/3072 and independently confirmed on validation
seeds 4096/5120. The selected value was frozen at `alpha=0.75` before test.

- Selection-seed Val NDCG@20: 0.10986 -> 0.12802.
- Held-out-seed Val NDCG@20: 0.10987 -> 0.12768.
- The same shuffle fusion decreased validation performance.

## One-shot OOD test result

| Method | NDCG@20 | Recall@20 | Precision@20 |
|---|---:|---:|---:|
| Stage1 | 0.027832 ± 0.000085 | 0.028794 ± 0.000201 | 0.019878 ± 0.000052 |
| Stage1 + semantic late fusion | **0.035244 ± 0.000204** | **0.037426 ± 0.000319** | **0.024166 ± 0.000158** |

Paired improvements occurred for all five seeds:

- NDCG@20: `+0.007412` absolute, `+26.63%` relative.
- Recall@20: `+0.008632` absolute, `+29.98%` relative.
- NDCG paired-difference 95% t interval: `[0.007131, 0.007693]`.

With only five seeds, parametric p-values are exploratory. The smallest possible
two-sided exact Wilcoxon p-value when all five differences share a sign is
0.0625, so the result should be reported with effect sizes and per-seed values,
not as definitive distribution-free significance.

## Reproducibility

- Evaluator: `scripts/evaluate_late_fusion.py`
- Runner: `scripts/run_late_fusion_v3_movielens.sh`
- Utility: `utils/late_fusion.py`
- Records: `experiments/records/movielens1m_latefusion_alpha075_strict_ood_seed*_v3.json`

