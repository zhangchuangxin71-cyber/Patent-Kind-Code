# MovieLens-1M for LSCI method validation

Date: 2026-08-03  
Status: approved (user OK)

## Goal

Replace Douban (no item text) with a dataset that has **aligned item content** for LSCI semantic prior, under **popularity OOD**.

## Source

- GroupLens MovieLens 1M: https://files.grouplens.org/datasets/movielens/ml-1m.zip
- Files: `ratings.dat`, `movies.dat` (Title + Genres keyed by MovieID)

## Protocol

| Step | Rule |
|------|------|
| Positive | rating ≥ 4 |
| Frequency | iterative k-core u≥20, i≥20 |
| Dense-adj fit | if n_user+n_item > 22000 → user subsample (subsample_seed=1024) |
| Split | popularity-uniform OOD 20%, remainder 7:1:2, seed=1024 |
| Text | title + genres → items.jsonl; SBERT semantic_prior |
| Experiments | E0 (CausalDiffRec) + E3 (LSCI SBERT), 5 seeds |

## Non-goals

- Not matching CausalDiffRec Table-1
- Not replacing Yelp as primary LSCI success case

## Outputs

`data_strict/processed/movielens1m/v1_strict/` + E0/E3 records under `experiments/records/`
