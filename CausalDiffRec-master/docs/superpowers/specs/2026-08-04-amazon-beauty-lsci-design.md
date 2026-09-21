# Amazon Beauty 5-core for LSCI method validation

Date: 2026-08-04  
Status: approved (user OK)

## Goal

Replace MovieLens (coarse title+genres; semantic unused) with a **rich-text** item dataset under popularity OOD.

## Source

- McAuley / SNAP Amazon product data
- `reviews_Beauty_5.json.gz` (5-core ratings)
- `meta_Beauty.json.gz` (title, description, brand, categories; join on `asin`)

## Protocol

| Step | Rule |
|------|------|
| Positive | rating ≥ 4 |
| Frequency | iterative k-core u≥5, i≥5（15/15 会塌缩到 ~250 用户，不可用） |
| Dense-adj fit | max_nodes=22000 user subsample if needed |
| Split | popularity-uniform OOD 20%, seed=1024 |
| Text | title + categories + description (+ brand) |
| Experiments | E0 + E3 (5-seed), then shuffle seed1024 |

## Non-goals

- Not CausalDiffRec Table-1
- Not replacing Yelp as primary success case

## Outputs

`data_strict/processed/amazon_beauty/v1_strict/`
