# Corrected-v4 controlled protocol

Date: 2026-08-31

## Purpose

This protocol replaces corrected-v3 for all new experiments.  Historical v3
records and reports remain immutable and must not be mixed with v4 results.

## Controlled ranking protocol

- E0 and every LSCI arm share a persistent three-layer LightGCN ranking head.
- Ranking learning rate, batch size and upstream refresh are explicitly recorded.
- The recommender and Adam state persist across outer epochs.
- Checkpoints are selected only by validation NDCG@20.
- Validation-only mechanism experiments do not load test ground truth.

## Causal arms

- `none`: no environment perturbation and no invariant-loss term.
- `random_pair_environment`: pair-symmetric random environments.
- `soft_pair_environment`: random warm-up environments, followed by learned
  pair-symmetric edge weights.  Learned weights are also used by the embedding
  generation forward pass.
- The unused legacy `Graph_Editer_LSCI` is no longer instantiated by LSCI.
- Upstream ranking defaults to zero and must be enabled as an explicit separate
  ablation.

## Semantic arms

- Conditional E3 remains available with matched and shuffled priors.
- `semantic_score_alpha != 0` enables a separate leakage-free score path:
  user profiles are built only from training interactions and fused with the
  collaborative score using per-user standardization.
- Alpha must be selected and confirmed on validation before any OOD test read.

## Correctness fixes

- `inv_mean_weight` now changes the invariant objective as documented.
- Risk-mean schedules have explicit start/end values; the default constant-one
  schedule preserves the previous objective.
- Deterministic generation disables dropout and uses the VGAE posterior mean
  when sampling noise is disabled.
- V4 checkpoints and records use a distinct suffix and protocol marker.

## Promotion gate

Run `scripts/run_v4_mechanism_smoke.sh` first.  Do not launch five-seed OOD
experiments unless validation shows all of the following:

1. learned gate exceeds the random gate by more than run-to-run noise;
2. matched semantics exceed shuffled semantics;
3. the chosen semantic alpha improves held-out validation seeds;
4. no arm loads OOD test ground truth during selection.

## Initial MovieLens validation-only smoke

Seed 1024, four outer epochs, fixed historical validation-selected semantic
alpha 0.75:

| Arm | selected NDCG@20 | final-epoch NDCG@20 |
|---|---:|---:|
| E0 | 0.10972 | 0.10944 |
| none | 0.10959 | 0.10949 |
| random environment | 0.10953 | 0.10938 |
| learned environment | 0.10938 | 0.10938 |
| matched semantic score fusion | 0.12762 | 0.12762 |
| shuffled semantic score fusion | 0.09207 | 0.09207 |

The causal gate did not pass its mechanism gate: learned and random were tied
at the matched final epoch.  The semantic score path passed its initial gate:
matched semantics strongly exceeded both the non-semantic and shuffled arms.
No OOD test ground truth was loaded.  A five-seed, 25-epoch validation-only
semantic confirmation was started; OOD testing remains locked until its checks
all pass.

## Frozen MovieLens OOD result

The five-seed confirmation passed every pre-registered gate, so the frozen
`alpha=0.75` comparison was evaluated once per checkpoint on OOD test:

| Arm | OOD NDCG@20 (mean ± sample std) |
|---|---:|
| none | 0.027838 ± 0.000098 |
| real semantic score fusion | **0.035068 ± 0.000094** |
| shuffled semantic score fusion | 0.026074 ± 0.000401 |

The matched semantic path improved over none by `+0.007230` absolute
(`+26.0%`) and over shuffled by `+0.008994`; all five paired seed differences
were positive.  The learned causal gate remains unsupported: its matched-final
validation score tied the random gate, so it was not advanced to OOD testing.

## Frozen Yelp OOD result

Yelp selected `alpha=0.25` only from validation seeds 1024/2048/3072 and
confirmed it on 4096/5120 before test access. Every validation gate passed.

| Arm | OOD NDCG@20 (mean ± sample std) |
|---|---:|
| alpha=0 collaborative baseline | 0.013448 ± 0.001146 |
| real semantic score fusion | **0.014524 ± 0.000796** |
| shuffled semantic score fusion | 0.013142 ± 0.001147 |

Real semantics improved over the paired alpha-zero baseline by `+0.001076`
absolute (`+8.0%`) and over shuffled by `+0.001382`; all five paired
differences were positive.
