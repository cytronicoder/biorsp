# Inference And Nulls

## Primary Null

For a fixed scope and embedding, directional anisotropy is evaluated against within-strata permutation of foreground assignments.

- Test statistic: `T = max_theta |R(theta)|`
- `p_T` is computed from the same max-over-theta statistic under permutation.

## Strata Definition

- Preferred: donor strata (`hubmap_id` or resolved donor key).
- Fallback: library-size quantile strata when donor key is missing or unusable.

When fallback strata are used, run metadata marks inference as limited.

## Multiplicity

- BH FDR is applied within each scored scope to produce `q_T`.
- Cross-scope multiplicity is not jointly controlled.

## Additional Diagnostics

- Moran's I is reported per feature (`continuous` and `binary` when defined).
- Negative controls include:
  - high `pct_counts_mt`
  - ribosomal high cells (or total-count proxy)
  - within-strata permutation sanity diagnostic

These diagnostics support interpretation quality; they are not proof of biological discovery.
