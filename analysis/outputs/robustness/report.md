# Robustness Benchmark Report

**Generated:** 2026-01-12 18:38:26

**Directory:** `/Users/cytronicoder/Documents/GitHub/biorsp-swordfish/analysis/outputs/robustness`

## Parameters

- **outdir:** /Users/cytronicoder/Documents/GitHub/biorsp-swordfish/analysis/outputs/robustness
- **seed:** 1000
- **n_reps:** 5
- **N:** [2000]
- **shape:** ['disk']
- **pattern:** ['wedge']
- **distortion_kind:** ['none', 'rotate']
- **n_permutations:** 100
- **mode:** quick
- **n_jobs:** 2
- **n_workers:** 2
- **checkpoint_every:** 25
- **resume:** False
- **permutation_scope:** none

## Summary Statistics

| shape   |    N | pattern   | distortion_kind   |   distortion_strength | category   |   baseline_s_mean |   baseline_s_std |   distorted_s_mean |   delta_mean |   delta_median |   delta_std |   abs_delta_median |   abs_delta_iqr |   correlation | is_stable   |   n_pairs |   abstain_rate |
|:--------|-----:|:----------|:------------------|----------------------:|:-----------|------------------:|-----------------:|-------------------:|-------------:|---------------:|------------:|-------------------:|----------------:|--------------:|:------------|----------:|---------------:|
| disk    | 2000 | wedge     | none              |                     0 | invariance |         0.0303617 |       0.0062005  |          0.0303617 |  0           |              0 | 0           |                  0 |               0 |             1 | True        |         5 |              0 |
| disk    | 2000 | wedge     | rotate            |                     0 | invariance |         0.0314346 |       0.00987177 |          0.0314346 |  0           |              0 | 0           |                  0 |               0 |             1 | True        |         5 |              0 |
| disk    | 2000 | wedge     | rotate            |                    15 | invariance |         0.0294169 |       0.0075155  |          0.0294169 | -6.93889e-19 |              0 | 1.38778e-18 |                  0 |               0 |             1 | True        |         5 |              0 |
| disk    | 2000 | wedge     | rotate            |                    45 | invariance |         0.0255566 |       0.00242589 |          0.0255566 |  6.93889e-19 |              0 | 1.38778e-18 |                  0 |               0 |             1 | True        |         5 |              0 |
| disk    | 2000 | wedge     | rotate            |                    90 | invariance |         0.0302056 |       0.00553964 |          0.0302056 | -1.38778e-18 |              0 | 2.77556e-18 |                  0 |               0 |             1 | True        |         5 |              0 |
| disk    | 2000 | wedge     | rotate            |                   180 | invariance |         0.0285162 |       0.00491924 |          0.0285162 | -6.93889e-19 |              0 | 1.38778e-18 |                  0 |               0 |             1 | True        |         5 |              0 |

## Interpretation

### What does this mean?

Robustness measures stability of BioRSP scores under coordinate perturbations (rotation, scaling, jitter). Low delta = stable, high delta = sensitive.


### Recommended actions

- Be cautious with heavily distorted or subsampled datasets.
- Consider preprocessing standardization for cross-study comparisons.

---

*This report was generated automatically by the BioRSP simulation framework.*