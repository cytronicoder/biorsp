# Archetypes Benchmark Report

**Generated:** 2026-01-12 18:38:16

**Directory:** `/Users/cytronicoder/Documents/GitHub/biorsp-swordfish/analysis/outputs/archetypes`

## Parameters

- **outdir:** /Users/cytronicoder/Documents/GitHub/biorsp-swordfish/analysis/outputs/archetypes
- **seed:** 5000
- **n_reps:** 10
- **N:** [2000]
- **shape:** ['disk']
- **coverage_regime:** ['high', 'low']
- **organization_regime:** ['structured', 'iid']
- **pattern_variant:** wedge_core
- **n_permutations:** 100
- **mode:** quick
- **n_jobs:** 2
- **n_workers:** 2
- **checkpoint_every:** 25
- **resume:** False
- **permutation_scope:** none
- **calibration_file:** None
- **s_cut:** None
- **c_cut:** 0.3

## Summary Statistics

| shape   |    N | coverage_regime   | organization_regime   | true_archetype   |   Spatial_Bias_Score_mean |   Spatial_Bias_Score_std |   Coverage_mean |   Coverage_std |   n_expr_cells_mean |   classification_accuracy |   abstain_rate |   n_reps |
|:--------|-----:|:------------------|:----------------------|:-----------------|---------------------:|--------------------:|----------------:|---------------:|--------------------:|--------------------------:|---------------:|---------:|
| disk    | 2000 | high              | iid                   | housekeeping     |            0.0584649 |          0.0112797  |          0.7433 |      0.0348721 |              1486.6 |                       0.9 |              0 |       10 |
| disk    | 2000 | high              | structured            | regional_program |            0.206898  |          0.0113367  |          0.7515 |      0.021682  |              1503   |                       1   |              0 |       10 |
| disk    | 2000 | low               | iid                   | sparse_noise     |            0.0578299 |          0.00774199 |          0.1237 |      0.0283414 |               247.4 |                       1   |              0 |       10 |
| disk    | 2000 | low               | structured            | niche_marker     |            0.124026  |          0.0134373  |          0.1162 |      0.0338643 |               232.4 |                       1   |              0 |       10 |

## Interpretation

### What does this mean?

Archetypes represent distinct gene expression patterns: **Housekeeping** (ubiquitous), **Niche** (spatially restricted), **Regional** (broad domains), **Scattered** (sparse). We test whether BioRSP's Coverage (C) and Spatial Bias Score (S) can distinguish these patterns.


### Recommended actions

- Archetype boundaries may overlap; consider multi-dimensional analysis.
- Review confusion matrix to identify specific misclassifications.

---

*This report was generated automatically by the BioRSP simulation framework.*