# Genegene Benchmark Report

**Generated:** 2026-01-12 18:38:20

**Directory:** `/Users/cytronicoder/Documents/GitHub/biorsp-swordfish/analysis/outputs/genegene`

## Parameters

- **outdir:** /Users/cytronicoder/Documents/GitHub/biorsp-swordfish/analysis/outputs/genegene
- **seed:** 8000
- **n_reps:** 5
- **N:** [2000]
- **shape:** ['disk']
- **scenario:** ['same']
- **n_permutations:** 100
- **mode:** quick
- **n_jobs:** 2
- **n_workers:** 2
- **checkpoint_every:** 25
- **resume:** False
- **permutation_scope:** none
- **topk_perm:** 100

## Summary Statistics

| shape   |    N | scenario   |   similarity_profile_mean |   similarity_profile_std |   copattern_score_mean |   copattern_score_std |   shared_mask_fraction_mean |   shared_mask_fraction_std |   n_reps |
|:--------|-----:|:-----------|--------------------------:|-------------------------:|-----------------------:|----------------------:|----------------------------:|---------------------------:|---------:|
| disk    | 2000 | same       |                  0.179189 |                  0.39882 |               0.159938 |              0.408923 |                           1 |                          0 |        5 |

## Interpretation

### What does this mean?

Gene-gene co-patterning identifies pairs of genes with similar spatial distributions. AUPRC (Area Under Precision-Recall Curve) measures retrieval quality: 1.0 = perfect, 0.5 = random.


### Recommended actions

- Consider combining spatial correlation with other features (expression level, cell type).

---

*This report was generated automatically by the BioRSP simulation framework.*