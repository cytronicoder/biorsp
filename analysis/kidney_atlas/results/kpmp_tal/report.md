# KPMP All-Gene Archetypes Report

**Generated:** 2026-01-11 21:00:33

## Dataset Summary

- **Cells analyzed:** 38,384
- **Embedding used:** `X_umap`
- **Subset query:** `subclass.l1` == "TAL"
- **Total genes in dataset:** 31,332
- **Genes passing filters:** 15,595

## Gene Filtering

| Criterion | Threshold | Passed |
|-----------|-----------|--------|
| Coverage | ≥ 0.005 | 15,595 |
| Nonzero cells | ≥ 50 | 19,254 |
| **Both** | - | **15,595** |

## Threshold Selection

### Coverage Cutoff (c_cut)
- **Value used:** 0.0711
- **Method:** auto_derived

### Spatial Bias Score Cutoff (s_cut)
- **Value used:** 0.1728
- **Method:** distribution_derived

## Archetype Classification

| Archetype | Count | Percentage |
|-----------|-------|------------|
| I: Ubiquitous | 7,650 | 49.1% |
| III: Patchy | 244 | 1.6% |
| II: Gradient | 149 | 1.0% |
| IV: Basal | 7,552 | 48.4% |

**Total classified:** 15,595

## Top Genes per Archetype

### I: Ubiquitous (n=7,650)

| Gene | Coverage (C) | Spatial (S) | Sign |
|------|--------------|-------------|------|
| ENSG00000198712 | 0.9998 | 0.1035 | 1 |
| ENSG00000198899 | 0.9998 | 0.1339 | 1 |
| ENSG00000198886 | 0.9997 | 0.1048 | -1 |
| ENSG00000198804 | 0.9997 | 0.0769 | -1 |
| ENSG00000198840 | 0.9997 | 0.1201 | 1 |
| ENSG00000198938 | 0.9996 | 0.0664 | 1 |
| ENSG00000198727 | 0.9993 | 0.0892 | 1 |
| ENSG00000198763 | 0.9990 | 0.0852 | 1 |
| ENSG00000198888 | 0.9978 | 0.0786 | 1 |
| ENSG00000198786 | 0.9941 | 0.0887 | -1 |

### III: Patchy (n=244)

| Gene | Coverage (C) | Spatial (S) | Sign |
|------|--------------|-------------|------|
| ENSG00000185112 | 0.0691 | 0.1872 | -1 |
| ENSG00000116983 | 0.0681 | 0.2787 | 1 |
| ENSG00000106541 | 0.0679 | 0.2111 | 1 |
| ENSG00000173068 | 0.0662 | 0.2114 | -1 |
| ENSG00000197249 | 0.0630 | 0.1770 | -1 |
| ENSG00000162670 | 0.0619 | 0.2094 | 1 |
| ENSG00000286481 | 0.0604 | 0.2273 | 1 |
| ENSG00000248441 | 0.0588 | 0.1861 | -1 |
| ENSG00000111424 | 0.0584 | 0.2213 | 1 |
| ENSG00000114854 | 0.0564 | 0.2088 | 1 |

### II: Gradient (n=149)

| Gene | Coverage (C) | Spatial (S) | Sign |
|------|--------------|-------------|------|
| ENSG00000137731 | 0.9777 | 0.1818 | 1 |
| ENSG00000156508 | 0.9506 | 0.2067 | 1 |
| ENSG00000229117 | 0.9430 | 0.2302 | 1 |
| ENSG00000143153 | 0.9349 | 0.2064 | 1 |
| ENSG00000164825 | 0.9275 | 0.2240 | 1 |
| ENSG00000137818 | 0.9201 | 0.1901 | 1 |
| ENSG00000163399 | 0.9162 | 0.1761 | 1 |
| ENSG00000233927 | 0.9090 | 0.2092 | 1 |
| ENSG00000167526 | 0.9077 | 0.1783 | 1 |
| ENSG00000138326 | 0.8969 | 0.1870 | 1 |

### IV: Basal (n=7,552)

| Gene | Coverage (C) | Spatial (S) | Sign |
|------|--------------|-------------|------|
| ENSG00000250802 | 0.0711 | 0.0842 | 1 |
| ENSG00000148411 | 0.0711 | 0.0658 | 1 |
| ENSG00000178852 | 0.0711 | 0.1062 | -1 |
| ENSG00000088538 | 0.0710 | 0.1329 | 1 |
| ENSG00000176715 | 0.0710 | 0.0625 | 1 |
| ENSG00000100281 | 0.0710 | 0.0617 | 1 |
| ENSG00000286214 | 0.0710 | 0.1617 | 1 |
| ENSG00000197776 | 0.0710 | 0.0667 | 1 |
| ENSG00000101546 | 0.0710 | 0.0568 | 1 |
| ENSG00000198736 | 0.0710 | 0.0877 | 1 |

## Reliability Checks

### Subsample Stability (70% splits)
- **Spearman correlation:** 0.877
- **Median |ΔS|:** 0.0047
- **Genes tested:** 500

## Output Files

- `runs_all_genes.csv`: Complete per-gene results
- `classification.csv`: Gene-to-archetype mapping
- `derived_thresholds.json`: Threshold derivation details
- `manifest.json`: Run provenance and configuration
- `figures/`: Publication-ready figures
- `examples/`: Per-archetype example gene plots

## Figures

1. **fig_CS_scatter**: All genes in C-S space with quadrant boundaries
2. **fig_CS_marginals**: Histograms of C and S distributions
3. **fig_top_tables**: Top genes per archetype
4. **fig_archetype_examples**: Representative gene visualizations
