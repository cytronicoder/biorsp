# KPMP All-Gene Archetypes Report

This report is an example output from a specific run. Values, thresholds, and figure filenames reflect the configuration used at that time.

**Generated:** 2026-01-11 04:45:12

## Dataset Summary

- **Cells analyzed:** 225,177
- **Embedding used:** `X_umap`
- **Subset query:** None
- **Total genes in dataset:** 31,332
- **Genes passing filters:** 16,459

## Gene Filtering

| Criterion | Threshold | Passed |
|-----------|-----------|--------|
| Coverage | ≥ 0.005 | 16,459 |
| Nonzero cells | ≥ 50 | 24,775 |
| **Both** | - | **16,459** |

## Threshold Selection

### Coverage Cutoff (c_cut)
- **Value used:** 0.0598
- **Method:** auto_derived

### Spatial Bias Score Cutoff (s_cut)
- **Value used:** 0.2121
- **Method:** distribution_derived

## Archetype Classification

| Archetype | Count | Percentage |
|-----------|-------|------------|
| Ubiquitous Uniform | 7,975 | 48.5% |
| Focal Marker | 356 | 2.2% |
| Regional Gradient | 255 | 1.5% |
| Rare Scattered | 7,873 | 47.8% |

**Total classified:** 16,459

## Top Genes per Archetype

### Ubiquitous Uniform (n=7,975)

| Gene | Coverage (C) | Spatial (S) | Sign |
|------|--------------|-------------|------|
| ENSG00000198712 | 0.9995 | 0.0973 | 1 |
| ENSG00000198899 | 0.9994 | 0.0972 | 1 |
| ENSG00000198804 | 0.9994 | 0.0880 | 1 |
| ENSG00000198840 | 0.9993 | 0.0797 | 1 |
| ENSG00000198938 | 0.9990 | 0.0955 | 1 |
| ENSG00000198886 | 0.9990 | 0.0701 | 1 |
| ENSG00000198727 | 0.9981 | 0.0883 | 1 |
| ENSG00000198763 | 0.9967 | 0.0816 | 1 |
| ENSG00000198888 | 0.9935 | 0.1091 | 1 |
| ENSG00000198786 | 0.9829 | 0.0612 | 1 |

### Focal Marker (n=356)

| Gene | Coverage (C) | Spatial (S) | Sign |
|------|--------------|-------------|------|
| ENSG00000102243 | 0.0080 | 0.3339 | 1 |
| ENSG00000227338 | 0.0195 | 0.3054 | 1 |
| ENSG00000179869 | 0.0098 | 0.2976 | 1 |
| ENSG00000254024 | 0.0100 | 0.2905 | 1 |
| ENSG00000196542 | 0.0067 | 0.2904 | 1 |
| ENSG00000198574 | 0.0058 | 0.2799 | -1 |
| ENSG00000243081 | 0.0339 | 0.2795 | 1 |
| ENSG00000164853 | 0.0097 | 0.2760 | 1 |
| ENSG00000153012 | 0.0154 | 0.2759 | 1 |
| ENSG00000110975 | 0.0219 | 0.2753 | 1 |

### Regional Gradient (n=255)

| Gene | Coverage (C) | Spatial (S) | Sign |
|------|--------------|-------------|------|
| ENSG00000170927 | 0.3611 | 0.2989 | 1 |
| ENSG00000196154 | 0.3383 | 0.2908 | 1 |
| ENSG00000169442 | 0.1129 | 0.2882 | -1 |
| ENSG00000205542 | 0.6975 | 0.2854 | -1 |
| ENSG00000166710 | 0.8835 | 0.2828 | -1 |
| ENSG00000184292 | 0.2874 | 0.2755 | 1 |
| ENSG00000165215 | 0.2238 | 0.2734 | 1 |
| ENSG00000196482 | 0.3320 | 0.2731 | 1 |
| ENSG00000142669 | 0.4601 | 0.2696 | -1 |
| ENSG00000083307 | 0.0852 | 0.2695 | 1 |

### Rare Scattered (n=7,873)

| Gene | Coverage (C) | Spatial (S) | Sign |
|------|--------------|-------------|------|
| ENSG00000167705 | 0.0597 | 0.0965 | -1 |
| ENSG00000253293 | 0.0597 | 0.1693 | 1 |
| ENSG00000103245 | 0.0597 | 0.0841 | 1 |
| ENSG00000117153 | 0.0597 | 0.0917 | 1 |
| ENSG00000170485 | 0.0597 | 0.1781 | 1 |
| ENSG00000162222 | 0.0597 | 0.0485 | 1 |
| ENSG00000223749 | 0.0596 | 0.1311 | 1 |
| ENSG00000115414 | 0.0596 | 0.1301 | 1 |
| ENSG00000166387 | 0.0596 | 0.1443 | 1 |
| ENSG00000160808 | 0.0596 | 0.1756 | 1 |

## Reliability Checks

### Subsample Stability (70% splits)
- **Spearman correlation:** 0.943
- **Median |ΔS|:** 0.0017
- **Genes tested:** 500

## Output Files

- `runs_all_genes.csv`: Complete per-gene results
- `classification.csv`: Gene-to-archetype mapping
- `derived_thresholds.json`: Threshold derivation details
- `manifest.json`: Run provenance and configuration
- `figures/`: Publication-ready figures
- `examples/`: Per-archetype example gene plots

## Figures

1. **fig_cs_scatter**: All genes in C vs S space with quadrant boundaries
2. **fig_CS_marginals**: Histograms of C and S distributions
3. **fig_top_tables**: Top genes per archetype
4. **fig_archetype_examples**: Representative gene visualizations
