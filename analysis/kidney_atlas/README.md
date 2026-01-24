# Kidney atlas case studies

This directory contains scripts for applying BioRSP to the KPMP kidney atlas (AnnData `.h5ad`). The workflows are designed for users with minimal coding experience.

## Dataset assumptions

The scripts expect:

- `adata.obsm` to include a 2D embedding (e.g., `X_umap`).
- `adata.var_names` to contain gene identifiers (optionally `adata.var["feature_name"]` for display names).
- `adata.obs` to include disease metadata for stratified analysis. The disease runner searches common keys such as `disease_category`, `disease`, or `disease_state`.
- Optional cell-type labels (for filtering) in a user-specified column.

## Main workflows

### 1) All-genes archetype summary

Scores a large set of genes and produces a report and standardized plots.

```bash
python analysis/kidney_atlas/runners/run_kpmp_archetypes_all_genes.py \
  --h5ad /path/to/kpmp.h5ad \
  --outdir results/kpmp_all_genes
```

**Key outputs** (in `--outdir`):

- `runs_all_genes.csv`: per-gene scores.
- `classification.csv`: gene-to-archetype mapping.
- `derived_thresholds.json`: derived `C_cut`/`S_cut` details.
- `manifest.json`: configuration and provenance.
- `report.md`: summary report.
- `figures/`: standardized plots.
- `examples/`: example metadata for archetype panels.

### 2) TAL analysis

Runs BioRSP on a subset of genes and optionally computes gene–gene pairs.

```bash
python analysis/kidney_atlas/runners/run_tal_analysis.py \
  --ref_data /path/to/kpmp.h5ad \
  --outdir results/kpmp_tal \
  --controls "SLC12A1,UMOD,EGF"
```

**Key outputs** (in `--outdir`):

- `tal_gene_results.csv`: per-gene scores and archetypes.
- `tal_top_genes.txt`: ranked gene list.
- `tal_gene_pairs.csv` (optional): gene–gene scores when enabled.
- `plots/`: diagnostic and summary plots.

### 3) Disease-stratified analysis

Stratifies the dataset by disease labels and runs the analysis per group.

```bash
python analysis/kidney_atlas/runners/run_disease_stratified_analysis.py \
  --ref_data /path/to/kpmp.h5ad \
  --outdir results/kpmp_disease
```

To restrict to a cell type:

```bash
python analysis/kidney_atlas/runners/run_disease_stratified_analysis.py \
  --ref_data /path/to/kpmp.h5ad \
  --outdir results/kpmp_disease_tal \
  --celltype_key subclass.l1 \
  --celltype_filter TAL
```

**Key outputs** (per disease subdirectory):

- `gene_scores.csv`: per-gene scores and archetypes.
- `gene_pairs.csv` (optional): gene–gene scores.
- `manifest.json`: configuration and provenance.
- `figures/`: standardized plots.

## Troubleshooting

- **Missing disease column**: rename your `adata.obs` column to a recognized key or update the script configuration.
- **No embedding found**: confirm `adata.obsm` contains a 2D embedding (`X_umap`, `X_pca`, or another 2D array).
- **Empty plots**: check filters such as `--min-coverage` and `--min-nonzero`.
- **Label normalization**: ensure disease and cell-type labels use consistent capitalization and spacing.
