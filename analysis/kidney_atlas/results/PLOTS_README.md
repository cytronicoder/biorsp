# KPMP embedding plots (example output)

This directory contains example UMAP embedding plots generated from a KPMP AnnData file. The figures and metadata reflect a specific run and are provided as reference outputs.

## Contents

- `embedding_plots/`: embedding plots colored by available metadata.
- `embedding_plots_qc/`: QC overlays (e.g., gene counts, mitochondrial percentage) when present.
- `embedding_plots_faceted/`: faceted plots (e.g., by disease category) when requested.

## How plots are generated

The plots are produced by `analysis/kidney_atlas/utils/plot_kpmp_embedding.py`. The script searches `adata.obs` for categorical fields and supports explicit `--color-by` and `--facet-by` arguments.

## Reproducibility

Each plot directory contains a `metadata.json` file with the embedding key, plotting parameters, and discovered columns. Use the metadata to regenerate plots with identical settings.
