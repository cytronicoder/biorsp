# UMAP vs densMAP Demo

This repository provides a small, self-contained demonstration of why standard UMAP can distort local densities in single-cell embeddings and how densMAP attempts to preserve them. Density preservation matters because crowded regions can hide rare states while sparse regions may be over-emphasized.

## What the script does
1. **Shape, density, distribution** – loads a publicly hosted `.h5ad` single-cell dataset, optionally subsamples, and summarizes library size, detected genes, and local density in PCA space.
2. **Embeddings** – computes standard UMAP and density-aware densMAP for a baseline parameter pair and for a small grid of `n_neighbors` and `min_dist` values. Diagnostic plots (scatter, density histograms, HD–LD correlation) are saved for each setting.
3. **Parameter impacts** – evaluates trustworthiness and 2D density statistics across the grid, compiling a summary CSV and a dashboard figure.

## How to run
```bash
python experiments/umap_vs_densmap/toy_umap_densmap_demo.py
```
Outputs are written under `experiments/umap_vs_densmap/` with figures in the `figures/` subdirectory and a metrics CSV at the root of that folder.

## What to inspect
Look at the side-by-side scatter plots for UMAP vs densMAP, the density histograms to see how densMAP spreads dense clusters, the trustworthiness chart to compare neighborhood preservation, and the HD–LD density correlation plots to check whether densMAP better reflects high-dimensional densities.
