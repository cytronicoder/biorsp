# Introduction

BioRSP analyzes gene-level spatial organization in a two-dimensional embedding. For each gene, it computes a coverage metric (fraction of cells expressing the gene) and a spatial organization metric derived from the radial distribution of foreground versus background cells. Archetype labels are assigned by applying coverage and spatial-score thresholds.

BioRSP does not perform clustering or trajectory inference. It relies on the embedding you provide (e.g., UMAP or spatial coordinates) and reports spatial organization within that embedding.

## Typical inputs

- A 2D embedding for cells (e.g., `adata.obsm["X_umap"]`).
- A gene expression matrix aligned to the embedding.

## Typical outputs

- Per-gene tables with coverage, spatial score, and related summaries.
- Optional archetype classification based on thresholds.
- Optional standardized plots summarizing scores and archetype composition.
