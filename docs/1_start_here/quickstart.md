# Quickstart

Get started with BioRSP in 5 minutes.

## Installation

```bash
pip install .
```

(Or from PyPI once released)

## Minimal Example

```python
import scanpy as sc
from biorsp import score_genes, classify_genes

# 1. Load Data
adata = sc.datasets.pbmc3k_processed()
# Ensure we have an embedding (e.g., UMAP)
if "X_umap" not in adata.obsm:
    sc.tl.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)

# 2. Score Genes
# This calculates Coverage (C) and Spatial Score (S)
genes = ["CD3D", "MS4A1", "CST3"]
scores = score_genes(adata, genes, embedding_key="X_umap")

# 3. Classify Archetypes
# Categorizes genes into: Housekeeping, Regional Program, Niche Marker, or Sparse Noise
results = classify_genes(scores)

print(results[["Coverage", "Spatial_Score", "Archetype"]])
```

## Understanding the Output

| Column | Description |
|--------|-------------|
| **Coverage** | Fraction of cells where the gene is expressed. (0.0 to 1.0) |
| **Spatial_Score** | Magnitude of directional bias. Higher = more spatially coherent. |
| **Directionality** | Signed value indicating orientation (if applicable). |
| **Archetype** | Biological classification of the pattern. |

## Next Steps

- Learn about the [Interpretation](interpretation.md) of scores.
- Run on [Custom Data](../3_guides/run_custom_data.md).
