# Quickstart

This quickstart uses a small example dataset from Scanpy and the Python API.

## Installation

```bash
pip install .
```

## Minimal example

```python
import scanpy as sc
from biorsp import score_genes, classify_genes

adata = sc.datasets.pbmc3k_processed()

scores = score_genes(adata, adata.var_names, embedding_key="X_umap")
results = classify_genes(scores)

print(results[["Coverage", "Spatial_Bias_Score", "Directionality", "Archetype"]].head())
```

## Output notes

- `Coverage`: fraction of cells above the foreground threshold.
- `Spatial_Bias_Score`: spatial organization summary (non-negative).
- `Directionality`: signed summary derived from radial statistics.
- `Archetype`: label derived from coverage and spatial-score thresholds.

See `docs/1_start_here/interpretation.md` for guidance on interpreting these fields.
