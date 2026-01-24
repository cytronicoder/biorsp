# Running on custom data

BioRSP expects an AnnData object with a two-dimensional embedding.

## Requirements

- `adata.X` with cells by genes (counts or normalized values).
- `adata.obsm` containing a 2D embedding (e.g., `X_umap`, `X_spatial`).

## Example workflow

```python
import scanpy as sc
from biorsp import score_genes, classify_genes

adata = sc.read_h5ad("path/to/data.h5ad")

if "X_umap" not in adata.obsm:
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)

genes = adata.var_names
scores = score_genes(adata, genes, embedding_key="X_umap")
results = classify_genes(scores)

results.to_csv("biorsp_results.csv", index=False)
```

## Common issues

- **Embedding choice**: BioRSP scores organization in the embedding you provide. Use the embedding that represents the spatial geometry you intend to analyze.
- **Gene filtering**: Large datasets may require a pre-filtered gene list for runtime reasons.
- **Threshold interpretation**: Archetype labels depend on thresholds; document how you set or derived them when reporting results.
