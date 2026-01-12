# Running on Custom Data

BioRSP is designed to work with standard AnnData objects.

## Prerequisites

1.  **AnnData Object (`adata`)**
    - `adata.X`: Expression matrix (counts or log-normalized).
    - `adata.obsm`: Embedding coordinates (e.g., UMAP, t-SNE, Spatial).

2.  **Embedding**
    - You must have a 2D embedding key (e.g., `"X_umap"` or `"X_spatial"`).
    - BioRSP calculates spatial scores *relative to the center* of this embedding.

## Step-by-Step Guide

```python
import scanpy as sc
from biorsp import score_genes, classify_genes

# 1. Read your data
adata = sc.read_h5ad("path/to/my_data.h5ad")

# 2. Check for embedding
if "X_umap" not in adata.obsm:
    print("Computing UMAP...")
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)

# 3. Select genes to score
# Usually, we score highly variable genes or marker genes
sc.pp.highly_variable_genes(adata, n_top_genes=2000)
genes_to_score = adata.var_names[adata.var["highly_variable"]]

# 4. Run Scoring
df = score_genes(adata, genes_to_score, embedding_key="X_umap")

# 5. Classify
results = classify_genes(df)

# 6. Save results
results.to_csv("biorsp_results.csv")
```

## Common Pitfalls

-   **Wrong Embedding**: BioRSP assumes the input embedding reflects the spatial geometry you care about. If you use UMAP, it scores organization in UMAP space. If you use physical coordinates (`X_spatial`), it scores physical organization.
-   **Sparse Matrices**: BioRSP handles sparse matrices automatically.
-   **Normalization**: BioRSP infers whether data is raw counts or log-normalized to set detection thresholds. You can override this in `BioRSPConfig`.
