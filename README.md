# BioRSP: Biological Radar Spatial Profiling

BioRSP allows you to quantify how "spatially organized" a gene is without defining clusters first. It calculates two key metrics for every gene:

1.  **Coverage (C)**: How widely is the gene expressed?
2.  **Spatial Score (S)**: How polarized/directed is the expression pattern in the tissue embedding?

By combining these, BioRSP classifies genes into **Archetypes**:
-   **Housekeeping**: High Coverage, Low Spatial Score -> Ubiquitous
-   **Regional Program**: High Coverage, High Spatial Score -> Gradients/Zonation
-   **Niche Marker**: Low Coverage, High Spatial Score -> Focal/Patchy
-   **Sparse Noise**: Low Coverage, Low Spatial Score -> Random

## Installation

```bash
pip install .
```

## Quick Start

```python
from biorsp import score_genes, classify_genes
import scanpy as sc

# Load data
adata = sc.datasets.pbmc3k_processed()

# Score genes based on UMAP embedding
scores = score_genes(adata, adata.var_names, embedding_key="X_umap")
results = classify_genes(scores)

print(results.head())
```

## Documentation

-   [Start Here](docs/1_start_here/intro.md)
-   [Quickstart](docs/1_start_here/quickstart.md)
-   [Guides](docs/3_guides/run_custom_data.md)
-   [Benchmarks](analysis/benchmarks/README.md)

## Repository Structure

-   `biorsp/`: Core library code.
-   `analysis/`: Reproducible analysis workflows (Benchmarks, Kidney Atlas).
-   `dev/`: Developer tools and smoke tests.
-   `examples/`: Runnable scripts to learn the API.

## Citation

Please cite [Paper Reference] if you use BioRSP.
