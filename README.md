# BioRSP: Biological Radar Spatial Profiling

BioRSP is a framework for quantifying spatial gene expression patterns using radar-based geometric profiling. It decomposes gene expression into interpretable metrics: Coverage, Spatial Score, and Directionality, identifying distinct spatial archetypes (e.g., Basal, Patchy, Gradient).

## Key Features

- **Permutation-based Inference**: Robust statistical significance testing for spatial coherence.
- **Archetype Classification**: Automatically categorizes genes into spatial patterns.
- **Standardized Metrics**: 
  - **Coverage**: Fraction of cells expressing the gene.
  - **Spatial Score**: Magnitude of spatial organization (0-1).
  - **Directionality**: Direction of expression gradient.

## Quickstart

```bash
pip install biorsp-swordfish
```

```python
import scanpy as sc
from biorsp import score_genes, classify_genes

# Load your AnnData (must have spatial coordinates)
adata = sc.read_h5ad("my_data.h5ad")

# Score genes (uses 'X_umap' or 'spatial' by default)
df = score_genes(adata, genes=adata.var_names, embedding_key="X_umap")

# Classify into archetypes
df = classify_genes(df, fdr_cut=0.05)

# Inspect results
print(df[["Coverage", "Spatial_Score", "Archetype"]].head())
```

## Documentation

- **[Introduction](docs/1_start_here/intro.md)**: What is BioRSP?
- **[Quickstart Guide](docs/1_start_here/quickstart.md)**: Detailed getting started steps.
- **[Interpretation](docs/1_start_here/interpretation.md)**: Understanding Coverage vs Spatial Score.
- **[Concepts](docs/2_concepts/)**: Deep dive into geometry and scoring logic.
- **[Guides](docs/3_guides/)**: Tutorials for custom data and QC.

## Project Structure

- `biorsp/`: Core package source.
- `analysis/`: Analysis workflows and case studies (e.g., Kidney Atlas).
- `dev/`: Developer tools and smoke tests.
- `examples/`: Minimal example scripts.
