# Quickstart Guide

## Installation

```bash
pip install biorsp-swordfish
```

## Basic Workflow

1. **Prepare Data**: Load your spatial transcriptomics data into an `AnnData` object. Ensure you have spatial coordinates in `adata.obsm` (e.g., `X_spatial` or `X_umap` for topological analysis).
2. **Define ROI (Optional)**: If analyzing a specific tissue structure, subset your `adata` first.
3. **Run Scoring**: Use `biorsp.score_genes()` to compute metrics.
4. **Classify**: Use `biorsp.classify_genes()` to assign archetypes.

## Example

See `examples/1_basic_scoring.py` for a runnable script.

```python
import biorsp
# ... load data ...
results = biorsp.score_genes(adata, genes=["GeneA", "GeneB"])
```
