"""
Example 2: Advanced Configuration
=================================

This example demonstrates how to customize BioRSP parameters using BioRSPConfig,
including defining the foreground, adjusting permutation settings, and
handling edge cases.
"""

import numpy as np
import scanpy as sc

from biorsp import BioRSPConfig, score_genes

# 1. Setup Data
n_cells = 500
adata = sc.AnnData(np.random.poisson(1, (n_cells, 20)))
adata.var_names = [f"Gene_{i}" for i in range(20)]
adata.obsm["X_umap"] = np.random.randn(n_cells, 2)
# Add cell type metadata for subsetting
adata.obs["cell_type"] = np.random.choice(["TypeA", "TypeB"], n_cells)

# 2. Configure Analysis
# See docs/2_concepts/inference.md for parameter details
config = BioRSPConfig(
    fdr_level=0.01,  # Stricter FDR control
    n_permutations=500,  # Custom permutation count
    seed=42,  # Reproducibility
    min_cells_detection=10,  # Ignore genes with fewer than 10 cells
    score_method="vantage",  # Use vantage point scoring (default)
)

print(f"Running with config: {config}")

# 3. Run Analysis on a Subset
# We only score cells labeled "TypeA"
subset_genes = ["Gene_0", "Gene_1", "Gene_2"]
results = score_genes(
    adata,
    genes=subset_genes,
    embedding_key="X_umap",
    subset={"cell_type": "TypeA"},  # Filter to specific cell type
    config=config,
)

print("\nResults for TypeA cells:")
print(results[["Coverage", "Spatial_Score", "Archetype"]])
