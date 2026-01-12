"""
Example 2: Advanced Configuration
=================================

This example demonstrates how to customize BioRSP parameters using BioRSPConfig,
including defining the foreground, adjusting permutation settings, and
handling edge cases.
"""

import numpy as np
import scanpy as sc

from biorsp import BioRSPConfig, classify_genes, score_genes

# 1. Setup Data
np.random.seed(42)  # For data generation
n_cells = 500
adata = sc.AnnData(np.random.poisson(1, (n_cells, 20)))
adata.var_names = [f"Gene_{i}" for i in range(20)]
adata.obsm["X_umap"] = np.random.randn(n_cells, 2)
# Add cell type metadata for subsetting
adata.obs["cell_type"] = np.random.choice(["TypeA", "TypeB"], n_cells)

# 2. Configure Analysis
# See docs/2_concepts/inference.md for parameter details
config = BioRSPConfig(
    n_permutations=500,  # Custom permutation count
    seed=42,  # Reproducibility
    min_fg_total=10,  # Ignore genes with fewer than 10 foreground cells
    delta_deg=30.0,  # Wider sectors (default 20)
    foreground_quantile=0.95,  # Stricter foreground definition
)

print("Running with customized config...")

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

# 4. Classify with custom FDR
classified = classify_genes(
    results,
    fdr_cut=0.01,
    c_cut=0.15,  # Stricter FDR control  # Higher coverage threshold
)

print("\nResults for TypeA cells:")
print(classified[["Coverage", "Spatial_Bias_Score", "Directionality", "Archetype"]])
