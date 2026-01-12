"""
Example 1: Basic Scoring Workflow
=================================

This example demonstrates how to run BioRSP on an AnnData object
to obtain Coverage, Spatial Score, Directionality, and Archetype classifications.
"""

import numpy as np
import scanpy as sc

from biorsp import classify_genes, score_genes

# 1. Create a synthetic AnnData object with spatial coordinates
print("Generating synthetic data...")
n_cells = 1000
n_genes = 50
adata = sc.AnnData(np.random.poisson(1, (n_cells, n_genes)))
adata.var_names = [f"Gene_{i}" for i in range(n_genes)]
# Add a spatial embedding (e.g., UMAP or spatial coordinates)
adata.obsm["X_topological"] = np.random.randn(n_cells, 2)

# 2. Run BioRSP Scoring
print("Scoring genes...")
# embedding_key points to your spatial/topological coordinates
df = score_genes(
    adata,
    genes=adata.var_names[:10],
    embedding_key="X_topological",
    n_permutations=100,  # Low for speed in example; use >=1000 for real analysis
)

# 3. Classify into Archetypes
print("Classifying genes...")
# thresholds can be auto-detected or manually set
df = classify_genes(df, fdr_cut=0.05)

# 4. Inspect Results
print("\nResults (Top 5):")
print(df[["Coverage", "Spatial_Bias_Score", "Directionality", "Archetype"]].head())

# The columns are standardized:
# - Coverage: Fraction of cells expressing the gene (biological)
# - Spatial_Bias_Score: Magnitude of spatial coherence (0 to 1)
# - Directionality: Direction of the gradient (-1 to 1) or NaN for non-directional
# - Archetype: Classification (e.g., "Basal", "Patchy", "Non-spatial")
