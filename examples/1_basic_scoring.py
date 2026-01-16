"""
Example 1: Basic Scoring Workflow
=================================

This example demonstrates how to run BioRSP on an AnnData object
to obtain Coverage, Spatial Bias Score, Directionality, and Archetype classifications.
"""

import numpy as np
import scanpy as sc

from biorsp import classify_genes, score_genes

print("Generating synthetic data...")
n_cells = 1000
n_genes = 50
adata = sc.AnnData(np.random.poisson(1, (n_cells, n_genes)))
adata.var_names = [f"Gene_{i}" for i in range(n_genes)]
adata.obsm["X_topological"] = np.random.randn(n_cells, 2)

print("Scoring genes...")
df = score_genes(
    adata,
    genes=adata.var_names[:10],
    embedding_key="X_topological",
    n_permutations=100,
)

print("Classifying genes...")
df = classify_genes(df, fdr_cut=0.05)

print("\nResults (Top 5):")
print(df[["Coverage", "Spatial_Bias_Score", "Directionality", "Archetype"]].head())
