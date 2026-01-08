"""
BioRSP Quickstart Example
-------------------------
Demonstrates the new score_genes workflow for biomarker discovery.
"""

import argparse

import anndata
import numpy as np

from biorsp import BioRSPConfig, classify_genes, score_genes


def main():
    parser = argparse.ArgumentParser(description="BioRSP Quickstart Example")
    parser.parse_args()

    print("Generating synthetic data...")
    n_cells = 1000
    rng = np.random.default_rng(42)

    # Random points in a disk
    r = np.sqrt(rng.random(n_cells))
    theta = rng.uniform(-np.pi, np.pi, n_cells)
    coords = np.column_stack([r * np.cos(theta), r * np.sin(theta)])

    # Gene 1: "Niche Biomarker" (Sparse but highly localized)
    # High expression only in a specific wedge (theta ~ 0)
    wedge_mask = np.abs(theta) < 0.5
    gene_niche = np.zeros(n_cells)
    gene_niche[wedge_mask] = rng.poisson(5, size=np.sum(wedge_mask))
    # Add some noise
    gene_niche += rng.poisson(0.1, size=n_cells)

    # Gene 2: "Housekeeping" (High coverage, low spatial score)
    gene_house = rng.poisson(3, size=n_cells)  # Uniformly high

    # Gene 3: "Sparse/Noise"
    gene_noise = rng.poisson(0.1, size=n_cells)

    adata = anndata.AnnData(X=np.column_stack([gene_niche, gene_house, gene_noise]))
    adata.var_names = ["Niche_Marker", "Housekeeper", "Noise"]
    adata.obsm["X_umap"] = coords

    # 2. Run Scoring
    print("Running BioRSP score_genes...")
    # Using recommended defaults
    config = BioRSPConfig(
        delta_deg=60,
        B=72,
        empty_fg_policy="zero",
        # We assume counts, so threshold will auto-detect to 1.
    )

    df = score_genes(adata, list(adata.var_names), config=config)

    # 3. Classify into Archetypes
    # We use default cutoffs for demonstration
    df_classified = classify_genes(df)

    print("\nBioRSP Results Table:")
    print("---------------------")
    cols = ["gene", "coverage_expr", "spatial_score", "spatial_sign", "archetype"]
    print(df_classified[cols].to_string(index=False))

    print("\nInterpretation:")
    print("- Niche_Marker should have Low C, High S -> 'niche_biomarker'")
    print("- Housekeeper should have High C, Low S -> 'housekeeping_uniform'")
    print("- Noise should have Low C, Low S -> 'sparse_presence'")


if __name__ == "__main__":
    main()
