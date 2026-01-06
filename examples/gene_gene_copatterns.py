"""
BioRSP Gene-Gene Relationships Example
--------------------------------------
Demonstrates score_gene_pairs for discovering spatial co-patterns.
"""

import anndata
import numpy as np

from biorsp import BioRSPConfig, score_gene_pairs


def main():
    print("Generating synthetic data with co-patterning...")
    n_cells = 1000
    rng = np.random.default_rng(99)
    r = np.sqrt(rng.random(n_cells))
    theta = rng.uniform(-np.pi, np.pi, n_cells)
    coords = np.column_stack([r * np.cos(theta), r * np.sin(theta)])

    # Gene A and B: Co-localized in wedge theta > 0
    mask_pos = theta > 0
    gA = rng.poisson(0.1, n_cells)
    gA[mask_pos] += rng.poisson(3, np.sum(mask_pos))

    gB = rng.poisson(0.1, n_cells)
    gB[mask_pos] += rng.poisson(3, np.sum(mask_pos))

    # Gene C: Opposite localized (theta < 0)
    mask_neg = theta < 0
    gC = rng.poisson(0.1, n_cells)
    gC[mask_neg] += rng.poisson(3, np.sum(mask_neg))

    adata = anndata.AnnData(X=np.column_stack([gA, gB, gC]))
    adata.var_names = ["GeneA", "GeneB", "GeneC"]
    adata.obsm["X_umap"] = coords

    # Run Pairwise Scoring
    print("Running score_gene_pairs...")
    config = BioRSPConfig(delta_deg=60, B=72, empty_fg_policy="zero")

    df_pairs = score_gene_pairs(adata, list(adata.var_names), config=config)

    print("\nPairwise Results:")
    print(df_pairs.to_string(index=False))

    print("\nExpectation:")
    print("- GeneA-GeneB: High copattern score, similarity_sign +1")
    print("- GeneA-GeneC: High copattern score magnitude, similarity_sign -1 (if opposite)")


if __name__ == "__main__":
    main()
