"""
BioRSP Quickstart Example
-------------------------
This script demonstrates the canonical entry point for BioRSP.
"""

import numpy as np
import pandas as pd

import biorsp


def main():
    # 1. Generate synthetic data
    # 1000 cells in a 2D disk
    n_cells = 1000
    r = np.sqrt(np.random.rand(n_cells))
    theta = 2 * np.pi * np.random.rand(n_cells)
    coords = np.column_stack([r * np.cos(theta), r * np.sin(theta)])

    # Create a gene with directional enrichment (e.g., enriched at theta=0)
    # Probability of expression depends on angle
    prob = 0.1 + 0.4 * np.exp(-0.5 * (theta / 0.5) ** 2)
    gene_expr = np.random.binomial(10, prob)

    # Create a control gene (uniform)
    control_expr = np.random.poisson(2, n_cells)

    df_expr = pd.DataFrame({"Enriched_Gene": gene_expr, "Control_Gene": control_expr})

    # 2. Run BioRSP
    print("Running BioRSP...")
    outdir = "example_results"
    summary = biorsp.run(coords=coords, expression=df_expr, outdir=outdir)

    # 3. Inspect results
    results_df = summary.to_dataframe()
    print("\nResults Summary:")
    print(results_df)

    # 4. Access specific results
    enriched_res = summary.feature_results["Enriched_Gene"]
    print(f"\nEnriched Gene Anisotropy: {enriched_res.summaries.anisotropy:.4f}")
    print(f"Enriched Gene P-value: {enriched_res.p_value}")

    print(f"\nResults and manifest saved to {outdir}/")


if __name__ == "__main__":
    main()
