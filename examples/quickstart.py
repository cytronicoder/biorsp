"""
BioRSP Quickstart Example
-------------------------
This script demonstrates the canonical entry point for BioRSP.
"""

import argparse

import numpy as np
import pandas as pd

import biorsp


def main():
    parser = argparse.ArgumentParser(description="BioRSP Quickstart Example")
    biorsp.add_common_args(parser)
    args = parser.parse_args()

    # 1. Generate synthetic data
    # 1000 cells in a 2D disk
    n_cells = 1000
    rng = np.random.default_rng(args.seed)
    r = np.sqrt(rng.random(n_cells))
    theta = 2 * np.pi * rng.random(n_cells)
    coords = np.column_stack([r * np.cos(theta), r * np.sin(theta)])

    # Create a gene with directional enrichment (e.g., enriched at theta=0)
    # Probability of expression depends on angle
    prob = 0.1 + 0.4 * np.exp(-0.5 * (theta / 0.5) ** 2)
    gene_expr = rng.binomial(10, prob)

    # Create a control gene (uniform)
    control_expr = rng.poisson(2, n_cells)

    df_expr = pd.DataFrame({"Enriched_Gene": gene_expr, "Control_Gene": control_expr})

    # 2. Run BioRSP
    print("Running BioRSP...")
    config = biorsp.config_from_args(args)
    outdir = biorsp.ensure_outdir(args.outdir)

    summary = biorsp.run(coords=coords, expression=df_expr, config=config, outdir=str(outdir))

    # 3. Inspect results
    results_df = summary.to_dataframe()
    print("\nResults Summary:")
    print(results_df)

    # 4. Access specific results
    enriched_res = summary.feature_results["Enriched_Gene"]
    print(f"\nEnriched Gene Anisotropy: {enriched_res.summaries.anisotropy:.4f}")
    print(f"Enriched Gene P-value: {enriched_res.p_value}")

    # 5. Save manifest
    biorsp.save_run_manifest(
        outdir,
        config,
        dataset_summary={"n_cells": n_cells, "n_features": df_expr.shape[1]},
    )

    print(f"\nResults and manifest saved to {outdir}/")


if __name__ == "__main__":
    main()
