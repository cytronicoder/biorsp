"""
BioRSP Example: Gene-Gene Co-patterns
-------------------------------------
This script demonstrates settings for broader integration of co-expression patterns.
We use a wider sector width (delta=120) and a lower foreground quantile (q=0.50)
to capture large-scale spatial relationships between genes.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import biorsp


def main():
    parser = argparse.ArgumentParser(description="BioRSP Gene-Gene Co-patterns Example")
    biorsp.add_common_args(parser)

    # Override defaults for broader integration
    parser.set_defaults(delta_deg=120.0, B=36, q=0.50)

    args = parser.parse_args()

    # Generate synthetic data: 2000 cells in a 2D disk
    n_cells = 2000
    rng = np.random.default_rng(args.seed)
    r = np.sqrt(rng.random(n_cells))
    theta = 2 * np.pi * rng.random(n_cells)
    coords = np.column_stack([r * np.cos(theta), r * np.sin(theta)])

    # Create two genes with a shared broad spatial gradient (e.g., Left vs Right)
    # Gene A: Enriched on the right (theta near 0)
    prob_a = 0.2 + 0.6 * np.cos(theta / 2) ** 2
    gene_a = rng.binomial(20, prob_a)

    # Gene B: Also enriched on the right, but with more noise
    prob_b = 0.1 + 0.4 * np.cos(theta / 2) ** 2
    gene_b = rng.binomial(20, prob_b)

    # Gene C: Enriched on the top (theta near pi/2) - orthogonal pattern
    prob_c = 0.2 + 0.6 * np.sin(theta / 2) ** 2
    gene_c = rng.binomial(20, prob_c)

    df_expr = pd.DataFrame(
        {"Broad_Right_A": gene_a, "Broad_Right_B": gene_b, "Broad_Top_C": gene_c}
    )

    print("Running BioRSP with Gene-Gene Co-pattern settings...")
    print(f"Settings: B={args.B}, delta={args.delta_deg}, q={args.q}")

    config = biorsp.config_from_args(args)
    outdir = biorsp.ensure_outdir(args.outdir)

    summary = biorsp.run(coords=coords, expression=df_expr, config=config, outdir=str(outdir))

    # Inspect results
    results_df = summary.to_dataframe()
    print("\nResults Summary:")
    print(results_df[["feature", "anisotropy", "r_mean", "coverage_fg"]])

    # Compute pairwise relationships between radar profiles
    print("\nComputing pairwise relationships...")
    synergy, complementarity = biorsp.compute_pairwise_relationships(
        {name: res.radar for name, res in summary.feature_results.items() if res.radar is not None}
    )

    print("\nTop Synergistic Pairs (High Radar Correlation):")
    for p in synergy[:3]:
        print(
            f"  {p.feature_a} <-> {p.feature_b}: corr={p.correlation:.4f}, peak_dist={p.peak_distance:.2f} rad"
        )

    # Plot the phenotype map
    fig, ax = plt.subplots(figsize=(7, 6))
    biorsp.plot_phenotype_map(summary.feature_results, ax=ax, delta_deg=config.delta_deg)
    plt.tight_layout()
    plt.savefig(outdir / "gene_gene_phenotype_map.png")
    print(f"\nPhenotype map saved to {outdir}/gene_gene_phenotype_map.png")


if __name__ == "__main__":
    main()
