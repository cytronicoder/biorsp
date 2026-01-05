"""
BioRSP Example: Marker Discovery
--------------------------------
This script demonstrates the recommended settings for high-sensitivity marker discovery.
We use a narrow sector width (delta=60) and a high foreground quantile (q=0.90)
to detect localized spatial enrichment patterns while avoiding selection bias.
"""

import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import biorsp

# Set up logging to see why features might be abstained
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main():
    parser = argparse.ArgumentParser(description="BioRSP Marker Discovery Example")
    biorsp.add_common_args(parser)
    args = parser.parse_args()

    # Recommended defaults for Marker Discovery are now the package defaults:
    # B=72, delta_deg=60, q=0.90, empty_fg_policy="zero"

    # Generate synthetic data: 2000 cells in a 2D disk
    n_cells = 2000
    rng = np.random.default_rng(args.seed)
    r = np.sqrt(rng.random(n_cells))
    theta = 2 * np.pi * rng.random(n_cells)
    coords = np.column_stack([r * np.cos(theta), r * np.sin(theta)])

    # 1. A localized "Wedge" marker (expressed everywhere, but shifted in a 45-degree slice)
    # This demonstrates high localization (L_g) and low coverage_fg (if q is high)
    # or high localization and high coverage (if q is low).
    wedge_expr = rng.poisson(0.1, n_cells)
    wedge_mask = (theta > 0) & (theta < np.radians(45))
    # Shift: peripheral enrichment only in the wedge
    wedge_expr[wedge_mask & (r > 0.7)] += rng.poisson(15.0, np.sum(wedge_mask & (r > 0.7)))

    # 2. A global "Rim" marker (enriched at the periphery everywhere)
    # This demonstrates low localization (global) and high coverage.
    rim_expr = rng.poisson(0.1, n_cells)
    rim_expr[r > 0.7] += rng.poisson(10.0, np.sum(r > 0.7))

    # 3. A null marker (uniform noise)
    null_expr = rng.poisson(2.0, n_cells)

    df_expr = pd.DataFrame(
        {"Wedge_Marker": wedge_expr, "Rim_Marker": rim_expr, "Null_Marker": null_expr}
    )

    print("Running BioRSP with Marker Discovery defaults...")
    print(f"Settings: B={args.B}, delta={args.delta_deg}, q={args.q}")

    # We can override config to be more permissive for narrow markers
    config = biorsp.config_from_args(args)
    from dataclasses import replace

    config = replace(config, min_coverage=0.05)

    outdir = biorsp.ensure_outdir(args.outdir)

    summary = biorsp.run(coords=coords, expression=df_expr, config=config, outdir=str(outdir))

    # Inspect results
    results_df = summary.to_dataframe()
    print("\nResults Summary (including Coverage metrics):")
    # 'localization' in the dataframe comes from 'localization_entropy'
    cols = ["feature", "anisotropy", "localization", "coverage_bg", "coverage_fg", "p_value"]
    print(results_df[cols])

    # Plot the landscape
    fig, ax = plt.subplots(figsize=(6, 5))
    biorsp.plot_localization_scatter(summary.feature_results, ax=ax, delta_deg=config.delta_deg)
    plt.tight_layout()
    plt.savefig(outdir / "marker_discovery_landscape.png")
    print(f"\nLandscape plot saved to {outdir}/marker_discovery_landscape.png")


if __name__ == "__main__":
    main()
