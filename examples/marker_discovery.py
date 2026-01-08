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

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main():
    parser = argparse.ArgumentParser(description="BioRSP Marker Discovery Example")
    biorsp.add_common_args(parser)
    args = parser.parse_args()

    n_cells = 2000
    rng = np.random.default_rng(args.seed)
    r = np.sqrt(rng.random(n_cells))
    theta = 2 * np.pi * rng.random(n_cells)
    coords = np.column_stack([r * np.cos(theta), r * np.sin(theta)])

    wedge_expr = rng.poisson(0.1, n_cells)
    wedge_mask = (theta > 0) & (theta < np.radians(45))

    wedge_expr[wedge_mask & (r > 0.7)] += rng.poisson(15.0, np.sum(wedge_mask & (r > 0.7)))

    rim_expr = rng.poisson(0.1, n_cells)
    rim_expr[r > 0.7] += rng.poisson(10.0, np.sum(r > 0.7))

    null_expr = rng.poisson(2.0, n_cells)

    df_expr = pd.DataFrame(
        {"Wedge_Marker": wedge_expr, "Rim_Marker": rim_expr, "Null_Marker": null_expr}
    )

    print("Running BioRSP with Marker Discovery defaults...")
    print(f"Settings: B={args.B}, delta={args.delta_deg}, q={args.q}")

    config = biorsp.config_from_args(args)
    from dataclasses import replace

    config = replace(config, min_coverage=0.05)

    outdir = biorsp.ensure_outdir(args.outdir)

    summary = biorsp.run(coords=coords, expression=df_expr, config=config, outdir=str(outdir))

    results_df = summary.to_dataframe()
    print("\nResults Summary (including Coverage metrics):")

    cols = ["feature", "anisotropy", "localization", "coverage_bg", "coverage_fg", "p_value"]
    print(results_df[cols])

    fig, ax = plt.subplots(figsize=(6, 5))
    biorsp.plot_localization_scatter(summary.feature_results, ax=ax, delta_deg=config.delta_deg)
    plt.tight_layout()
    plt.savefig(outdir / "marker_discovery_landscape.png")
    print(f"\nLandscape plot saved to {outdir}/marker_discovery_landscape.png")


if __name__ == "__main__":
    main()
