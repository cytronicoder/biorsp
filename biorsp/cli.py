"""
Command-line interface for BioRSP.

Commands:
- run: Compute RSP for a dataset.
- plot: Visualize results.
"""

import argparse
import sys

from .adequacy import gene_adequacy
from .config import BioRSPConfig
from .foreground import binary_foreground
from .geometry import geometric_median, polar_coordinates
from .inference import compute_p_value
from .io import load_expression_matrix, load_spatial_coords, save_results
from .manifest import create_manifest, save_manifest
from .radar import compute_rsp_radar
from .summaries import compute_scalar_summaries

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    # Fallback noop progress iterator when tqdm is not available
    def tqdm(iterable):
        """No-op progress iterator used when tqdm is not installed."""
        return iterable


def run_analysis(args):
    """Execute the analysis pipeline."""
    print(f"Loading expression from {args.expression}...")
    df_expr = load_expression_matrix(args.expression, transpose=args.transpose)

    print(f"Loading coordinates from {args.coords}...")
    coords = load_spatial_coords(args.coords)

    if len(df_expr) != len(coords):
        print(
            "Error: Number of cells in expression and coords mismatch: "
            f"{len(df_expr)} != {len(coords)}"
        )
        sys.exit(1)

    # 1. Geometry
    print("Computing geometric median and polar coordinates...")
    center, _, _ = geometric_median(coords)
    _, theta = polar_coordinates(coords, center)

    results = {}
    genes = df_expr.columns

    # Config
    config = BioRSPConfig(
        n_angles=args.B,
        sector_width_deg=args.delta,
        n_permutations=args.n_perm,
        min_fg_sector=args.min_count,
        seed=args.seed,
    )

    print(f"Processing {len(genes)} genes...")

    for gene in tqdm(genes):
        x = df_expr[gene].values

        # 2. Foreground
        y, threshold, coverage = binary_foreground(x, quantile=args.quantile)

        # 3. Adequacy
        adequacy = gene_adequacy(y, theta, config.n_angles, min_count=args.min_count)

        gene_res = {
            "coverage": coverage,
            "threshold": threshold,
            "is_adequate": adequacy.is_adequate,
            "adequacy_reason": adequacy.reason,
        }

        if adequacy.is_adequate:
            # 4. Radar
            theta_fg = theta[y]
            radar = compute_rsp_radar(
                theta_fg,
                B=config.n_angles,
                delta_deg=config.sector_width_deg,
            )

            # 5. Summaries
            summary = compute_scalar_summaries(radar)

            gene_res.update(
                {
                    "max_rsp": summary.max_rsp,
                    "mean_abs_rsp": summary.mean_abs_rsp,
                    "peak_angle": summary.peak_angle,
                    "integrated_rsp": summary.integrated_rsp,
                }
            )

            # 6. Inference (optional)
            if args.inference:
                p_val, _ = compute_p_value(
                    summary.mean_abs_rsp,
                    theta,
                    y,
                    B=config.n_angles,
                    delta_deg=config.sector_width_deg,
                    n_perm=config.n_permutations,
                    seed=config.seed,
                )
                gene_res["p_value"] = p_val

            # Optionally store full RSP profile (disabled by default)
            # gene_res["rsp_profile"] = radar.rsp.tolist()

        results[gene] = gene_res

    # Save results
    print(f"Saving results to {args.output}...")
    save_results(results, args.output)

    # Save manifest
    manifest = create_manifest(
        parameters=vars(args),
        seed=config.seed,
        extra_metadata={
            "n_genes": len(genes),
            "n_cells": len(coords),
        },
    )
    save_manifest(manifest, args.output + ".manifest.json")
    print("Done.")


def main(argv=None):
    """Parse CLI arguments and dispatch subcommands."""
    parser = argparse.ArgumentParser(
        prog="biorsp",
        description="BioRSP: Radial Spatial Patterning",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run RSP analysis")
    run_parser.add_argument("--expression", required=True, help="Expression matrix file")
    run_parser.add_argument("--coords", required=True, help="Spatial coordinates file")
    run_parser.add_argument("--output", required=True, help="Output JSON file")
    run_parser.add_argument(
        "--transpose",
        action="store_true",
        help="Transpose expression matrix (cells x genes)",
    )

    # Parameters
    run_parser.add_argument("--B", type=int, default=360, help="Number of sectors")
    run_parser.add_argument("--delta", type=float, default=20.0, help="Sector width (degrees)")
    run_parser.add_argument("--quantile", type=float, default=0.90, help="Foreground quantile")
    run_parser.add_argument(
        "--min-count", type=int, default=10, help="Min foreground cells per sector"
    )
    run_parser.add_argument("--inference", action="store_true", help="Run permutation test")
    run_parser.add_argument("--n-perm", type=int, default=1000, help="Number of permutations")
    run_parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args(argv)

    if args.command == "run":
        run_analysis(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
