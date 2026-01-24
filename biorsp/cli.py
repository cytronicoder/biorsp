"""Command-line interface for BioRSP.

Commands:
- run: Compute RSP for a dataset.
"""

import argparse

from biorsp.api import BioRSPConfig
from biorsp.io.loaders import (
    align_inputs,
    load_expression_matrix,
    load_spatial_coords,
    load_umi_counts,
)
from biorsp.main import run
from biorsp.utils.constants import (
    B_DEFAULT,
    DELTA_DEG_DEFAULT,
    K_EXPLORATORY_DEFAULT,
    N_BG_MIN_DEFAULT,
    N_FG_MIN_DEFAULT,
    N_FG_TOT_MIN_DEFAULT,
)
from biorsp.utils.logging import setup_logging


def run_analysis(args):
    """Execute the analysis pipeline.

    Args:
        args: Parsed CLI arguments.
    """
    setup_logging()

    print(f"Loading expression from {args.expression}...")
    df_expr = load_expression_matrix(args.expression, transpose=args.transpose)

    print(f"Loading coordinates from {args.coords}...")
    coords_df = load_spatial_coords(args.coords)

    umi_counts = None
    if args.inference:
        if args.umis:
            print(f"Loading UMI counts from {args.umis}...")
            umi_counts = load_umi_counts(
                args.umis,
                n_cells=None,
                column=args.umi_column,
            )
        else:
            print("Warning: Using expression row sums as UMI counts for stratified inference.")
            umi_counts = df_expr.sum(axis=1).values

    print("Aligning inputs...")
    df_expr, coords_df, umi_counts, alignment_report = align_inputs(
        expr=df_expr,
        coords=coords_df,
        umi=umi_counts,
        how="inner",
        min_overlap=0.5,
        verbose=True,
    )

    coords = coords_df[["x", "y"]].values

    assert (
        df_expr.shape[0] == coords.shape[0]
    ), "Expression and coordinates size mismatch after alignment"
    assert df_expr.index.equals(
        coords_df.index
    ), "Expression and coordinates index mismatch after alignment"

    print(f"Final dataset: {df_expr.shape[0]} cells × {df_expr.shape[1]} genes")

    config = BioRSPConfig(
        B=args.B,
        delta_deg=args.delta,
        foreground_quantile=args.q,
        n_permutations=args.n_perm,
        perm_mode=args.perm_mode,
        n_r_bins=args.n_r_bins,
        n_theta_bins=args.n_theta_bins,
        umi_bins=args.n_umi_bins,
        min_stratum_size=args.min_stratum_size,
        min_fg_sector=args.min_count,
        min_bg_sector=args.min_bg_count,
        min_fg_total=args.min_fg_total,
        min_adequacy_fraction=args.min_adequacy_fraction,
        seed=args.seed,
        sector_weight_mode=args.sector_weight_mode,
        sector_weight_k=args.sector_weight_k,
    )

    summary = run(
        coords=coords,
        expression=df_expr,
        umi_counts=umi_counts,
        config=config,
        outdir=args.outdir if hasattr(args, "outdir") else None,
    )

    if args.output:
        results_df = summary.to_dataframe()
        results_df.to_csv(args.output)
        print(f"Results saved to {args.output}")

    print("Analysis complete.")


def main(argv=None):
    """Parse CLI arguments and dispatch subcommands.

    Args:
        argv: Optional list of CLI arguments.
    """
    parser = argparse.ArgumentParser(
        prog="biorsp",
        description="BioRSP: Radial Spatial Patterning",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    run_parser = subparsers.add_parser("run", help="Run RSP analysis")
    run_parser.add_argument("--expression", required=True, help="Expression matrix file")
    run_parser.add_argument("--coords", required=True, help="Spatial coordinates file")
    run_parser.add_argument("--output", required=True, help="Output CSV file")
    run_parser.add_argument("--outdir", help="Output directory for manifest and plots")
    run_parser.add_argument(
        "--transpose",
        action="store_true",
        help="Transpose expression matrix (cells x genes)",
    )

    run_parser.add_argument("--B", type=int, default=B_DEFAULT, help="Number of sectors")
    run_parser.add_argument(
        "--delta", type=float, default=DELTA_DEG_DEFAULT, help="Sector width (degrees)"
    )
    run_parser.add_argument("--q", type=float, default=0.90, help="Foreground quantile")
    run_parser.add_argument(
        "--min-count", type=int, default=N_FG_MIN_DEFAULT, help="Min foreground cells per sector"
    )
    run_parser.add_argument(
        "--min-bg-count", type=int, default=N_BG_MIN_DEFAULT, help="Min background cells per sector"
    )
    run_parser.add_argument(
        "--min-fg-total", type=int, default=N_FG_TOT_MIN_DEFAULT, help="Min total foreground cells"
    )
    run_parser.add_argument(
        "--min_adequacy_fraction",
        type=float,
        default=0.9,
        help="Min fraction of adequate sectors required",
    )
    run_parser.add_argument(
        "--umis",
        help="Optional CSV/TSV file with UMI counts per cell (required for stratification)",
    )
    run_parser.add_argument(
        "--umi-column",
        help="Column name to read UMI counts from the UMI file (defaults to umi/umis)",
    )
    run_parser.add_argument("--inference", action="store_true", help="Run permutation test")
    run_parser.add_argument(
        "--n-perm", type=int, default=K_EXPLORATORY_DEFAULT, help="Number of permutations"
    )
    run_parser.add_argument(
        "--perm-mode",
        choices=["radial", "joint", "rt_umi", "none"],
        default="radial",
        help="Permutation mode",
    )
    run_parser.add_argument("--n-r-bins", type=int, default=10, help="Number of radial bins")
    run_parser.add_argument("--n-theta-bins", type=int, default=4, help="Number of angular bins")
    run_parser.add_argument("--n-umi-bins", type=int, default=10, help="Number of UMI bins")
    run_parser.add_argument(
        "--min-stratum-size", type=int, default=50, help="Min cells per stratum"
    )
    run_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    run_parser.add_argument(
        "--sector-weight-mode",
        choices=["none", "sqrt_frac", "effective_min", "logistic_support"],
        default="none",
        help="Mode for support-based sector weighting",
    )
    run_parser.add_argument(
        "--sector-weight-k",
        type=float,
        default=5.0,
        help="Tunable parameter for sector weighting",
    )

    args = parser.parse_args(argv)

    if args.command == "run":
        run_analysis(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
