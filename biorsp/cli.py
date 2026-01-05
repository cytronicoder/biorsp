"""
Command-line interface for BioRSP.

Commands:
- run: Compute RSP for a dataset.
- plot: Visualize results.
"""

import argparse
import sys
import warnings

import numpy as np

from .config import BioRSPConfig
from .core import assess_adequacy, compute_rsp_radar
from .foreground import define_foreground
from .geometry import compute_vantage, polar_coordinates
from .io import load_expression_matrix, load_spatial_coords, load_umi_counts, save_results
from .manifest import create_manifest, save_manifest
from .pairwise import compute_pairwise_relationships
from .preprocessing import normalize_radii
from .results import FeatureResult, RunSummary, assign_feature_types
from .robustness import compute_robustness_score
from .stats import bh_fdr, compute_p_value
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
    center = compute_vantage(coords, method="geometric_median")
    r, theta = polar_coordinates(coords, center)

    # Within-set robust radial normalization
    # This ensures that radial distances are comparable across different spatial scales.
    r, norm_stats = normalize_radii(r)

    feature_results = {}
    genes = df_expr.columns

    # Config
    config = BioRSPConfig(
        B=args.B,
        delta_deg=args.delta,
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
    )

    rng = np.random.default_rng(config.seed)

    umi_counts = None
    if args.inference:
        if args.umis:
            umi_counts = load_umi_counts(
                args.umis,
                n_cells=len(coords),
                column=args.umi_column,
            )
        else:
            warnings.warn(
                "Using expression row sums as UMI counts. For stratified inference, "
                "provide raw counts or use --umis.",
                RuntimeWarning,
                stacklevel=2,
            )
            umi_counts = df_expr.sum(axis=1).values

    print(f"Processing {len(genes)} genes...")

    for gene in tqdm(genes):
        x = df_expr[gene].values

        # 2. Foreground
        y, fg_info = define_foreground(
            x,
            mode=args.fg_mode,
            q=args.quantile,
            abs_threshold=args.abs_threshold,
            rng=rng,
            min_fg=args.min_fg_total,
        )
        coverage_abs = np.mean(x > args.t_detect)

        if y is None:
            # Create a dummy adequacy report for underpowered genes
            from .summaries import ScalarSummaries
            from .typing import AdequacyReport

            n_sectors = config.B
            adequacy = AdequacyReport(
                is_adequate=False,
                reason=fg_info["status"],
                counts_fg=np.zeros(n_sectors),
                counts_bg=np.zeros(n_sectors),
                sector_mask=np.zeros(n_sectors, dtype=bool),
                n_foreground=0.0,
                n_background=float(len(x)),
                adequacy_fraction=0.0,
                sector_indices=None,
            )
            feature_results[gene] = FeatureResult(
                feature=gene,
                threshold_quantile=fg_info.get("tau", 0.0),
                coverage_quantile=0.0,
                coverage_prevalence=float(coverage_abs),
                adequacy=adequacy,
                summaries=ScalarSummaries(
                    peak_distal=np.nan,
                    peak_distal_angle=np.nan,
                    peak_proximal=np.nan,
                    peak_proximal_angle=np.nan,
                    peak_extremal=np.nan,
                    peak_extremal_angle=np.nan,
                    anisotropy=np.nan,
                    max_rsp=np.nan,
                    min_rsp=np.nan,
                    integrated_rsp=np.nan,
                ),
                foreground_info=fg_info,
                sector_weight_mode=config.sector_weight_mode,
                sector_weight_k=config.sector_weight_k,
            )
            continue

        # 3. Adequacy
        adequacy = assess_adequacy(
            y,
            theta,
            config=config,
        )

        radar = compute_rsp_radar(
            r,
            theta,
            y,
            config=config,
            adequacy=adequacy,
        )

        summary = compute_scalar_summaries(radar)

        feature_results[gene] = FeatureResult(
            feature=gene,
            threshold_quantile=float(
                fg_info.get("tau", args.abs_threshold if args.fg_mode == "absolute" else 0.0)
            ),
            coverage_quantile=float(fg_info["realized_frac"]),
            coverage_prevalence=float(coverage_abs),
            adequacy=adequacy,
            summaries=summary,
            foreground_info=fg_info,
            radar=radar if args.store_rsp or args.pairwise else None,
            sector_weight_mode=config.sector_weight_mode,
            sector_weight_k=config.sector_weight_k,
        )

        if args.inference and adequacy.is_adequate:
            inf_res = compute_p_value(
                r,
                theta,
                y,
                config=config,
                n_perm=config.n_permutations,
                umi_counts=umi_counts,
                rng=rng,
                adequacy=adequacy,
                show_progress=False,
            )
            feature_results[gene].p_value = inf_res.p_value
            feature_results[gene].perm_mode = inf_res.perm_mode
            feature_results[gene].K_eff = inf_res.K_eff
            feature_results[gene].empty_sector_count = inf_res.empty_sector_count

    if args.inference:
        eligible = [fr for fr in feature_results.values() if fr.adequacy.is_adequate]
        p_values = np.array([fr.p_value for fr in eligible], dtype=float)
        q_values = bh_fdr(p_values)
        for fr, q_val in zip(eligible, q_values):
            fr.q_value = float(q_val) if np.isfinite(q_val) else np.nan

    typing_thresholds = None
    if args.typing:
        feature_results, typing_thresholds = assign_feature_types(
            feature_results,
            coverage_field="coverage_prevalence",
            method=args.typing_threshold_method,
            c_hi=args.c_hi,
            A_hi=args.a_hi,
        )

    if args.robustness:
        adequate_features = [fr for fr in feature_results.values() if fr.adequacy.is_adequate]
        ordered = sorted(adequate_features, key=lambda fr: fr.summaries.anisotropy, reverse=True)
        if args.robustness_top_k:
            ordered = ordered[: args.robustness_top_k]
        for fr in ordered:
            x = df_expr[fr.feature].values
            fr.robustness = compute_robustness_score(
                x,
                r,
                theta,
                B=config.n_angles,
                delta_deg=config.sector_width_deg,
                n_subsample=args.robustness_subsamples,
                subsample_frac=args.robustness_frac,
                seed=config.seed,
                min_fg_sector=config.min_fg_sector,
                min_bg_sector=config.min_bg_sector,
                quantile=args.quantile,
                fg_mode=args.fg_mode,
                abs_threshold=args.abs_threshold,
            )

    pairwise_results = None
    if args.pairwise:
        radar_by_feature = {
            name: fr.radar
            for name, fr in feature_results.items()
            if fr.adequacy.is_adequate and fr.radar is not None
        }
        synergy, complement = compute_pairwise_relationships(
            radar_by_feature, top_k=args.pairwise_top_k
        )
        pairwise_results = {
            "synergy": [result.__dict__ for result in synergy],
            "complementarity": [result.__dict__ for result in complement],
        }
        if args.pairwise_output:
            save_results(pairwise_results, args.pairwise_output)

    # Save results
    print(f"Saving results to {args.output}...")
    summary = RunSummary(
        typing_thresholds=typing_thresholds,
        pairwise=pairwise_results,
    )
    output = {
        "features": {
            name: {
                "threshold_quantile": fr.threshold_quantile,
                "coverage_quantile": fr.coverage_quantile,
                "coverage_prevalence": fr.coverage_prevalence,
                "is_adequate": fr.adequacy.is_adequate,
                "adequacy_reason": fr.adequacy.reason,
                "adequacy_fraction": fr.adequacy.adequacy_fraction,
                "n_fg_total": fr.adequacy.n_foreground,
                "n_bg_total": fr.adequacy.n_background,
                "anisotropy": fr.summaries.anisotropy,
                "max_rsp": fr.summaries.max_rsp,
                "min_rsp": fr.summaries.min_rsp,
                "integrated_rsp": fr.summaries.integrated_rsp,
                "peak_distal": fr.summaries.peak_distal,
                "peak_distal_angle": fr.summaries.peak_distal_angle,
                "peak_proximal": fr.summaries.peak_proximal,
                "peak_proximal_angle": fr.summaries.peak_proximal_angle,
                "peak_extremal": fr.summaries.peak_extremal,
                "peak_extremal_angle": fr.summaries.peak_extremal_angle,
                "feature_type": fr.feature_type,
                "p_value": fr.p_value,
                "q_value": fr.q_value,
                "perm_mode": fr.perm_mode,
                "K_eff": fr.K_eff,
                "empty_sector_count": fr.empty_sector_count,
                "sector_weight_mode": config.sector_weight_mode,
                "sector_weight_k": config.sector_weight_k,
                "mean_support_weight": (
                    float(np.nanmean(fr.radar.sector_weights))
                    if fr.radar is not None and fr.radar.sector_weights is not None
                    else 1.0
                ),
                "min_support_weight": (
                    float(np.nanmin(fr.radar.sector_weights))
                    if fr.radar is not None and fr.radar.sector_weights is not None
                    else 1.0
                ),
                "mean_profile_corr": fr.robustness.mean_correlation if fr.robustness else None,
                "cv_anisotropy": fr.robustness.cv_anisotropy if fr.robustness else None,
                "rsp_profile": (
                    fr.radar.rsp.tolist() if args.store_rsp and fr.radar is not None else None
                ),
                "rsp_centers": (
                    fr.radar.centers.tolist() if args.store_rsp and fr.radar is not None else None
                ),
            }
            for name, fr in feature_results.items()
        },
        "typing_thresholds": (
            summary.typing_thresholds.__dict__ if summary.typing_thresholds else None
        ),
        "pairwise": pairwise_results,
    }

    # Save manifest
    manifest = create_manifest(
        parameters=vars(args),
        seed=config.seed,
        extra_metadata={
            "n_genes": len(genes),
            "n_cells": len(coords),
            "radial_normalization": norm_stats,
        },
    )
    manifest_path = args.output + ".manifest.json"
    save_manifest(manifest, manifest_path)
    output["manifest_path"] = manifest_path
    save_results(output, args.output)
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
    run_parser.add_argument(
        "--fg-mode",
        choices=["quantile", "absolute", "auto"],
        default="quantile",
        help="Foreground selection mode",
    )
    run_parser.add_argument("--quantile", type=float, default=0.90, help="Foreground quantile")
    run_parser.add_argument(
        "--abs-threshold",
        type=float,
        help="Absolute threshold for foreground (expr >= T)",
    )
    run_parser.add_argument(
        "--t-detect",
        type=float,
        default=0.0,
        help="Detection threshold for prevalence coverage",
    )
    run_parser.add_argument(
        "--min-count", type=int, default=10, help="Min foreground cells per sector"
    )
    run_parser.add_argument(
        "--min-bg-count", type=int, default=50, help="Min background cells per sector"
    )
    run_parser.add_argument(
        "--min-fg-total", type=int, default=100, help="Min total foreground cells"
    )
    run_parser.add_argument(
        "--min-adequacy-fraction",
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
    run_parser.add_argument("--n-perm", type=int, default=200, help="Number of permutations")
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
        "--typing",
        action="store_true",
        default=True,
        help="Assign coverage × anisotropy types",
    )
    run_parser.add_argument(
        "--no-typing",
        action="store_false",
        dest="typing",
        help="Disable coverage × anisotropy typing",
    )
    run_parser.add_argument(
        "--typing-threshold-method",
        choices=["median", "user"],
        default="median",
        help="Threshold method for typing",
    )
    run_parser.add_argument("--c-hi", type=float, help="Coverage threshold for typing")
    run_parser.add_argument("--a-hi", type=float, help="Anisotropy threshold for typing")
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
    run_parser.add_argument(
        "--pairwise",
        action="store_true",
        help="Compute pairwise synergy/complementarity scores",
    )
    run_parser.add_argument(
        "--pairwise-top-k",
        type=int,
        help="Limit pairwise computation to top-K anisotropy features",
    )
    run_parser.add_argument(
        "--pairwise-output",
        help="Optional JSON output for pairwise results",
    )
    run_parser.add_argument(
        "--robustness",
        action="store_true",
        help="Compute robustness diagnostics via subsampling",
    )
    run_parser.add_argument(
        "--robustness-top-k",
        type=int,
        help="Limit robustness computation to top-K anisotropy features",
    )
    run_parser.add_argument(
        "--robustness-subsamples",
        type=int,
        default=20,
        help="Number of robustness subsamples",
    )
    run_parser.add_argument(
        "--robustness-frac",
        type=float,
        default=0.8,
        help="Subsample fraction for robustness",
    )
    run_parser.add_argument(
        "--store-rsp",
        action="store_true",
        help="Store full RSP profiles in output",
    )

    args = parser.parse_args(argv)

    if args.command == "run":
        run_analysis(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
