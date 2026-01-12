"""
Robustness Benchmark for BioRSP Methods Paper.

Evaluates the stability of BioRSP metrics (Spatial Score S, Coverage C) under:

INVARIANCES (expected to be robust):
1. Rotation - Should not affect radial statistics
2. Jitter - Small positional noise
3. Subsampling - Downsampling cells

SENSITIVITIES (expected failure modes):
4. Anisotropic Scaling - Stretching breaks radial symmetry
5. Swirl - Radial warping creates artifactual radial shifts

Uses PAIRED evaluation: same seed generates baseline and distorted data,
then computes within-replicate deltas for proper statistical comparison.

Outputs: runs.csv, summary.csv, report.md, manifest.json, robustness curves
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from biorsp import BioRSPConfig

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

INVARIANCE_DISTORTIONS = {"none", "rotate", "jitter", "subsample"}


def run_robustness_pair(config_dict: dict, seed: int, config: BioRSPConfig) -> dict:
    """Run paired robustness evaluation: baseline and distorted with same seed."""
    from biorsp.simulations import (
        datasets,
        distortions,
        expression,
        rng,
        scoring,
        shapes,
    )

    shape = config_dict["shape"]
    N = config_dict["N"]
    pattern = config_dict["pattern"]
    distortion_kind = config_dict["distortion_kind"]
    distortion_strength = config_dict["distortion_strength"]

    condition_key = rng.condition_key(shape, N, pattern, distortion_kind, distortion_strength)
    gen = rng.make_rng(seed, "robustness", condition_key)

    coords_base, shape_meta = shapes.generate_coords(shape, N, gen)

    libsize = expression.simulate_library_size(
        N, gen, model="lognormal", params={"mean": 2000, "std": 0.5}
    )

    field = expression.generate_signal_field(coords_base, pattern, params={})

    counts = expression.generate_expression_from_field(
        field, libsize, gen, expr_model="nb", params={"phi": 10.0, "abundance": 1e-3}
    )

    adata_base = datasets.package_as_anndata(
        coords_base,
        counts[:, None],
        var_names=[f"{pattern}_gene"],
        obs_meta=None,
        embedding_key="X_sim",
    )

    t0 = time.time()
    results_base = scoring.score_dataset(adata_base, genes=[f"{pattern}_gene"], config=config)

    if len(results_base) == 0:
        baseline_s = np.nan
        baseline_c = np.nan
        baseline_abstain = True
    else:
        row_base = results_base.iloc[0]
        baseline_s = row_base["Spatial_Score"]
        baseline_c = row_base["Coverage"]
        baseline_abstain = row_base["abstain_flag"]

    if distortion_kind == "none" or distortion_strength == 0.0:
        distorted_s = baseline_s
        distorted_c = baseline_c
        distorted_abstain = baseline_abstain
    else:
        gen_dist = rng.make_rng(seed + 100000, "robustness_dist", condition_key)

        coords_dist, dist_meta = distortions.apply_distortion(
            coords_base, distortion_kind, distortion_strength, gen_dist, params={}
        )

        if distortion_kind == "subsample" and distortion_strength < 1.0:
            n_keep = int(N * distortion_strength)
            indices = gen_dist.choice(N, n_keep, replace=False)
            coords_dist = coords_base[indices]
            counts_dist = counts[indices]
        else:
            counts_dist = counts

        adata_dist = datasets.package_as_anndata(
            coords_dist,
            counts_dist[:, None],
            var_names=[f"{pattern}_gene"],
            obs_meta=None,
            embedding_key="X_sim",
        )

        results_dist = scoring.score_dataset(adata_dist, genes=[f"{pattern}_gene"], config=config)

        if len(results_dist) == 0:
            distorted_s = np.nan
            distorted_c = np.nan
            distorted_abstain = True
        else:
            row_dist = results_dist.iloc[0]
            distorted_s = row_dist["Spatial_Score"]
            distorted_c = row_dist["Coverage"]
            distorted_abstain = row_dist["abstain_flag"]

    elapsed = time.time() - t0

    delta_s = (
        distorted_s - baseline_s if not (np.isnan(baseline_s) or np.isnan(distorted_s)) else np.nan
    )

    return {
        "shape": shape,
        "N": N,
        "pattern": pattern,
        "distortion_kind": distortion_kind,
        "distortion_strength": distortion_strength,
        "baseline_spatial_score": baseline_s,
        "baseline_coverage": baseline_c,
        "baseline_abstain": baseline_abstain,
        "distorted_spatial_score": distorted_s,
        "distorted_coverage": distorted_c,
        "distorted_abstain": distorted_abstain,
        "delta_s": delta_s,
        "abs_delta_s": abs(delta_s) if not np.isnan(delta_s) else np.nan,
        "time": elapsed,
    }


def main():
    from biorsp.simulations import (
        checkpoint,
        docs,
        io,
        plotting,
        sweeps,
        validation,
    )

    parser = argparse.ArgumentParser(description="Robustness benchmark")
    parser.add_argument("--outdir", type=str, default=str(ROOT / "outputs" / "robustness"))
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--n_reps", type=int, default=50)
    parser.add_argument("--N", type=int, nargs="+", default=[2000])
    parser.add_argument("--shape", type=str, nargs="+", default=["disk"])
    parser.add_argument("--pattern", type=str, nargs="+", default=["wedge"])
    parser.add_argument(
        "--distortion_kind",
        type=str,
        nargs="+",
        default=["none", "rotate", "aniso_scale", "jitter", "subsample", "swirl"],
    )
    parser.add_argument("--n_permutations", type=int, default=250)
    parser.add_argument(
        "--mode", type=str, choices=["quick", "validation", "publication"], default="quick"
    )
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--n_workers", type=int, default=-1, help="Number of parallel workers")
    parser.add_argument(
        "--checkpoint_every", type=int, default=25, help="Save checkpoint every N runs"
    )
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument(
        "--permutation_scope",
        type=str,
        choices=["none", "all"],
        default="none",
        help="Permutation strategy: 'none' (no p-values, faster), 'all' (compute p-values)",
    )
    args = parser.parse_args()

    if args.n_workers != -1:
        args.n_jobs = args.n_workers

    strength_map = {
        "none": [0.0],
        "rotate": [0, 15, 45, 90, 180],
        "aniso_scale": [1.0, 1.2, 1.5, 2.0, 3.0],
        "jitter": [0.0, 0.02, 0.05, 0.1, 0.2],
        "subsample": [1.0, 0.8, 0.5, 0.3, 0.2],
        "swirl": [0, 1, 2, 5],
    }

    if args.mode == "quick":
        args.n_reps = 5
        args.N = [2000]
        args.shape = ["disk"]
        args.pattern = ["wedge"]
        args.distortion_kind = ["none", "rotate"]
        args.n_permutations = 100
        args.permutation_scope = "none"
    elif args.mode == "validation":
        args.n_reps = 15
        args.N = [1500, 2500]
        args.shape = ["disk", "peanut"]
        args.pattern = ["wedge", "core"]
        args.distortion_kind = ["none", "rotate", "jitter", "aniso_scale"]
        args.n_permutations = 250
        args.permutation_scope = "none"
    elif args.mode == "publication":
        if args.n_reps == 50:
            args.n_reps = 50
            args.N = [1000, 2000]
            args.shape = ["disk"]
            args.pattern = ["wedge", "core"]

            args.distortion_kind = ["none", "rotate", "jitter", "aniso_scale"]
            args.n_permutations = 500
            args.permutation_scope = "topk"
        else:
            args.n_reps = max(args.n_reps, 100)

            args.N = [1000, 2000, 5000]

            args.shape = ["disk", "annulus", "peanut"]

            args.pattern = ["uniform", "wedge", "core", "rim"]

            args.distortion_kind = [
                "none",
                "rotate",
                "jitter",
                "subsample",
                "aniso_scale",
                "swirl",
            ]

            args.n_permutations = 1000

            args.permutation_scope = "all"

    n_perms = args.n_permutations if args.permutation_scope == "all" else 0

    output_dir = io.ensure_output_dir("robustness", base_dir=args.outdir.rsplit("/", 1)[0])

    runs_csv_path = output_dir / "runs.csv"
    skip_completed = set()
    if args.resume and runs_csv_path.exists():
        skip_completed = checkpoint.load_completed_runs(runs_csv_path)
        print(f"Resuming: {len(skip_completed)} runs already completed")

    config = BioRSPConfig(
        B=72,
        delta_deg=60.0,
        n_permutations=n_perms,
    )

    configs = []
    for shape in args.shape:
        for N in args.N:
            for pattern in args.pattern:
                configs.extend(
                    [
                        {
                            "shape": shape,
                            "N": N,
                            "pattern": pattern,
                            "distortion_kind": distortion_kind,
                            "distortion_strength": strength,
                        }
                        for distortion_kind in args.distortion_kind
                        for strength in strength_map[distortion_kind]
                    ]
                )

    print(f"Running robustness benchmark: {len(configs)} conditions × {args.n_reps} reps")

    def save_checkpoint(results: list):
        """Save incremental checkpoint."""
        if not results:
            return
        checkpoint_df = pd.DataFrame(results)
        checkpoint.append_to_runs_csv(checkpoint_df, runs_csv_path)
        print(f"✓ Checkpoint saved ({len(results)} results)")

    start_time = time.time()

    runs_df = sweeps.run_replicates(
        run_robustness_pair,
        configs,
        args.n_reps,
        seed_start=args.seed,
        progress=True,
        n_jobs=args.n_jobs,
        fn_args=(config,),
        checkpoint_every=args.checkpoint_every,
        checkpoint_callback=save_checkpoint,
        skip_completed=skip_completed,
    )

    runtime = time.time() - start_time

    io.write_runs_csv(runs_df, output_dir, benchmark="robustness")

    INVARIANCE_DISTORTIONS = {"none", "rotate", "jitter", "subsample"}

    from biorsp.simulations import metrics

    summary_rows = []
    for (shape, N, pattern, distortion_kind, distortion_strength), group in runs_df.groupby(
        ["shape", "N", "pattern", "distortion_kind", "distortion_strength"]
    ):
        baseline_s = group["baseline_spatial_score"].values
        distorted_s = group["distorted_spatial_score"].values

        if len(baseline_s) >= 3:
            paired_stats = metrics.compute_paired_deltas(baseline_s, distorted_s)
        else:
            paired_stats = {
                "delta_mean": np.nan,
                "delta_median": np.nan,
                "delta_std": np.nan,
                "abs_delta_median": np.nan,
                "abs_delta_iqr": np.nan,
                "correlation": np.nan,
                "n_pairs": len(baseline_s),
            }

        category = "invariance" if distortion_kind in INVARIANCE_DISTORTIONS else "sensitivity"

        is_stable = (paired_stats["abs_delta_median"] < 0.05) or (paired_stats["correlation"] > 0.9)

        summary_rows.append(
            {
                "shape": shape,
                "N": N,
                "pattern": pattern,
                "distortion_kind": distortion_kind,
                "distortion_strength": distortion_strength,
                "category": category,
                "baseline_s_mean": (
                    baseline_s[~np.isnan(baseline_s)].mean() if len(baseline_s) > 0 else np.nan
                ),
                "baseline_s_std": (
                    baseline_s[~np.isnan(baseline_s)].std() if len(baseline_s) > 0 else np.nan
                ),
                "distorted_s_mean": (
                    distorted_s[~np.isnan(distorted_s)].mean() if len(distorted_s) > 0 else np.nan
                ),
                "delta_mean": paired_stats["delta_mean"],
                "delta_median": paired_stats["delta_median"],
                "delta_std": paired_stats["delta_std"],
                "abs_delta_median": paired_stats["abs_delta_median"],
                "abs_delta_iqr": paired_stats["abs_delta_iqr"],
                "correlation": paired_stats["correlation"],
                "is_stable": is_stable,
                "n_pairs": paired_stats["n_pairs"],
                "abstain_rate": group["distorted_abstain"].mean(),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    io.write_summary_csv(summary_df, output_dir, benchmark="robustness")

    print("Generating plots...")

    try:
        validation.validate_dataframe_for_plot(
            summary_df,
            required_columns=["distortion_kind", "distortion_strength", "abs_delta_median"],
            min_rows=1,
            name="robustness curves",
        )
    except validation.ValidationError as e:
        print(f"⚠ Skipping plots: {e}")
    else:
        for dist_kind in args.distortion_kind:
            if dist_kind == "none":
                continue
            subset = summary_df[summary_df["distortion_kind"] == dist_kind]
            if len(subset) > 0:
                try:
                    fig = plotting.plot_robustness_delta(
                        subset,
                        x_var="distortion_strength",
                        y_var="abs_delta_median",
                        title=f"{'Invariance' if dist_kind in INVARIANCE_DISTORTIONS else 'Sensitivity'}: {dist_kind}",
                    )
                    io.save_figure(fig, output_dir, f"robustness_{dist_kind}.png")
                except Exception as e:
                    print(f"⚠ Skipping plot for {dist_kind}: {e}")

    interpretation = docs.interpret_robustness(summary_df)
    docs.write_report(
        output_dir,
        "robustness",
        summary_df,
        params=vars(args),
        interpretation=interpretation,
    )

    io.write_manifest(
        output_dir,
        benchmark_name="robustness",
        params=vars(args),
        n_replicates=args.n_reps * len(configs),
        runtime_seconds=runtime,
        biorsp_config=config,
    )

    print("\n✅ Robustness benchmark complete!")
    print(f"   Output directory: {output_dir}")
    print(f"   Runtime: {runtime:.1f}s")


if __name__ == "__main__":
    main()
