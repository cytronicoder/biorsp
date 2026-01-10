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


def run_robustness_condition(config_dict: dict, seed: int, config: BioRSPConfig) -> dict:
    """Run one robustness replicate."""
    from simlib import (
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

    coords_dist, dist_meta = distortions.apply_distortion(
        coords_base, distortion_kind, distortion_strength, gen, params={}
    )

    if distortion_kind == "subsample" and distortion_strength < 1.0:
        n_keep = int(N * distortion_strength)
        indices = gen.choice(N, n_keep, replace=False)
        coords_dist = coords_base[indices]
        counts = counts[indices]
        libsize = libsize[indices]

    adata = datasets.package_as_anndata(
        coords_dist,
        counts[:, None],
        var_names=[f"{pattern}_gene"],
        obs_meta=None,
        embedding_key="X_sim",
    )

    t0 = time.time()
    results_df = scoring.score_dataset(adata, genes=[f"{pattern}_gene"], config=config)
    elapsed = time.time() - t0

    if len(results_df) == 0:
        return {
            "shape": shape,
            "N": N,
            "pattern": pattern,
            "distortion_kind": distortion_kind,
            "distortion_strength": distortion_strength,
            "spatial_score": np.nan,
            "coverage_expr": np.nan,
            "abstain_flag": True,
            "time": elapsed,
        }

    row = results_df.iloc[0]
    return {
        "shape": shape,
        "N": N,
        "pattern": pattern,
        "distortion_kind": distortion_kind,
        "distortion_strength": distortion_strength,
        "spatial_score": row["spatial_score"],
        "coverage_expr": row["coverage_expr"],
        "abstain_flag": row["abstain_flag"],
        "time": elapsed,
    }


def main():
    from simlib import (
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
    parser.add_argument("--mode", type=str, choices=["quick", "publication"], default="quick")
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
            # Note: For focused analysis, run separately:

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
        qc_mode="principled",
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
        run_robustness_condition,
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

    summary_rows = []

    baseline_df = runs_df[
        (runs_df["distortion_kind"] == "none") | (runs_df["distortion_strength"] == 0.0)
    ]
    baseline_s_mean = baseline_df["spatial_score"].mean() if len(baseline_df) > 0 else np.nan
    baseline_s_std = baseline_df["spatial_score"].std() if len(baseline_df) > 0 else np.nan

    for (shape, N, pattern, distortion_kind, distortion_strength), group in runs_df.groupby(
        ["shape", "N", "pattern", "distortion_kind", "distortion_strength"]
    ):
        s_mean = group["spatial_score"].mean()
        s_std = group["spatial_score"].std()
        delta_s = abs(s_mean - baseline_s_mean) if not np.isnan(baseline_s_mean) else np.nan

        category = "invariance" if distortion_kind in INVARIANCE_DISTORTIONS else "sensitivity"

        is_unstable = delta_s > 2 * baseline_s_std if not np.isnan(baseline_s_std) else False

        summary_rows.append(
            {
                "shape": shape,
                "N": N,
                "pattern": pattern,
                "distortion_kind": distortion_kind,
                "distortion_strength": distortion_strength,
                "category": category,
                "spatial_score_mean": s_mean,
                "spatial_score_std": s_std,
                "coverage_expr_mean": group["coverage_expr"].mean(),
                "coverage_expr_std": group["coverage_expr"].std(),
                "delta_from_baseline": delta_s,
                "is_unstable": is_unstable,
                "abstain_rate": group["abstain_flag"].mean(),
                "n_tests": len(group),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    io.write_summary_csv(summary_df, output_dir, benchmark="robustness")

    print("Generating plots...")
    figs_dir = ROOT / "outputs" / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    try:
        validation.validate_dataframe_for_plot(
            summary_df,
            required_columns=["distortion_kind", "distortion_strength", "delta_from_baseline"],
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
                        y_var="delta_from_baseline",
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
    print(f"   Outputs: {output_dir}")
    print(f"   Figures: {figs_dir}")
    print(f"   Runtime: {runtime:.1f}s")


if __name__ == "__main__":
    main()
