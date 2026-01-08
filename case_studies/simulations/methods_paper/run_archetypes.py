"""
Archetype Recovery Benchmark for BioRSP Methods Paper.

Evaluates the method's ability to detect and distinguish diverse spatial archetypes:
1. Housekeeping (uniform expression)
2. Niche (core, rim, wedge patterns)
3. Regional (broad spatial domains)
4. Scattered (sparse expression)

Outputs: runs.csv, summary.csv, report.md, manifest.json, scatter plots, confusion matrix
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


def run_archetype_condition(config_dict: dict, seed: int, config: BioRSPConfig) -> dict:
    """Run one archetype replicate."""
    from simlib import (
        datasets,
        expression,
        rng,
        scoring,
        shapes,
    )

    shape = config_dict["shape"]
    N = config_dict["N"]
    pattern = config_dict["pattern"]

    condition_key = rng.condition_key(shape, N, pattern)
    gen = rng.make_rng(seed, "archetype", condition_key)

    coords, shape_meta = shapes.generate_coords(shape, N, gen)

    libsize = expression.simulate_library_size(
        N, gen, model="lognormal", params={"mean": 2000, "std": 0.5}
    )

    field = expression.generate_signal_field(coords, pattern, params={})

    counts = expression.generate_expression_from_field(
        field, libsize, gen, expr_model="nb", params={"phi": 10.0, "abundance": 1e-3}
    )

    adata = datasets.package_as_anndata(
        coords, counts[:, None], var_names=[f"{pattern}_gene"], obs_meta=None, embedding_key="X_sim"
    )

    t0 = time.time()
    results_df = scoring.score_dataset(adata, genes=[f"{pattern}_gene"], config=config)
    elapsed = time.time() - t0
    if len(results_df) == 0:
        return {
            "shape": shape,
            "N": N,
            "pattern": pattern,
            "p_value": np.nan,
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
        "p_value": row["p_value"],
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
        sweeps,
        validation,
    )

    parser = argparse.ArgumentParser(description="Archetype benchmark")
    parser.add_argument("--outdir", type=str, default=str(ROOT / "outputs" / "archetypes"))
    parser.add_argument("--seed", type=int, default=5000)
    parser.add_argument("--n_reps", type=int, default=50)
    parser.add_argument("--N", type=int, nargs="+", default=[2000, 5000])
    parser.add_argument("--shape", type=str, nargs="+", default=["disk", "peanut", "crescent"])
    parser.add_argument(
        "--pattern",
        type=str,
        nargs="+",
        default=[
            "uniform",
            "sparse",
            "core",
            "rim",
            "wedge",
            "wedge_core",
            "wedge_rim",
            "two_wedges",
        ],
    )
    parser.add_argument("--n_permutations", type=int, default=250)
    parser.add_argument("--mode", type=str, choices=["quick", "publication"], default="quick")
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument(
        "--n_workers", type=int, default=-1, help="Number of parallel workers (alias for --n_jobs)"
    )
    parser.add_argument(
        "--checkpoint_every", type=int, default=25, help="Save checkpoint every N runs"
    )
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument(
        "--permutation_scope",
        type=str,
        choices=["none", "all"],
        default="all",
        help="Permutation strategy: 'none' (no p-values), 'all' (compute p-values for all replicates)",
    )
    args = parser.parse_args()

    if args.n_workers != -1:
        args.n_jobs = args.n_workers

    if args.mode == "quick":

        args.n_reps = 5
        args.N = [2000]
        args.shape = ["disk"]
        args.pattern = ["uniform", "core", "rim"]
        args.n_permutations = 100
        args.permutation_scope = "none"
    elif args.mode == "publication":

        if args.n_reps == 50:

            args.n_reps = 50
            args.N = [1000, 2000]
            args.shape = ["disk", "peanut"]
            args.pattern = ["uniform", "core", "rim", "wedge"]
            args.n_permutations = 500
            args.permutation_scope = "topk"
        else:

            args.n_reps = max(args.n_reps, 100)

            args.N = [1000, 2000, 5000]

            args.shape = ["disk", "peanut", "crescent"]

            args.pattern = [
                "uniform",
                "sparse",
                "core",
                "rim",
                "wedge",
                "wedge_core",
                "wedge_rim",
                "two_wedges",
            ]

            args.n_permutations = 1000

            args.permutation_scope = "all"

    n_perms = args.n_permutations if args.permutation_scope == "all" else 0

    output_dir = io.ensure_output_dir("archetypes", base_dir=args.outdir.rsplit("/", 1)[0])

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

    configs = sweeps.expand_grid(shape=args.shape, N=args.N, pattern=args.pattern)

    print(f"Running archetype benchmark: {len(configs)} conditions × {args.n_reps} reps")

    def save_checkpoint(results: list):
        """Save incremental checkpoint."""
        if not results:
            return
        checkpoint_df = pd.DataFrame(results)
        checkpoint.append_to_runs_csv(checkpoint_df, runs_csv_path)
        print(f"✓ Checkpoint saved ({len(results)} results)")

    start_time = time.time()

    runs_df = sweeps.run_replicates(
        run_archetype_condition,
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

    io.write_runs_csv(runs_df, output_dir, benchmark="archetypes")

    summary_rows = []
    for (shape, N, pattern), group in runs_df.groupby(["shape", "N", "pattern"]):

        cov_mean = group["coverage_expr"].mean()
        ss_mean = group["spatial_score"].mean()

        high_cov = cov_mean > 0.3
        high_ss = ss_mean > 0.02

        if high_cov and high_ss:
            quadrant = "niche_localized"
            interpretation = "High prevalence, strong spatial structure"
        elif high_cov and not high_ss:
            quadrant = "housekeeping"
            interpretation = "High prevalence, uniform spatial distribution"
        elif not high_cov and high_ss:
            quadrant = "rare_localized"
            interpretation = "Low prevalence, spatially restricted"
        else:
            quadrant = "sparse"
            interpretation = "Low prevalence, sparse scattered expression"

        summary_rows.append(
            {
                "shape": shape,
                "N": N,
                "pattern": pattern,
                "spatial_score_mean": ss_mean,
                "spatial_score_std": group["spatial_score"].std(),
                "coverage_expr_mean": cov_mean,
                "coverage_expr_std": group["coverage_expr"].std(),
                "abstain_rate": group["abstain_flag"].mean(),
                "n_tests": len(group),
                "quadrant": quadrant,
                "interpretation": interpretation,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    io.write_summary_csv(summary_df, output_dir, benchmark="archetypes")

    print("Generating plots...")
    figs_dir = ROOT / "figs"
    figs_dir.mkdir(exist_ok=True)

    for shape in args.shape:
        subset = runs_df[runs_df["shape"] == shape]

        try:
            validation.validate_dataframe_for_plot(
                subset,
                required_columns=["coverage_expr", "spatial_score", "pattern"],
                min_rows=1,
                name=f"archetype scatter plot for {shape}",
            )
        except validation.ValidationError as e:
            print(f"⚠ Skipping scatter plot for {shape}: {e}")
            continue

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 6))
        for pattern in args.pattern:
            pattern_subset = subset[subset["pattern"] == pattern]
            ax.scatter(
                pattern_subset["coverage_expr"],
                pattern_subset["spatial_score"],
                label=pattern,
                alpha=0.6,
                s=20,
            )
        ax.set_xlabel("Coverage (C)")
        ax.set_ylabel("Spatial Score (S)")
        ax.set_title(f"Archetype Classification: {shape}")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        io.save_figure(fig, output_dir, f"archetypes_scatter_{shape}.png")

    interpretation = docs.interpret_archetypes(summary_df)
    docs.write_report(
        output_dir,
        "archetypes",
        summary_df,
        params=vars(args),
        interpretation=interpretation,
    )

    io.write_manifest(
        output_dir,
        benchmark_name="archetypes",
        params=vars(args),
        n_replicates=args.n_reps * len(configs),
        runtime_seconds=runtime,
        biorsp_config=config,
    )

    print("\n✅ Archetype benchmark complete!")
    print(f"   Outputs: {output_dir}")
    print(f"   Figures: {figs_dir}")
    print(f"   Runtime: {runtime:.1f}s")


if __name__ == "__main__":
    main()
