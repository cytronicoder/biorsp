"""
Calibration Benchmark for BioRSP Methods Paper.

Evaluates type I error control under various null hypotheses:
1. IID Null (Standard) - True null, no spatial structure
2. Depth Confounded - Library size varies spatially but expression is IID
3. Mask Stress - Very low prevalence to stress sector masking

NOTE: density_confounded creates TRUE spatial signal (expression ~ density),
so it is NOT a null hypothesis. It has been moved to archetypes as a "stress test"
for confound sensitivity analysis.

Outputs: runs.csv, summary.csv, report.md, manifest.json, QQ plots, FPR grids
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


def run_calibration_condition(config_dict: dict, seed: int, config: BioRSPConfig) -> dict:
    """Run one calibration replicate."""
    from simlib import (
        datasets,
        expression,
        rng,
        scoring,
        shapes,
    )

    shape = config_dict["shape"]
    N = config_dict["N"]
    null_type = config_dict["null_type"]

    condition_key = rng.condition_key(shape, N, null_type)
    gen = rng.make_rng(seed, "calibration", condition_key)

    coords, shape_meta = shapes.generate_coords(shape, N, gen)

    libsize = expression.simulate_library_size(
        N, gen, model="lognormal", params={"mean": 1500, "std": 0.5}
    )

    counts, expr_meta = expression.generate_confounded_null(
        coords, libsize, gen, null_type=null_type, params={}
    )

    adata = datasets.package_as_anndata(
        coords, counts[:, None], var_names=["null_gene"], obs_meta=None, embedding_key="X_sim"
    )

    t0 = time.time()
    results_df = scoring.score_dataset(adata, genes=["null_gene"], config=config)
    elapsed = time.time() - t0
    if len(results_df) == 0:
        return {
            "shape": shape,
            "N": N,
            "null_type": null_type,
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
        "null_type": null_type,
        "p_value": row["p_value"],
        "spatial_score": row["spatial_score"],
        "coverage_expr": row["coverage_expr"],
        "abstain_flag": row["abstain_flag"],
        "time": elapsed,
    }


def main():
    from simlib import (
        docs,
        io,
        metrics,
        plotting,
        sweeps,
    )

    parser = argparse.ArgumentParser(description="Calibration benchmark")
    parser.add_argument("--outdir", type=str, default=str(ROOT / "outputs" / "calibration"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_reps", type=int, default=100)
    parser.add_argument("--N", type=int, nargs="+", default=[500, 2000, 5000])
    parser.add_argument("--shape", type=str, nargs="+", default=["disk", "annulus", "peanut"])
    parser.add_argument(
        "--null_type",
        type=str,
        nargs="+",
        default=["iid", "depth_confounded", "mask_stress"],
        help="Null types: iid, depth_confounded, mask_stress (NOT density_confounded - that creates signal!)",
    )

    parser.add_argument("--n_permutations", type=int, default=250)
    parser.add_argument("--mode", type=str, choices=["quick", "publication"], default="quick")
    parser.add_argument(
        "--n_workers", type=int, default=1, help="Parallel workers (-1 = all cores)"
    )
    parser.add_argument("--checkpoint_every", type=int, default=25, help="Save every N replicates")
    parser.add_argument("--resume", action="store_true", help="Resume from existing runs.csv")
    parser.add_argument(
        "--permutation_scope",
        type=str,
        choices=["none", "topk", "all"],
        default="all",
        help="Permutation strategy",
    )
    args = parser.parse_args()

    if args.mode == "quick":

        args.n_reps = 10
        args.N = [1000]
        args.shape = ["disk"]
        args.null_type = ["iid"]
        args.n_permutations = 100
        args.permutation_scope = "none"
    elif args.mode == "publication":

        if args.n_reps == 50:

            args.n_reps = 50
            args.N = [500, 2000]
            args.shape = ["disk", "annulus"]
            args.null_type = ["iid", "depth_confounded", "mask_stress"]
            args.n_permutations = 500
            args.permutation_scope = "topk"
        else:

            args.n_reps = max(args.n_reps, 100)

            args.N = [500, 1000, 2000, 5000]

            args.shape = ["disk", "annulus", "peanut"]

            args.null_type = ["iid", "depth_confounded", "mask_stress"]
            args.n_permutations = 1000
            args.permutation_scope = "all"

    output_dir = io.ensure_output_dir("calibration", base_dir=args.outdir.rsplit("/", 1)[0])
    runs_csv_path = output_dir / "runs.csv"

    from simlib import checkpoint

    skip_completed = None
    if args.resume:
        skip_completed = checkpoint.load_completed_runs(runs_csv_path)
        if len(skip_completed) > 0:
            print(f"Resume mode: found {len(skip_completed)} completed runs")

    n_perms = args.n_permutations if args.permutation_scope != "none" else 0
    config = BioRSPConfig(
        B=72,
        delta_deg=60.0,
        n_permutations=n_perms,
        qc_mode="principled",
    )

    configs = sweeps.expand_grid(shape=args.shape, N=args.N, null_type=args.null_type)

    print(f"Running calibration benchmark: {len(configs)} conditions × {args.n_reps} reps")

    start_time = time.time()

    def save_checkpoint(results):
        checkpoint.append_to_runs_csv(results, runs_csv_path, overwrite=False)
        print(f"\n✓ Checkpoint saved ({len(results)} results)")

    runs_df = sweeps.run_replicates(
        run_calibration_condition,
        configs,
        args.n_reps,
        seed_start=args.seed,
        progress=True,
        n_jobs=args.n_workers,
        fn_args=(config,),
        checkpoint_every=args.checkpoint_every,
        checkpoint_callback=save_checkpoint if args.checkpoint_every > 0 else None,
        skip_completed=skip_completed,
    )

    runtime = time.time() - start_time

    if args.resume and runs_csv_path.exists():
        df_existing = pd.read_csv(runs_csv_path)
        if len(runs_df) > 0:
            runs_df = pd.concat([df_existing, runs_df], ignore_index=True)
        else:
            runs_df = df_existing

    io.write_runs_csv(runs_df, output_dir, benchmark="calibration")

    summary_rows = []
    for (shape, N, null_type), group in runs_df.groupby(["shape", "N", "null_type"]):
        p_values = group["p_value"].values

        fpr_05, ci_low_05, ci_high_05 = metrics.fpr_with_ci(p_values, alpha=0.05)
        fpr_01, ci_low_01, ci_high_01 = metrics.fpr_with_ci(p_values, alpha=0.01)
        ks_stat, ks_pval = metrics.ks_uniform(p_values)
        abstain_rate = group["abstain_flag"].mean()

        summary_rows.append(
            {
                "shape": shape,
                "N": N,
                "null_type": null_type,
                "fpr_0p05": fpr_05,
                "fpr_0p05_ci_low": ci_low_05,
                "fpr_0p05_ci_high": ci_high_05,
                "fpr_0p01": fpr_01,
                "fpr_0p01_ci_low": ci_low_01,
                "fpr_0p01_ci_high": ci_high_01,
                "ks_stat": ks_stat,
                "ks_pval": ks_pval,
                "abstain_rate": abstain_rate,
                "n_tests": len(p_values),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    io.write_summary_csv(summary_df, output_dir, benchmark="calibration")

    print("Generating plots...")
    figs_dir = ROOT / "figs"
    figs_dir.mkdir(exist_ok=True)

    from simlib import validation

    for null_type in args.null_type:
        subset = runs_df[runs_df["null_type"] == null_type]
        p_values = subset["p_value"].dropna().values
        if len(p_values) > 0:
            expected, observed = metrics.qq_quantiles(p_values)
            fig = plotting.plot_qq(expected, observed, title=f"QQ Plot: {null_type}")
            io.save_figure(fig, output_dir, f"calibration_qq_{null_type}.png")
        else:
            print(f"Warning: No valid p-values for null_type={null_type}, skipping QQ plot")

    if len(summary_df) > 0:
        plot_df = summary_df.rename(columns={"fpr_0p05": "fpr"})
        for shape in args.shape:
            shape_df = plot_df[plot_df["shape"] == shape]
            if len(shape_df) > 1:
                try:
                    validation.validate_dataframe_for_plot(
                        shape_df, required_columns=["fpr", "N", "null_type"], min_rows=2
                    )
                    fig = plotting.plot_fpr_grid(
                        shape_df,
                        row_var="null_type",
                        col_var="N",
                        title=f"False Positive Rate (α=0.05): {shape}",
                    )
                    io.save_figure(fig, output_dir, f"calibration_fpr_grid_{shape}.png")
                except validation.ValidationError as e:
                    print(f"Skipping FPR grid for {shape}: {e}")

    interpretation = docs.interpret_calibration(summary_df, alpha=0.05)
    docs.write_report(
        output_dir,
        "calibration",
        summary_df,
        params=vars(args),
        interpretation=interpretation,
    )

    io.write_manifest(
        output_dir,
        benchmark_name="calibration",
        params=vars(args),
        n_replicates=args.n_reps * len(configs),
        runtime_seconds=runtime,
        biorsp_config=config,
    )

    print("\n✅ Calibration benchmark complete!")
    print(f"   Outputs: {output_dir}")
    print(f"   Figures: {figs_dir}")
    print(f"   Runtime: {runtime:.1f}s")


if __name__ == "__main__":
    main()
