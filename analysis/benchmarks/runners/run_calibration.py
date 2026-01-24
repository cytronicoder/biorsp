"""Calibration benchmark with held-out evaluation and abstention handling."""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from biorsp import BioRSPConfig

# Local benchmark helpers are imported where needed inside `main()` to avoid module-level
# import-after-code issues (E402).
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run_calibration_condition(config_dict: dict, seed: int, config: "BioRSPConfig") -> dict:
    """Run a single calibration replicate.

    Args:
        config_dict: Condition configuration with keys such as `shape`, `N`,
            and `null_type`.
        seed: Random seed for the replicate.
        config: BioRSP scoring configuration (e.g., `B`, `delta_deg`,
            `n_permutations`).

    Returns:
        Row dictionary containing schema-required columns and metadata. May
        include abstention indicators (`abstain_flag`, `abstain_reason`).
    """
    from biorsp.simulations import (
        datasets,
        expression,
        rng,
        scoring,
        shapes,
    )

    shape = config_dict["shape"]
    N = config_dict["N"]
    null_type = config_dict["null_type"]

    permutation_scheme = expression.get_permutation_scheme(null_type)

    condition_key = rng.condition_key(shape, N, null_type)
    gen = rng.make_rng(seed, "calibration", condition_key)

    coords, shape_meta = shapes.generate_coords(shape, N, gen)

    libsize = expression.simulate_library_size(
        N, gen, model="lognormal", params={"mean": 1500, "std": 0.5}
    )

    counts, expr_meta = expression.generate_confounded_null(
        coords, libsize, gen, null_type=null_type, params={"n_depth_bins": 5}
    )

    adata = datasets.package_as_anndata(
        coords, counts[:, None], var_names=["null_gene"], obs_meta=None, embedding_key="X_sim"
    )

    if null_type == "depth_confounded" and "depth_bins" in expr_meta:
        adata.obs["depth_bin"] = expr_meta["depth_bins"]

    t0 = time.time()
    results_df = scoring.score_dataset(adata, genes=["null_gene"], config=config)
    elapsed = time.time() - t0

    perm_floor = 1.0 / (config.n_permutations + 1) if config.n_permutations > 0 else np.nan
    base_row = {
        "benchmark": "calibration",
        "shape": shape,
        "N": N,
        "null_type": null_type,
        "permutation_scheme": permutation_scheme,
        "n_permutations": config.n_permutations,
        "config_B": config.B,
        "config_delta_deg": config.delta_deg,
        "seed": seed,
        "time": elapsed,
        "test_stat": np.nan,
        "perm_floor": perm_floor,
        "alpha": 0.05,
    }

    if len(results_df) == 0:
        base_row.update(
            {
                "p_value": np.nan,
                "Spatial_Bias_Score": np.nan,
                "Coverage": np.nan,
                "coverage_bg": np.nan,
                "coverage_fg": np.nan,
                "abstain_flag": True,
                "abstain_reason": "no_results",
                "is_fp": False,
            }
        )
        return base_row

    row = results_df.iloc[0]
    base_row.update(
        {
            "p_value": row.get("p_value", np.nan),
            "Spatial_Bias_Score": row.get("Spatial_Bias_Score", np.nan),
            "Coverage": row.get("Coverage", np.nan),
            "coverage_bg": row.get("coverage_bg", np.nan),
            "coverage_fg": row.get("coverage_fg", np.nan),
            "abstain_flag": row.get("abstain_flag", False),
            "abstain_reason": row.get("abstain_reason", "ok"),
            "test_stat": row.get("test_stat", np.nan),
        }
    )
    base_row["is_fp"] = bool(
        (not base_row["abstain_flag"])
        and np.isfinite(base_row["p_value"])
        and base_row["p_value"] < base_row["alpha"]
    )
    return base_row


def main():
    """Run the calibration benchmark CLI.

    Parses command-line options, executes replicates across a condition grid,
    derives thresholds, produces plots, and writes contract artifacts.
    """
    from analysis.benchmarks.simlib.io_contract import BenchmarkContractConfig, init_run_dir
    from analysis.benchmarks.simlib.runner_harness import (
        finalize_contract,
        normalize_scores_df,
        safe_metric_mask,
        split_train_test,
    )
    from biorsp import BioRSPConfig
    from biorsp.plotting.standard import make_standard_plot_set
    from biorsp.simulations import (
        metrics,
        plotting,
        sweeps,
        validation,
    )

    parser = argparse.ArgumentParser(description="Calibration benchmark (held-out)")
    parser.add_argument(
        "--outdir",
        type=str,
        default=str(ROOT / "analysis" / "benchmarks" / "outputs"),
        help="Base output directory",
    )
    parser.add_argument(
        "--run_id", type=str, default=None, help="Run identifier (default: timestamp)"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_reps", type=int, default=100)
    parser.add_argument("--N", type=int, nargs="+", default=[500, 1000, 2000])
    parser.add_argument(
        "--shape", type=str, nargs="+", default=["disk", "ellipse", "annulus", "peanut"]
    )
    parser.add_argument(
        "--null_type",
        type=str,
        nargs="+",
        default=["iid", "depth_confounded", "mask_stress"],
        help="Null types: iid, depth_confounded, mask_stress (NOT density_confounded - that creates signal!)",
    )

    parser.add_argument("--n_permutations", type=int, default=250)
    parser.add_argument(
        "--mode", type=str, choices=["quick", "validation", "publication"], default="quick"
    )
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

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    config = BenchmarkContractConfig(
        outdir=args.outdir,
        benchmark="calibration",
        run_id=run_id,
        seed=args.seed,
        mode=args.mode,
    )
    paths = init_run_dir(config)

    start_time = time.time()

    # Mode presets: adjust args for different run modes
    if args.mode == "quick":
        args.n_reps = 100
        args.N = [500, 2000]  # Two sample sizes for quick validation
        args.shape = ["disk", "peanut"]  # Two geometries: convex + non-convex
        args.null_type = ["iid"]  # Focus on IID for quick calibration check
        args.n_permutations = 100
        args.permutation_scope = "all"
    elif args.mode == "validation":
        args.n_reps = 50
        args.N = [500, 1000, 2000]  # Three sample sizes
        args.shape = ["disk", "ellipse", "annulus", "peanut"]  # Four geometries
        args.null_type = ["iid", "depth_confounded"]
        args.n_permutations = 250
        args.permutation_scope = "all"
    elif args.mode == "publication":
        if args.n_reps == 50:
            args.n_reps = 50
            args.N = [500, 2000]
            args.shape = ["disk", "annulus"]
            args.null_type = ["iid", "depth_confounded", "mask_stress"]
            args.n_permutations = 500
            args.permutation_scope = "all"
        else:
            args.n_reps = max(args.n_reps, 100)
            args.N = [500, 1000, 2000, 5000]  # Four sample sizes
            args.shape = ["disk", "ellipse", "annulus", "peanut", "crescent"]  # Five geometries
            args.null_type = ["iid", "depth_confounded", "mask_stress"]
            args.n_permutations = 1000
            args.permutation_scope = "all"

    runs_csv_path = paths["runs_csv"]

    from biorsp.simulations import checkpoint

    skip_completed = None
    if args.resume:
        skip_completed = checkpoint.load_completed_runs(runs_csv_path)
        if len(skip_completed) > 0:
            print(f"Resume mode: found {len(skip_completed)} completed runs")

    # Define checkpoint callback
    def save_checkpoint(results: list) -> None:
        """Save partial results to runs.csv for resumability."""
        df_partial = pd.DataFrame(results)
        df_partial.to_csv(runs_csv_path, index=False)

    # Determine n_permutations based on permutation scope
    n_perms = args.n_permutations if args.permutation_scope != "none" else 0

    # Create sweep grid
    grid = sweeps.expand_grid(
        shape=args.shape,
        N=args.N,
        null_type=args.null_type,
    )

    # BioRSP config for scoring
    rsp_config = BioRSPConfig(B=72, delta_deg=60.0, n_permutations=n_perms)
    runs_df = sweeps.run_replicates(
        run_calibration_condition,
        grid,
        args.n_reps,
        seed_start=args.seed,
        progress=True,
        n_jobs=args.n_workers,
        fn_args=(rsp_config,),
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

    if runs_df.empty:
        raise RuntimeError("Calibration produced no runs; check configuration")

    runs_df = normalize_scores_df(runs_df)
    if "replicate" in runs_df.columns:
        runs_df = runs_df.rename(columns={"replicate": "replicate_id"})
    if "replicate_id" not in runs_df.columns:
        runs_df["replicate_id"] = np.arange(len(runs_df))
    runs_df["run_id"] = run_id
    runs_df["benchmark"] = "calibration"
    runs_df["mode"] = args.mode
    runs_df["timestamp"] = datetime.now(timezone.utc).isoformat()
    runs_df["status"] = np.where(runs_df.get("abstain_flag", False), "abstain", "ok")
    runs_df["case_id"] = runs_df.apply(
        lambda r: f"{r['shape']}-{r['N']}-{r['null_type']}-{int(r['seed'])}", axis=1
    )

    split = split_train_test(runs_df, group_cols=["case_id"], test_frac=0.25, seed=args.seed)
    runs_df["split"] = "train"
    runs_df.loc[split.test_idx, "split"] = "test"

    train_df = runs_df.loc[split.train_idx]
    test_df = runs_df.loc[split.test_idx]

    # Derive thresholds on train only (use Spatial_Score quantiles per null_type)
    threshold_rows = []
    for (null_type, shape), group in train_df.groupby(["null_type", "shape"]):
        mask = safe_metric_mask(group["Spatial_Score"])
        s_cut = float(np.quantile(group.loc[mask, "Spatial_Score"], 0.95)) if mask.any() else np.nan
        threshold_rows.append(
            {
                "alpha": 0.05,
                "threshold": s_cut,
                "n_train": int(mask.sum()),
                "n_test": int(len(test_df[test_df["null_type"] == null_type])),
                "seed": args.seed,
                "null_type": null_type,
                "shape": shape,
                "density": np.nan,
                "B": rsp_config.B,
                "delta_deg": rsp_config.delta_deg,
            }
        )
    calibration_thresholds = pd.DataFrame(threshold_rows)
    thresholds_path = paths["root"] / "calibration_thresholds.csv"
    calibration_thresholds.to_csv(thresholds_path, index=False)

    derived_path = paths["root"] / "derived_thresholds.json"
    with open(derived_path, "w") as f:
        json.dump({"source": "train_quantile_0.95", "n_train": len(train_df)}, f, indent=2)

    summary_rows = []
    figures = {}

    # Evaluate on test
    perm_floor = 1.0 / (args.n_permutations + 1) if args.n_permutations > 0 else None
    for (null_type, shape), group in test_df.groupby(["null_type", "shape"]):
        mask = safe_metric_mask(group["p_value"])
        n_total = len(group)
        n_tested = int(mask.sum())
        n_abstained = n_total - n_tested
        abstain_rate = n_abstained / n_total if n_total > 0 else np.nan

        if n_tested == 0:
            fig_empty, ax = plt.subplots(figsize=(5, 4))
            ax.text(0.5, 0.5, "No finite p-values; all abstained", ha="center", va="center")
            ax.axis("off")
            figures[f"debug_{null_type}_{shape}_pvals"] = (
                paths["root"] / "figures" / f"calibration_hist_{null_type}_{shape}.png"
            )
            fig_empty.savefig(
                figures[f"debug_{null_type}_{shape}_pvals"], dpi=300, bbox_inches="tight"
            )
            plt.close(fig_empty)
            continue

        p_values = group.loc[mask, "p_value"].to_numpy()
        fpr_05, ci_low_05, ci_high_05 = metrics.fpr_with_ci(p_values, alpha=0.05)
        fpr_01, ci_low_01, ci_high_01 = metrics.fpr_with_ci(p_values, alpha=0.01)
        ks_stat, ks_pval = metrics.ks_uniform(p_values)

        summary_rows.append(
            {
                "benchmark": "calibration",
                "shape": shape,
                "N": group["N"].iloc[0],
                "null_type": null_type,
                "permutation_scheme": group["permutation_scheme"].iloc[0],
                "n_reps": len(group),
                "n_total": n_total,
                "n_tested": n_tested,
                "n_abstained": n_abstained,
                "abstain_rate": abstain_rate,
                "fpr_05": fpr_05,
                "fpr_05_ci_low": ci_low_05,
                "fpr_05_ci_high": ci_high_05,
                "fpr_01": fpr_01,
                "fpr_01_ci_low": ci_low_01,
                "fpr_01_ci_high": ci_high_01,
                "ks_stat": ks_stat,
                "ks_pval": ks_pval,
                "metric": "fpr",
                "group_keys": json.dumps({"null_type": null_type, "shape": shape}),
                "mean": fpr_05,
                "std": np.nan,
                "n": n_tested,
                "ci_low": ci_low_05,
                "ci_high": ci_high_05,
                "method": "wilson",
            }
        )

        expected, observed = metrics.qq_quantiles(p_values)
        fig = plotting.plot_qq(
            expected,
            observed,
            title=f"QQ Plot: {null_type} ({shape})",
            perm_floor=perm_floor,
        )
        fig_path = paths["root"] / "figures" / f"calibration_qq_{null_type}_{shape}.png"
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        figures[f"qq_{null_type}_{shape}"] = fig_path

        fig_hist, ax_hist = plt.subplots(figsize=(6, 4))
        ax_hist.hist(p_values, bins=20, color="#2196F3", alpha=0.8, density=True)
        ax_hist.axhline(1.0, color="black", linestyle="--", linewidth=1.0, label="Uniform")
        ax_hist.set_title(
            f"p-value histogram ({null_type}, {shape})\nn_tested={n_tested}, abstain={abstain_rate:.2f}"
        )
        ax_hist.set_xlabel("p-value")
        ax_hist.set_ylabel("Density")
        ax_hist.legend()
        fig_hist.tight_layout()
        hist_path = paths["root"] / "figures" / f"calibration_hist_{null_type}_{shape}.png"
        fig_hist.savefig(hist_path, dpi=300, bbox_inches="tight")
        plt.close(fig_hist)
        figures[f"hist_{null_type}_{shape}"] = hist_path

    summary_df = pd.DataFrame(summary_rows)

    # Standard plot set if scores available
    try:
        std_figs = make_standard_plot_set(
            scores_df=runs_df,
            outdir=paths["root"],
            thresholds={"C_cut": 0.3, "S_cut": 0.15},
            truth_col=None,
            pred_col="Archetype_pred" if "Archetype_pred" in runs_df.columns else None,
            gene_col="gene" if "gene" in runs_df.columns else "gene",
            title="Calibration scores",
            debug=False,
        )
        figures.update(std_figs)
    except Exception:
        pass

    report_lines = [
        f"# Calibration Benchmark ({args.mode})",
        f"Run ID: {run_id}",
        f"Replicates: {len(runs_df)}",
        f"Train/Test split: {len(train_df)}/{len(test_df)}",
    ]
    if not summary_df.empty:
        row = summary_df.iloc[0]
        report_lines.append(
            f"Example FPR@0.05 ({row['null_type']}, {row['shape']}): {row['fpr_05']:.3f}"
        )

    manifest = {
        "benchmark": "calibration",
        "config": config.to_dict(),
        "n_rows": len(runs_df),
        "n_train": len(train_df),
        "n_test": len(test_df),
        "calibration_thresholds": str(thresholds_path.relative_to(paths["root"])),
    }

    finalize_contract(
        paths["root"],
        runs_df=runs_df,
        summary_df=(
            summary_df
            if not summary_df.empty
            else pd.DataFrame(
                [
                    {
                        "metric": "n_tested",
                        "group_keys": json.dumps({"scope": "test"}),
                        "mean": 0,
                        "std": np.nan,
                        "n": 1,
                        "ci_low": 0,
                        "ci_high": 0,
                        "method": "count",
                    }
                ]
            )
        ),
        manifest=manifest,
        report_md="\n".join(report_lines),
        figures=figures,
    )

    print(f"✅ Calibration benchmark complete → {paths['root']}")

    # Validate output data integrity
    print("\nValidating output data...")
    _, val_report = validation.load_and_validate_runs(
        runs_csv_path,
        benchmark="calibration",
        expected_shapes=args.shape,
        expected_N=args.N,
        expected_null_types=args.null_type,
        write_debug_json=True,
    )
    if not val_report.valid:
        print("⚠ Validation issues found:")
        for err in val_report.errors:
            print(f"  ERROR: {err}")
    for warn in val_report.warnings:
        print(f"  WARNING: {warn}")
    if val_report.valid:
        print("✓ Output validation passed")

    print("\n✅ Calibration benchmark complete!")
    print(f"   Output directory: {paths['root']}")
    print(f"   Runtime: {runtime:.1f}s")


if __name__ == "__main__":
    main()
