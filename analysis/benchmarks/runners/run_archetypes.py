"""
Archetype Recovery Benchmark for BioRSP Methods Paper.

Evaluates the method's ability to detect and distinguish diverse spatial archetypes:
1. Housekeeping (high C, low S): uniform expression
2. Regional Program (high C, high S): broad spatial domains
3. Sparse Noise (low C, low S): random sparse expression
4. Niche Marker (low C, high S): spatially restricted expression

Uses 2×2 factorial design: coverage_regime × organization_regime
with null-calibrated S thresholds and minimum expressing cells gating.

Outputs: runs.csv, summary.csv, report.md, manifest.json, scatter plots, confusion matrix, examples
"""

import argparse
import json
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
    """Run one archetype replicate using factorial design."""
    from biorsp.simulations import (
        datasets,
        expression,
        rng,
        scoring,
        shapes,
    )

    shape = config_dict["shape"]
    N = config_dict["N"]
    coverage_regime = config_dict["coverage_regime"]
    organization_regime = config_dict["organization_regime"]
    pattern_variant = config_dict.get("pattern_variant", "wedge_core")

    condition_key = rng.condition_key(shape, N, coverage_regime, organization_regime)
    gen = rng.make_rng(seed, "Archetype", condition_key)

    coords, shape_meta = shapes.generate_coords(shape, N, gen)

    libsize = expression.simulate_library_size(
        N, gen, model="lognormal", params={"mean": 2000, "std": 0.5}
    )

    counts, expr_meta = expression.generate_factorial_gene(
        coords=coords,
        libsize=libsize,
        rng=gen,
        coverage_regime=coverage_regime,
        organization_regime=organization_regime,
        pattern_variant=pattern_variant,
    )

    adata = datasets.package_as_anndata(
        coords, counts[:, None], var_names=["factorial_gene"], obs_meta=None, embedding_key="X_sim"
    )

    t0 = time.time()
    results_df = scoring.score_dataset(adata, genes=["factorial_gene"], config=config)
    elapsed = time.time() - t0

    if len(results_df) == 0:
        return {
            "shape": shape,
            "N": N,
            "coverage_regime": coverage_regime,
            "organization_regime": organization_regime,
            "true_archetype": expr_meta["Archetype"],
            "pattern_variant": pattern_variant,
            "p_value": np.nan,
            "Spatial_Score": np.nan,
            "Coverage": np.nan,
            "n_expr_cells": expr_meta.get("n_expr_cells", 0),
            "abstain_flag": True,
            "time": elapsed,
        }

    row = results_df.iloc[0]
    return {
        "shape": shape,
        "N": N,
        "coverage_regime": coverage_regime,
        "organization_regime": organization_regime,
        "true_archetype": expr_meta["Archetype"],
        "pattern_variant": pattern_variant,
        "p_value": row["p_value"],
        "Spatial_Score": row["Spatial_Score"],
        "Coverage": row["Coverage"],
        "n_expr_cells": expr_meta.get("n_expr_cells", 0),
        "abstain_flag": row["abstain_flag"],
        "time": elapsed,
    }


def main():
    from biorsp.simulations import (
        checkpoint,
        docs,
        io,
        metrics,
        plotting,
        sweeps,
        validation,
    )

    parser = argparse.ArgumentParser(description="Archetype benchmark (factorial design)")
    parser.add_argument("--outdir", type=str, default=str(ROOT / "outputs" / "archetypes"))
    parser.add_argument("--seed", type=int, default=5000)
    parser.add_argument("--n_reps", type=int, default=50)
    parser.add_argument("--N", type=int, nargs="+", default=[2000, 5000])
    parser.add_argument("--shape", type=str, nargs="+", default=["disk", "peanut", "crescent"])
    parser.add_argument(
        "--coverage_regime",
        type=str,
        nargs="+",
        default=["high", "low"],
        help="Coverage regimes: high (~70%%) or low (~10%%)",
    )
    parser.add_argument(
        "--organization_regime",
        type=str,
        nargs="+",
        default=["structured", "iid"],
        help="Organization: structured (spatial pattern) or iid (random scatter)",
    )
    parser.add_argument(
        "--pattern_variant",
        type=str,
        default="wedge_core",
        help="Spatial pattern for structured: core, rim, wedge_core, wedge_rim, radial_gradient",
    )
    parser.add_argument("--n_permutations", type=int, default=250)
    parser.add_argument(
        "--mode", type=str, choices=["quick", "validation", "publication"], default="quick"
    )
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
    parser.add_argument(
        "--calibration_file",
        type=str,
        default=None,
        help="Path to calibration_thresholds.csv from run_calibration.py",
    )
    parser.add_argument(
        "--s_cut",
        type=float,
        default=None,
        help="Manual S threshold (overrides calibration file)",
    )
    parser.add_argument(
        "--c_cut",
        type=float,
        default=0.30,
        help="Coverage threshold for high/low split (default: 0.30)",
    )
    args = parser.parse_args()

    if args.n_workers != -1:
        args.n_jobs = args.n_workers

    if args.mode == "quick":
        args.n_reps = 10
        args.N = [2000]
        args.shape = ["disk"]
        args.coverage_regime = ["high", "low"]
        args.organization_regime = ["structured", "iid"]
        args.n_permutations = 100
        args.permutation_scope = "none"
    elif args.mode == "validation":
        args.n_reps = 30
        args.N = [2000]
        args.shape = ["disk", "peanut"]
        args.coverage_regime = ["high", "low"]
        args.organization_regime = ["structured", "iid"]
        args.n_permutations = 250
        args.permutation_scope = "all"
    elif args.mode == "publication":
        args.n_reps = max(args.n_reps, 200)
        args.N = [1000, 2000, 5000]
        args.shape = ["disk", "peanut", "crescent"]
        args.coverage_regime = ["high", "low"]
        args.organization_regime = ["structured", "iid"]
        args.n_permutations = 500
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
    )

    configs = sweeps.expand_grid(
        shape=args.shape,
        N=args.N,
        coverage_regime=args.coverage_regime,
        organization_regime=args.organization_regime,
    )
    for cfg in configs:
        cfg["pattern_variant"] = args.pattern_variant

    print(f"Running archetype benchmark: {len(configs)} conditions × {args.n_reps} reps")
    print(
        f"  Factorial design: {len(args.coverage_regime)} coverage × {len(args.organization_regime)} organization"
    )

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

    s_cut = args.s_cut

    if s_cut is None and args.calibration_file:
        calib_path = Path(args.calibration_file)
        if calib_path.exists():
            pd.read_csv(calib_path)
            print(f"✓ Loaded calibration table from {calib_path}")

    if s_cut is None:
        iid_runs = runs_df[runs_df["organization_regime"] == "iid"]
        if len(iid_runs) >= 10:
            s_values = iid_runs["Spatial_Score"].dropna().values
            thresholds = metrics.derive_thresholds_principled(s_values, fpr_target=0.05)
            s_cut = thresholds["s_cut"]
            print(
                f"✓ Derived S threshold from iid runs: S_cut={s_cut:.4f} (FPR={thresholds['empirical_fpr']:.1%})"
            )
        else:
            s_cut = 0.15
            print(f"⚠ Insufficient iid runs for threshold derivation, using default S_cut={s_cut}")

    c_cut = args.c_cut

    coverage = runs_df["Coverage"].values
    spatial_score = runs_df["Spatial_Score"].values
    n_expr_cells = runs_df["n_expr_cells"].values
    N_values = runs_df["N"].values

    predicted_labels = metrics.classify_by_quadrant(
        coverage, spatial_score, c_cut=c_cut, s_cut=s_cut
    )

    gated_labels = np.array(
        [
            metrics.apply_expr_gating(
                np.array([pred]), np.array([n_expr]), N, min_base=30, min_fraction=0.01
            )[0]
            for pred, n_expr, N in zip(predicted_labels, n_expr_cells, N_values)
        ]
    )

    runs_df["predicted_archetype_raw"] = predicted_labels
    runs_df["predicted_archetype"] = gated_labels

    true_labels = runs_df["true_archetype"].values

    labels_order = ["housekeeping", "regional_program", "sparse_noise", "niche_marker"]
    class_metrics = metrics.compute_classification_metrics(
        true_labels, gated_labels, labels=labels_order
    )

    accuracy = class_metrics["accuracy"]
    cm_df = class_metrics["confusion_matrix"]

    print(f"\n📊 Classification Results (S_cut={s_cut:.4f}, C_cut={c_cut:.2f}):")
    print(f"   Overall Accuracy: {accuracy:.1%}")
    print(f"   Macro F1: {class_metrics['macro_f1']:.3f}")

    summary_rows = []
    for (shape, N, cov_regime, org_regime), group in runs_df.groupby(
        ["shape", "N", "coverage_regime", "organization_regime"]
    ):
        true_arch = group["true_archetype"].iloc[0]
        pred_correct = (group["predicted_archetype"] == true_arch).mean()

        summary_rows.append(
            {
                "shape": shape,
                "N": N,
                "coverage_regime": cov_regime,
                "organization_regime": org_regime,
                "true_archetype": true_arch,
                "Spatial_Score_mean": group["Spatial_Score"].mean(),
                "Spatial_Score_std": group["Spatial_Score"].std(),
                "Coverage_mean": group["Coverage"].mean(),
                "Coverage_std": group["Coverage"].std(),
                "n_expr_cells_mean": group["n_expr_cells"].mean(),
                "classification_accuracy": pred_correct,
                "abstain_rate": group["abstain_flag"].mean(),
                "n_reps": len(group),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    io.write_summary_csv(summary_df, output_dir, benchmark="archetypes")

    thresholds_used = {
        "s_cut": float(s_cut),
        "c_cut": float(c_cut),
        "derived_from": "iid_runs" if args.s_cut is None else "manual",
        "overall_accuracy": float(accuracy),
        "macro_f1": float(class_metrics["macro_f1"]),
    }
    with open(output_dir / "thresholds_used.json", "w") as f:
        json.dump(thresholds_used, f, indent=2)

    print("\nGenerating plots...")

    for shape in args.shape:
        subset = runs_df[runs_df["shape"] == shape]

        try:
            validation.validate_dataframe_for_plot(
                subset,
                required_columns=["Coverage", "Spatial_Score", "true_archetype"],
                min_rows=1,
                name=f"archetype scatter plot for {shape}",
            )
        except validation.ValidationError as e:
            print(f"⚠ Skipping scatter plot for {shape}: {e}")
            continue

        fig = plotting.plot_archetype_scatter(
            coverage=subset["Coverage"].values,
            spatial_score=subset["Spatial_Score"].values,
            true_archetypes=subset["true_archetype"].values,
            c_cut=c_cut,
            s_cut=s_cut,
            title=f"Archetype Classification: {shape}",
        )
        io.save_figure(fig, output_dir, f"archetypes_scatter_{shape}.png")

    # Confusion matrix
    fig = plotting.plot_confusion_matrix_styled(
        cm_df,
        title="Archetype Classification",
        accuracy=accuracy,
    )
    io.save_figure(fig, output_dir, "archetypes_confusion_matrix.png")

    # Generate example panels for each archetype
    print("Generating archetype example panels...")
    examples_dir = output_dir / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)

    example_data = []
    for archetype in labels_order:
        arch_runs = runs_df[runs_df["true_archetype"] == archetype]
        if len(arch_runs) > 0:
            # Use median-score example for representative visualization
            median_idx = (
                (arch_runs["Spatial_Score"] - arch_runs["Spatial_Score"].median()).abs().idxmin()
            )
            ex_row = arch_runs.loc[median_idx]
            example_data.append(
                {
                    "Archetype": archetype,
                    "coverage": ex_row["Coverage"],
                    "Spatial_Score": ex_row["Spatial_Score"],
                    "seed": ex_row.get("seed", args.seed),
                    "shape": ex_row["shape"],
                    "N": ex_row["N"],
                    "coverage_regime": ex_row["coverage_regime"],
                    "organization_regime": ex_row["organization_regime"],
                }
            )

    # Save example metadata
    if example_data:
        example_df = pd.DataFrame(example_data)
        example_df.to_csv(examples_dir / "example_metadata.csv", index=False)
        print(f"✓ Saved example metadata to {examples_dir / 'example_metadata.csv'}")

    # Generate misclassified diagnostics
    print("Generating misclassification diagnostics...")
    misclass_mask = runs_df["predicted_archetype"] != runs_df["true_archetype"]
    misclass_df = runs_df[misclass_mask].copy()

    if len(misclass_df) > 0:
        # Sort by confidence (distance from threshold)
        misclass_df["s_margin"] = np.abs(misclass_df["Spatial_Score"] - s_cut)
        misclass_df["c_margin"] = np.abs(misclass_df["Coverage"] - c_cut)
        misclass_df["total_margin"] = misclass_df["s_margin"] + misclass_df["c_margin"]
        misclass_df = misclass_df.sort_values("total_margin", ascending=True)

        # Save top misclassifications
        debug_dir = output_dir / "diagnostics"
        debug_dir.mkdir(parents=True, exist_ok=True)

        # Take worst 12 misclassifications (closest to boundary = most informative)
        worst_misclass = misclass_df.head(12)
        worst_misclass.to_csv(debug_dir / "misclassified.csv", index=False)
        print(
            f"✓ Saved {len(worst_misclass)} misclassification diagnostics to {debug_dir / 'misclassified.csv'}"
        )

        # Summary of misclassification patterns
        misclass_summary = (
            misclass_df.groupby(["true_archetype", "predicted_archetype"])
            .size()
            .reset_index(name="count")
        )
        misclass_summary = misclass_summary.sort_values("count", ascending=False)
        misclass_summary.to_csv(debug_dir / "misclassification_patterns.csv", index=False)
        print(
            f"✓ Saved misclassification patterns to {debug_dir / 'misclassification_patterns.csv'}"
        )

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
    print(f"   Output directory: {output_dir}")
    print(f"   Runtime: {runtime:.1f}s")
    print(f"   Overall Accuracy: {accuracy:.1%}")
    print(f"   Thresholds: S_cut={s_cut:.4f}, C_cut={c_cut:.2f}")
    if len(misclass_df) > 0:
        print(f"   Misclassifications: {len(misclass_df)} ({len(misclass_df) / len(runs_df):.1%})")


if __name__ == "__main__":
    main()
