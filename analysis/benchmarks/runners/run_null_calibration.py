"""
Null Calibration for Deriving Thresholds.

Runs a lightweight calibration to derive S_cut and C_cut thresholds from null simulations.
These thresholds are then used by run_story_onepager.py for classification.

Usage:
    python run_null_calibration.py --mode quick --outdir outputs/calibration_thresholds --seed 42

Outputs:
    - thresholds.json: Derived thresholds for use by other scripts
    - null_distribution.csv: Raw null S values
    - report.md: Calibration summary
    - runs.csv, summary.csv, manifest.json (contract outputs)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Local helper imports are performed inside `run_calibration()` to avoid module-level
# imports after side-effectful code (E402).

MODE_CONFIGS = {
    "quick": {
        "n_reps": 30,
        "n_cells": 1500,
        "null_types": ["iid"],
        "n_permutations": 0,
    },
    "validation": {
        "n_reps": 75,
        "n_cells": 2000,
        "null_types": ["iid", "depth_confounded"],
        "n_permutations": 100,
    },
    "publication": {
        "n_reps": 150,
        "n_cells": 2500,
        "null_types": ["iid", "depth_confounded", "mask_stress"],
        "n_permutations": 500,
    },
}


def run_calibration(args):
    """Run null calibration.

    Args:
        args: Parsed CLI arguments.
    """
    from analysis.benchmarks.simlib.io_contract import BenchmarkContractConfig, init_run_dir
    from analysis.benchmarks.simlib.runner_harness import (
        finalize_contract,
        normalize_scores_df,
        safe_metric_mask,
    )
    from biorsp import BioRSPConfig
    from biorsp.plotting.standard import make_standard_plot_set
    from biorsp.simulations import datasets, expression, metrics, rng, scoring, shapes

    mode_cfg = MODE_CONFIGS[args.mode]

    output_dir = Path(args.outdir)
    contract_cfg = BenchmarkContractConfig(require_runs_csv=True, require_summary_csv=True)
    session_id = init_run_dir(output_dir, clear_existing=True, contract_config=contract_cfg)

    print("=" * 60)
    print("Null Calibration for Threshold Derivation")
    print(f"Mode: {args.mode}")
    print(f"Replicates: {mode_cfg['n_reps']} per null type")
    print("=" * 60)

    start_time = time.time()

    config = BioRSPConfig(
        B=72,
        delta_deg=60.0,
        n_permutations=mode_cfg["n_permutations"],
    )

    all_results = []

    for null_type in mode_cfg["null_types"]:
        print(f"\nRunning {null_type} null ({mode_cfg['n_reps']} reps)...")

        for rep in range(mode_cfg["n_reps"]):
            gen = rng.make_rng(args.seed + rep, "null_calibration", null_type)

            coords, _ = shapes.generate_coords("disk", mode_cfg["n_cells"], gen)
            libsize = expression.simulate_library_size(
                mode_cfg["n_cells"], gen, model="lognormal", params={"mean": 2000, "sigma": 0.5}
            )

            counts, _ = expression.generate_confounded_null(
                coords, libsize, gen, null_type=null_type, params={}
            )

            adata = datasets.package_as_anndata(
                coords, counts[:, None], var_names=["null_gene"], embedding_key="X_sim"
            )

            try:
                scores_df = scoring.score_dataset(
                    adata, genes=["null_gene"], config=config, embedding_key="X_sim"
                )

                scores_df = normalize_scores_df(scores_df)

                if len(scores_df) > 0:
                    row = scores_df.iloc[0]
                    all_results.append(
                        {
                            "benchmark": "null_calibration",
                            "null_type": null_type,
                            "replicate": rep,
                            "seed": args.seed + rep,
                            "Spatial_Bias_Score": row["Spatial_Bias_Score"],
                            "Coverage": row["Coverage"],
                            "p_value": row.get("p_value", np.nan),
                            "abstain_flag": row["abstain_flag"],
                            "abstain_reason": row.get("abstain_reason", ""),
                            "config_B": config.B,
                            "config_delta_deg": config.delta_deg,
                        }
                    )
            except Exception as e:
                print(f"  Warning: Rep {rep} failed: {e}")

    results_df = pd.DataFrame(all_results)

    # Compute finite-only metrics for calibration (abstention-aware)
    finite_mask = safe_metric_mask(results_df, "p_value")
    valid_s = results_df.loc[finite_mask, "Spatial_Bias_Score"].values
    valid_c = results_df.loc[finite_mask, "Coverage"].values
    valid_p = results_df.loc[finite_mask, "p_value"].values

    thresholds = metrics.derive_thresholds_from_null(
        null_s_values=valid_s,
        null_c_values=valid_c,
        s_quantile=0.95 if args.mode == "publication" else 0.90,
        c_quantile=0.30,
    )

    print("\nDerived thresholds:")
    print(f"  S_cut (95th percentile): {thresholds['s_cut']:.4f}")
    print(f"  C_cut: {thresholds['c_cut']:.4f}")
    print(f"  Based on {thresholds['n_samples']} valid null samples")

    # KS test for p-value uniformity
    if len(valid_p) > 10:
        ks_stat, ks_pval = metrics.ks_uniform(valid_p)
        thresholds["ks_stat"] = float(ks_stat)
        thresholds["ks_pval"] = float(ks_pval)
        print(f"  KS test for uniformity: stat={ks_stat:.4f}, p={ks_pval:.4f}")

    # Write outputs
    results_df.to_csv(output_dir / "runs.csv", index=False)

    with open(output_dir / "calibration_thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=2)

    # Summary table
    summary_rows = []
    for nt in mode_cfg["null_types"]:
        nt_df = results_df[results_df["null_type"] == nt]
        nt_finite = nt_df[~nt_df["abstain_flag"]]
        summary_rows.append(
            {
                "null_type": nt,
                "n_reps": len(nt_df),
                "n_abstained": nt_df["abstain_flag"].sum(),
                "mean_S": nt_finite["Spatial_Bias_Score"].mean(),
                "std_S": nt_finite["Spatial_Bias_Score"].std(),
                "mean_C": nt_finite["Coverage"].mean(),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "summary.csv", index=False)

    # QQ plot, FPR, p-value histogram (abstention-aware)
    plot_null_calibration_diagnostics(output_dir, results_df, valid_p, thresholds, args.mode)

    # Standard plot set (if applicable)
    if len(results_df[~results_df["abstain_flag"]]) > 0:
        make_standard_plot_set(
            scores_df=results_df[~results_df["abstain_flag"]],
            outdir=output_dir,
            plot_config={"show_truth": False},
        )

    elapsed = time.time() - start_time

    report = f"""# Null Calibration Report

Mode: {args.mode}
Runtime: {elapsed:.1f}s
Session ID: {session_id}

## Thresholds

| Parameter | Value | Notes |
|-----------|-------|-------|
| S_cut | {thresholds["s_cut"]:.4f} | 95th percentile of null S |
| C_cut | {thresholds["c_cut"]:.4f} | 30th percentile of null C |
| N samples | {thresholds["n_samples"]} | Valid null samples |

Genes with S > S_cut are likely to have true spatial structure.
Genes with C < C_cut are considered "low coverage".
These thresholds define the quadrant boundaries for classification.

## Abstention Summary

Total runs: {len(results_df)}
Abstained: {results_df['abstain_flag'].sum()}
Finite p-values: {len(valid_p)}

## Usage

Load thresholds in other scripts:
```python
import json
with open('calibration_thresholds.json') as f:
    thresholds = json.load(f)
s_cut = thresholds['s_cut']
c_cut = thresholds['c_cut']
```
"""

    with open(output_dir / "report.md", "w") as f:
        f.write(report)

    # Finalize contract
    finalize_contract(
        output_dir=output_dir,
        benchmark_name="null_calibration",
        contract_config=contract_cfg,
        session_id=session_id,
        mode=args.mode,
        n_conditions=len(mode_cfg["null_types"]) * mode_cfg["n_reps"],
        n_completed=len(results_df),
        runtime_seconds=elapsed,
    )

    print(f"\n✓ Calibration complete! Outputs in {output_dir}")

    return thresholds


def plot_null_calibration_diagnostics(output_dir, results_df, valid_p, thresholds, mode):
    """Plot QQ, FPR, and p-value histograms with abstention handling.

    Args:
        output_dir: Output directory for diagnostics.
        results_df: Runs DataFrame with p-values and metadata.
        valid_p: Array of finite p-values.
        thresholds: Thresholds derived from null runs.
        mode: Benchmark mode name.
    """

    n_abstained = results_df["abstain_flag"].sum()
    n_total = len(results_df)

    if len(valid_p) == 0:
        # All abstained case: create explicit diagnostic figure
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.text(
            0.5,
            0.5,
            f"ALL ABSTAINED\n\nTotal runs: {n_total}\nAbstained: {n_abstained}\nFinite p-values: 0",
            ha="center",
            va="center",
            fontsize=14,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        fig.savefig(
            output_dir / "fig_null_calibration_diagnostics.png", dpi=150, bbox_inches="tight"
        )
        plt.close(fig)
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # QQ plot
    theoretical = np.linspace(0, 1, len(valid_p))
    observed = np.sort(valid_p)
    axes[0].plot(theoretical, observed, "o", alpha=0.6, markersize=4)
    axes[0].plot([0, 1], [0, 1], "r--", label="Uniform")
    axes[0].set_xlabel("Expected (Uniform)")
    axes[0].set_ylabel("Observed p-value")
    axes[0].set_title("QQ Plot (Null p-values)")
    axes[0].legend()

    # FPR at various alphas
    alphas = [0.01, 0.05, 0.10, 0.20]
    fprs = [np.mean(valid_p <= alpha) for alpha in alphas]
    axes[1].plot(alphas, fprs, "o-", label="Observed FPR")
    axes[1].plot([0, 0.25], [0, 0.25], "r--", label="Nominal")
    axes[1].set_xlabel("Alpha")
    axes[1].set_ylabel("False Positive Rate")
    axes[1].set_title("FPR Calibration")
    axes[1].legend()

    # P-value histogram
    axes[2].hist(valid_p, bins=20, alpha=0.7, edgecolor="black", density=True)
    axes[2].axhline(1.0, color="red", linestyle="--", label="Uniform")
    axes[2].set_xlabel("p-value")
    axes[2].set_ylabel("Density")
    axes[2].set_title(f"P-value Distribution (N={len(valid_p)})")
    axes[2].legend()

    plt.tight_layout()
    fig.savefig(output_dir / "fig_null_calibration_diagnostics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Run null calibration for threshold derivation")
    parser.add_argument(
        "--mode", type=str, choices=["quick", "validation", "publication"], default="quick"
    )
    parser.add_argument(
        "--outdir", type=str, default=str(ROOT / "outputs" / "calibration_thresholds")
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    run_calibration(args)


if __name__ == "__main__":
    main()
