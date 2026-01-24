"""
Abstention/Failure Mode Evaluation.

Tests that BioRSP correctly abstains (returns NaN or flags) under stress conditions:
- Extremely low coverage (1-3% prevalence)
- Very small sample sizes
- Disconnected geometries

Usage:
    python run_abstention.py --mode quick --outdir outputs/abstention --seed 42

Outputs:
    - runs.csv, summary.csv, manifest.json, report.md
    - fig_abstention_summary.png
    - Debug panels per condition
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Local helper imports are performed inside `run_abstention()` to avoid module-level
# imports after side-effectful code (E402).


def run_abstention(args):
    """Run abstention evaluation.

    Args:
        args: Parsed CLI arguments.
    """
    from analysis.benchmarks.simlib.io_contract import BenchmarkContractConfig, init_run_dir
    from analysis.benchmarks.simlib.runner_harness import finalize_contract, normalize_scores_df
    from biorsp import BioRSPConfig
    from biorsp.plotting.standard import make_standard_plot_set
    from biorsp.simulations import datasets, expression, rng as rng_module, scoring, shapes

    output_dir = Path(args.outdir)
    contract_cfg = BenchmarkContractConfig(require_runs_csv=True, require_summary_csv=True)
    session_id = init_run_dir(output_dir, clear_existing=True, contract_config=contract_cfg)

    print("=" * 60)
    print("Abstention/Failure Mode Evaluation")
    print(f"Mode: {args.mode}")
    print("=" * 60)

    start_time = time.time()

    config = BioRSPConfig(
        B=72,
        delta_deg=60.0,
        n_permutations=0,
    )

    conditions = []
    conditions.append(
        {
            "name": "Normal (N=2000, C=30%)",
            "n_cells": 2000,
            "target_coverage": 0.30,
            "shape": "disk",
            "expected_abstain": False,
        }
    )

    conditions.append(
        {
            "name": "Low Coverage (C=5%)",
            "n_cells": 2000,
            "target_coverage": 0.05,
            "shape": "disk",
            "expected_abstain": False,
        }
    )

    conditions.append(
        {
            "name": "Very Low Coverage (C=2%)",
            "n_cells": 2000,
            "target_coverage": 0.02,
            "shape": "disk",
            "expected_abstain": True,
        }
    )

    conditions.append(
        {
            "name": "Extreme Low Coverage (C=1%)",
            "n_cells": 2000,
            "target_coverage": 0.01,
            "shape": "disk",
            "expected_abstain": True,
        }
    )

    conditions.append(
        {
            "name": "Small N (N=200)",
            "n_cells": 200,
            "target_coverage": 0.30,
            "shape": "disk",
            "expected_abstain": False,
        }
    )

    conditions.append(
        {
            "name": "Very Small N (N=50)",
            "n_cells": 50,
            "target_coverage": 0.30,
            "shape": "disk",
            "expected_abstain": True,
        }
    )

    conditions.append(
        {
            "name": "Disconnected Geometry",
            "n_cells": 2000,
            "target_coverage": 0.30,
            "shape": "disconnected_blobs",
            "expected_abstain": False,
        }
    )

    n_reps = 5 if args.mode == "quick" else 20
    all_runs = []

    for cond in conditions:
        print(f"\nTesting: {cond['name']}...")

        for rep in range(n_reps):
            rep_gen = rng_module.make_rng(args.seed + rep, "abstention", cond["name"])

            coords, _ = shapes.generate_coords(cond["shape"], cond["n_cells"], rep_gen)
            libsize = expression.simulate_library_size(
                cond["n_cells"], rep_gen, model="lognormal", params={"mean": 2000, "sigma": 0.5}
            )

            counts, meta = expression.generate_expression_targeted(
                coords=coords,
                libsize=libsize,
                rng=rep_gen,
                pattern="wedge",
                target_coverage=cond["target_coverage"],
                pattern_params={"angle_center": 0, "width_rad": np.pi / 4},
            )

            adata = datasets.package_as_anndata(
                coords, counts[:, None], var_names=["test_gene"], embedding_key="X_sim"
            )

            try:
                scores_df = scoring.score_dataset(
                    adata, genes=["test_gene"], config=config, embedding_key="X_sim"
                )

                scores_df = normalize_scores_df(scores_df)

                if len(scores_df) == 0:
                    # No results returned
                    all_runs.append(
                        {
                            "benchmark": "abstention",
                            "condition": cond["name"],
                            "n_cells": cond["n_cells"],
                            "target_coverage": cond["target_coverage"],
                            "shape": cond["shape"],
                            "expected_abstain": cond["expected_abstain"],
                            "replicate": rep,
                            "seed": args.seed + rep,
                            "Spatial_Bias_Score": np.nan,
                            "Coverage": np.nan,
                            "abstain_flag": True,
                            "abstain_reason": "no_results",
                        }
                    )
                else:
                    row = scores_df.iloc[0]
                    all_runs.append(
                        {
                            "benchmark": "abstention",
                            "condition": cond["name"],
                            "n_cells": cond["n_cells"],
                            "target_coverage": cond["target_coverage"],
                            "shape": cond["shape"],
                            "expected_abstain": cond["expected_abstain"],
                            "replicate": rep,
                            "seed": args.seed + rep,
                            "Spatial_Bias_Score": row.get("Spatial_Bias_Score", np.nan),
                            "Coverage": row.get("Coverage", np.nan),
                            "abstain_flag": row.get("abstain_flag", True),
                            "abstain_reason": row.get("abstain_reason", ""),
                        }
                    )

            except Exception as e:
                print(f"  Rep {rep} error: {e}")
                all_runs.append(
                    {
                        "benchmark": "abstention",
                        "condition": cond["name"],
                        "n_cells": cond["n_cells"],
                        "target_coverage": cond["target_coverage"],
                        "shape": cond["shape"],
                        "expected_abstain": cond["expected_abstain"],
                        "replicate": rep,
                        "seed": args.seed + rep,
                        "Spatial_Bias_Score": np.nan,
                        "Coverage": np.nan,
                        "abstain_flag": True,
                        "abstain_reason": str(e),
                    }
                )

        cond_df = pd.DataFrame([r for r in all_runs if r["condition"] == cond["name"]])
        abstain_rate = cond_df["abstain_flag"].mean()
        status = "✓" if (abstain_rate > 0.5) == cond["expected_abstain"] else "?"
        print(f"   {status} Abstention rate: {abstain_rate:.0%}")

    runs_df = pd.DataFrame(all_runs)

    # Summary per condition
    summary_rows = []
    for cond in conditions:
        cond_df = runs_df[runs_df["condition"] == cond["name"]]
        abstain_rate = cond_df["abstain_flag"].mean()
        finite_scores = cond_df.loc[~cond_df["abstain_flag"], "Spatial_Bias_Score"]
        summary_rows.append(
            {
                "condition": cond["name"],
                "n_cells": cond["n_cells"],
                "target_coverage": cond["target_coverage"],
                "expected_abstain": cond["expected_abstain"],
                "abstention_rate": abstain_rate,
                "n_reps": len(cond_df),
                "n_valid": len(finite_scores),
                "mean_score": finite_scores.mean() if len(finite_scores) > 0 else np.nan,
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    # Write outputs
    runs_df.to_csv(output_dir / "runs.csv", index=False)
    summary_df.to_csv(output_dir / "summary.csv", index=False)

    # Plot abstention summary
    plot_abstention_summary(output_dir, summary_df)

    # Debug panels per condition (small example from each)
    for cond in conditions[:3]:  # First 3 for quick mode
        cond_df = runs_df[runs_df["condition"] == cond["name"]]
        if len(cond_df[~cond_df["abstain_flag"]]) > 0:
            cond_outdir = output_dir / f"debug_{cond['name'].replace(' ', '_')}"
            cond_outdir.mkdir(exist_ok=True)
            make_standard_plot_set(
                scores_df=cond_df[~cond_df["abstain_flag"]],
                outdir=cond_outdir,
                truth_col=None,
            )

    elapsed = time.time() - start_time

    report = f"""# Abstention Evaluation Report

Mode: {args.mode}
Runtime: {elapsed:.1f}s
Session ID: {session_id}
Replicates per condition: {n_reps}

## Purpose

BioRSP should abstain (return NaN or flag) when there is insufficient data
to compute reliable spatial scores.

## Results

| Condition | Abstention Rate | Expected | Status |
|-----------|-----------------|----------|--------|
"""

    for _, row in summary_df.iterrows():
        expected_str = "High" if row["expected_abstain"] else "Low"
        actual_high = row["abstention_rate"] > 0.5
        status = "✓" if actual_high == row["expected_abstain"] else "✗"
        report += (
            f"| {row['condition']} | {row['abstention_rate']:.0%} | {expected_str} | {status} |\n"
        )

    report += """

## Interpretation

- **Low coverage stress**: At very low coverage (1-2%), most cells don't express
  the gene, making spatial analysis unreliable. BioRSP should abstain.
- **Small N stress**: With very few cells (<100), statistical estimates become
  unreliable. BioRSP should abstain or flag.
- **Geometry stress**: Disconnected geometries may confuse center-based methods
  but shouldn't necessarily cause abstention.

## Criteria

1. Normal conditions (N≥200, C≥5%): Should NOT abstain
2. Very low coverage (C<3%): SHOULD abstain
3. Very small N (N<100): SHOULD abstain
4. Disconnected geometry: May produce unreliable scores but abstention is optional
"""

    with open(output_dir / "report.md", "w") as f:
        f.write(report)

    # Finalize contract
    finalize_contract(
        output_dir=output_dir,
        benchmark_name="abstention",
        contract_config=contract_cfg,
        session_id=session_id,
        mode=args.mode,
        n_conditions=len(conditions) * n_reps,
        n_completed=len(runs_df),
        runtime_seconds=elapsed,
    )

    print(f"\n✓ Abstention evaluation complete! Outputs in {output_dir}")

    return runs_df


def plot_abstention_summary(output_dir, summary_df):
    """Plot abstention rate summary.

    Args:
        output_dir: Output directory for the plot.
        summary_df: Summary DataFrame with abstention metrics.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    x_pos = np.arange(len(summary_df))
    colors = ["green" if not exp else "orange" for exp in summary_df["expected_abstain"]]

    ax.bar(x_pos, summary_df["abstention_rate"], color=colors, alpha=0.7, edgecolor="black")
    ax.axhline(0.5, color="red", linestyle="--", label="50% threshold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(summary_df["condition"], rotation=45, ha="right")
    ax.set_ylabel("Abstention Rate")
    ax.set_title("Abstention Behavior Under Stress")
    ax.set_ylim(0, 1)
    ax.legend()

    plt.tight_layout()
    fig.savefig(output_dir / "fig_abstention_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Run abstention evaluation")
    parser.add_argument(
        "--mode", type=str, choices=["quick", "validation", "publication"], default="quick"
    )
    parser.add_argument("--outdir", type=str, default=str(ROOT / "outputs" / "abstention"))
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    run_abstention(args)


if __name__ == "__main__":
    main()
