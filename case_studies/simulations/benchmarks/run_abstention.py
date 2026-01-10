#!/usr/bin/env python3
"""
Abstention/Failure Mode Evaluation.

Tests that BioRSP correctly abstains (returns NaN or flags) under stress conditions:
- Extremely low coverage (1-3% prevalence)
- Very small sample sizes
- Disconnected geometries

Usage:
    python run_abstention.py --mode quick --outdir outputs/abstention --seed 42

Outputs:
    - fig_abstention.png
    - abstention_summary.csv
    - report.md
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from biorsp import BioRSPConfig


def run_abstention(args):
    """Run abstention evaluation."""
    from simlib import (
        datasets,
        expression,
        io,
        plotting,
        rng as rng_module,
        scoring,
        shapes,
    )

    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Abstention/Failure Mode Evaluation")
    print(f"Mode: {args.mode}")
    print("=" * 60)

    start_time = time.time()
    gen = rng_module.make_rng(args.seed, "abstention")

    config = BioRSPConfig(
        B=72,
        delta_deg=60.0,
        n_permutations=0,
        qc_mode="principled",
    )

    # Define stress conditions
    conditions = []

    # Condition 1: Normal (baseline)
    conditions.append(
        {
            "name": "Normal (N=2000, C=30%)",
            "n_cells": 2000,
            "target_coverage": 0.30,
            "shape": "disk",
            "expected_abstain": False,
        }
    )

    # Condition 2: Low coverage
    conditions.append(
        {
            "name": "Low Coverage (C=5%)",
            "n_cells": 2000,
            "target_coverage": 0.05,
            "shape": "disk",
            "expected_abstain": False,  # Should still work
        }
    )

    # Condition 3: Very low coverage
    conditions.append(
        {
            "name": "Very Low Coverage (C=2%)",
            "n_cells": 2000,
            "target_coverage": 0.02,
            "shape": "disk",
            "expected_abstain": True,  # Should abstain
        }
    )

    # Condition 4: Extremely low coverage
    conditions.append(
        {
            "name": "Extreme Low Coverage (C=1%)",
            "n_cells": 2000,
            "target_coverage": 0.01,
            "shape": "disk",
            "expected_abstain": True,
        }
    )

    # Condition 5: Small N
    conditions.append(
        {
            "name": "Small N (N=200)",
            "n_cells": 200,
            "target_coverage": 0.30,
            "shape": "disk",
            "expected_abstain": False,  # May still work
        }
    )

    # Condition 6: Very small N
    conditions.append(
        {
            "name": "Very Small N (N=50)",
            "n_cells": 50,
            "target_coverage": 0.30,
            "shape": "disk",
            "expected_abstain": True,
        }
    )

    # Condition 7: Disconnected geometry
    conditions.append(
        {
            "name": "Disconnected Geometry",
            "n_cells": 2000,
            "target_coverage": 0.30,
            "shape": "disconnected_blobs",
            "expected_abstain": False,  # May have issues but shouldn't abstain
        }
    )

    # Run each condition
    n_reps = 5 if args.mode == "quick" else 20
    results = []

    for cond in conditions:
        print(f"\nTesting: {cond['name']}...")

        abstain_count = 0
        nan_score_count = 0
        valid_scores = []

        for rep in range(n_reps):
            rep_gen = rng_module.make_rng(args.seed + rep, "abstention", cond["name"])

            # Generate data
            coords, _ = shapes.generate_coords(cond["shape"], cond["n_cells"], rep_gen)
            libsize = expression.simulate_library_size(
                cond["n_cells"], rep_gen, model="lognormal", params={"mean": 2000, "sigma": 0.5}
            )

            # Generate expression with target coverage
            counts, meta = expression.generate_expression_targeted(
                coords=coords,
                libsize=libsize,
                rng=rep_gen,
                pattern="wedge",
                target_coverage=cond["target_coverage"],
                pattern_params={"angle_center": 0, "width_rad": np.pi / 4},
            )

            # Package and score
            adata = datasets.package_as_anndata(
                coords, counts[:, None], var_names=["test_gene"], embedding_key="X_sim"
            )

            try:
                scores_df = scoring.score_dataset(
                    adata, genes=["test_gene"], config=config, embedding_key="X_sim"
                )

                if len(scores_df) == 0:
                    abstain_count += 1
                else:
                    row = scores_df.iloc[0]
                    if row["abstain_flag"]:
                        abstain_count += 1
                    if pd.isna(row["spatial_score"]):
                        nan_score_count += 1
                    else:
                        valid_scores.append(row["spatial_score"])

            except Exception as e:
                print(f"  Rep {rep} error: {e}")
                abstain_count += 1

        abstain_rate = abstain_count / n_reps
        nan_rate = nan_score_count / n_reps
        mean_score = np.mean(valid_scores) if valid_scores else np.nan

        results.append(
            {
                "condition": cond["name"],
                "n_cells": cond["n_cells"],
                "target_coverage": cond["target_coverage"],
                "shape": cond["shape"],
                "expected_abstain": cond["expected_abstain"],
                "abstention_rate": abstain_rate,
                "nan_rate": nan_rate,
                "mean_score": mean_score,
                "n_reps": n_reps,
                "n_valid": len(valid_scores),
            }
        )

        status = "✓" if (abstain_rate > 0.5) == cond["expected_abstain"] else "?"
        print(f"   {status} Abstention rate: {abstain_rate:.0%}")

    results_df = pd.DataFrame(results)

    # Save results
    results_df.to_csv(output_dir / "abstention_summary.csv", index=False)

    # Generate figure
    fig = plotting.plot_abstention_summary(
        results_df[["condition", "abstention_rate", "n_reps"]].rename(
            columns={"n_reps": "n_samples"}
        ),
        title="Abstention Behavior Under Stress",
    )
    io.save_figure(fig, output_dir, "fig_abstention.png")
    plt.close(fig)

    # Report
    elapsed = time.time() - start_time

    report = f"""# Abstention Evaluation Report

Mode: {args.mode}
Runtime: {elapsed:.1f}s
Replicates per condition: {n_reps}

## Summary

BioRSP should abstain (return NaN or flag) when there is insufficient data
to compute reliable spatial scores.

## Results

| Condition | Abstention Rate | Expected | Status |
|-----------|-----------------|----------|--------|
"""

    for _, row in results_df.iterrows():
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

## Expected Behavior

1. Normal conditions (N≥200, C≥5%): Should NOT abstain
2. Very low coverage (C<3%): SHOULD abstain
3. Very small N (N<100): SHOULD abstain
4. Disconnected geometry: May produce unreliable scores but abstention is optional
"""

    with open(output_dir / "report.md", "w") as f:
        f.write(report)

    print(f"\n✓ Abstention evaluation complete! Outputs in {output_dir}")

    return results_df


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
