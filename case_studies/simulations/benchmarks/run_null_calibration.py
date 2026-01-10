#!/usr/bin/env python3
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
"""

import argparse
import json
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
    """Run null calibration."""
    from simlib import (
        datasets,
        expression,
        io,
        metrics,
        rng,
        scoring,
        shapes,
    )

    mode_cfg = MODE_CONFIGS[args.mode]

    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
        qc_mode="principled",
    )

    all_results = []

    for null_type in mode_cfg["null_types"]:
        print(f"\nRunning {null_type} null ({mode_cfg['n_reps']} reps)...")

        for rep in range(mode_cfg["n_reps"]):
            gen = rng.make_rng(args.seed + rep, "calibration", null_type)

            # Generate data
            coords, _ = shapes.generate_coords("disk", mode_cfg["n_cells"], gen)
            libsize = expression.simulate_library_size(
                mode_cfg["n_cells"], gen, model="lognormal", params={"mean": 2000, "sigma": 0.5}
            )

            # Generate null expression
            counts, _ = expression.generate_confounded_null(
                coords, libsize, gen, null_type=null_type, params={}
            )

            # Package and score
            adata = datasets.package_as_anndata(
                coords, counts[:, None], var_names=["null_gene"], embedding_key="X_sim"
            )

            try:
                scores_df = scoring.score_dataset(
                    adata, genes=["null_gene"], config=config, embedding_key="X_sim"
                )

                if len(scores_df) > 0:
                    row = scores_df.iloc[0]
                    all_results.append(
                        {
                            "null_type": null_type,
                            "replicate": rep,
                            "spatial_score": row["spatial_score"],
                            "coverage_expr": row["coverage_expr"],
                            "p_value": row.get("p_value", np.nan),
                            "abstain_flag": row["abstain_flag"],
                        }
                    )
            except Exception as e:
                print(f"  Warning: Rep {rep} failed: {e}")

    results_df = pd.DataFrame(all_results)

    # Derive thresholds
    valid_s = results_df.loc[~results_df["abstain_flag"], "spatial_score"].values
    valid_c = results_df.loc[~results_df["abstain_flag"], "coverage_expr"].values

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

    # KS test for p-value uniformity (if p-values computed)
    if "p_value" in results_df.columns:
        p_vals = results_df["p_value"].dropna().values
        if len(p_vals) > 10:
            ks_stat, ks_pval = metrics.ks_uniform(p_vals)
            thresholds["ks_stat"] = float(ks_stat)
            thresholds["ks_pval"] = float(ks_pval)
            print(f"  KS test for uniformity: stat={ks_stat:.4f}, p={ks_pval:.4f}")

    # Save outputs
    results_df.to_csv(output_dir / "null_distribution.csv", index=False)

    with open(output_dir / "thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # S distribution
    axes[0].hist(valid_s, bins=30, alpha=0.7, edgecolor="black")
    axes[0].axvline(
        thresholds["s_cut"],
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"S_cut = {thresholds['s_cut']:.3f}",
    )
    axes[0].set_xlabel("Spatial Score (S)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Null Distribution of S")
    axes[0].legend()

    # C distribution
    axes[1].hist(valid_c, bins=30, alpha=0.7, edgecolor="black", color="green")
    axes[1].axvline(
        thresholds["c_cut"],
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"C_cut = {thresholds['c_cut']:.3f}",
    )
    axes[1].set_xlabel("Coverage (C)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Null Distribution of C")
    axes[1].legend()

    plt.tight_layout()
    io.save_figure(fig, output_dir, "fig_null_distributions.png")
    plt.close(fig)

    # Report
    elapsed = time.time() - start_time

    report = f"""# Null Calibration Report

Mode: {args.mode}
Runtime: {elapsed:.1f}s

## Derived Thresholds

| Parameter | Value | Notes |
|-----------|-------|-------|
| S_cut | {thresholds['s_cut']:.4f} | 95th percentile of null S |
| C_cut | {thresholds['c_cut']:.4f} | 30th percentile of null C |
| N samples | {thresholds['n_samples']} | Valid null samples |

## Interpretation

- Genes with S > S_cut are likely to have true spatial structure
- Genes with C < C_cut are considered "low coverage"
- These thresholds define the quadrant boundaries for classification

## Usage

Load thresholds in other scripts:
```python
import json
with open('thresholds.json') as f:
    thresholds = json.load(f)
s_cut = thresholds['s_cut']
c_cut = thresholds['c_cut']
```
"""

    with open(output_dir / "report.md", "w") as f:
        f.write(report)

    print(f"\n✓ Calibration complete! Outputs in {output_dir}")

    return thresholds


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
