"""
Validation script for BioRSP p-value calibration.
Generates null datasets and checks if p-values are uniformly distributed.
"""

import argparse
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from biorsp import (
    BioRSPConfig,
    compute_p_value,
    geometric_median,
    polar_coordinates,
)
from biorsp.simulations.generator import _sample_annulus, _sample_blob, _sample_disk

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def simulate_null_gene(n_cells: int, n_fg: int, seed: int = None) -> np.ndarray:
    """Simulate a null gene by randomly assigning foreground labels."""
    rng = np.random.default_rng(seed)
    y = np.zeros(n_cells)
    fg_idx = rng.choice(n_cells, size=n_fg, replace=False)
    y[fg_idx] = 1.0
    return y


def run_calibration_test(
    shape: str = "disk",
    n_cells: int = 2000,
    n_genes: int = 100,
    n_perm: int = 200,
    perm_mode: str = "radial",
    seed: int = 42,
    n_workers: int = 1,
):
    """Run calibration test for a given shape and permutation mode."""
    np.random.seed(seed)

    # 1. Generate geometry
    if shape == "disk":
        coords = _sample_disk(n_cells)
    elif shape == "annulus":
        coords = _sample_annulus(n_cells)
    elif shape == "blob":
        coords = _sample_blob(n_cells)
    else:
        raise ValueError(f"Unknown shape: {shape}")

    v, _, _ = geometric_median(coords)
    r, theta = polar_coordinates(coords, v)

    # 2. Configure BioRSP
    config = BioRSPConfig(
        perm_mode=perm_mode,
        n_permutations=n_perm,
        seed=seed,
    )

    p_values = []

    logger.info(f"Running calibration test: shape={shape}, mode={perm_mode}, n_genes={n_genes}")

    for i in range(n_genes):
        # Simulate null gene with ~10% foreground
        n_fg = int(0.1 * n_cells)
        y = simulate_null_gene(n_cells, n_fg, seed=seed + i)

        res = compute_p_value(
            r=r,
            theta=theta,
            y=y,
            config=config,
            n_perm=n_perm,
            seed=seed + i,
            n_workers=n_workers,
            show_progress=False,
        )
        if np.isfinite(res.p_value):
            p_values.append(res.p_value)

    p_values = np.array(p_values)
    return p_values


def plot_results(p_values, title, out_path):
    """Generate QQ plot and histogram."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram
    axes[0].hist(p_values, bins=20, density=True, alpha=0.7, color="skyblue", edgecolor="black")
    axes[0].axhline(1.0, color="red", linestyle="--")
    axes[0].set_title(f"P-value Histogram\n{title}")
    axes[0].set_xlabel("p-value")
    axes[0].set_ylabel("Density")

    # QQ Plot
    sorted_p = np.sort(p_values)
    expected_p = np.linspace(0, 1, len(p_values) + 1)[1:]
    axes[1].scatter(expected_p, sorted_p, s=10, alpha=0.6)
    axes[1].plot([0, 1], [0, 1], color="red", linestyle="--")
    axes[1].set_title("QQ Plot")
    axes[1].set_xlabel("Expected Uniform")
    axes[1].set_ylabel("Observed p-value")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Validate BioRSP p-value calibration.")
    parser.add_argument("--n_genes", type=int, default=50, help="Number of null genes")
    parser.add_argument("--n_cells", type=int, default=1000, help="Number of cells")
    parser.add_argument("--n_perm", type=int, default=100, help="Number of permutations")
    parser.add_argument(
        "--modes", nargs="+", default=["radial", "joint", "none"], help="Permutation modes to test"
    )
    parser.add_argument("--shapes", nargs="+", default=["disk", "blob"], help="Shapes to test")
    parser.add_argument(
        "--outdir", type=str, default="results/calibration", help="Output directory"
    )
    parser.add_argument("--n_workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    results = []

    for shape in args.shapes:
        for mode in args.modes:
            p_values = run_calibration_test(
                shape=shape,
                n_cells=args.n_cells,
                n_genes=args.n_genes,
                n_perm=args.n_perm,
                perm_mode=mode,
                seed=args.seed,
                n_workers=args.n_workers,
            )

            # Metrics
            fpr_05 = np.mean(p_values <= 0.05)
            fpr_01 = np.mean(p_values <= 0.01)
            ks_stat, ks_p = stats.kstest(p_values, "uniform")

            logger.info(f"Results for {shape}/{mode}:")
            logger.info(f"  FPR@0.05: {fpr_05:.3f}")
            logger.info(f"  FPR@0.01: {fpr_01:.3f}")
            logger.info(f"  KS stat: {ks_stat:.3f} (p={ks_p:.3f})")

            results.append(
                {
                    "shape": shape,
                    "mode": mode,
                    "fpr_05": fpr_05,
                    "fpr_01": fpr_01,
                    "ks_stat": ks_stat,
                    "ks_p": ks_p,
                }
            )

            plot_results(
                p_values,
                f"Shape: {shape}, Mode: {mode}",
                os.path.join(args.outdir, f"calibration_{shape}_{mode}.png"),
            )

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(args.outdir, "calibration_summary.csv"), index=False)
    print("\nCalibration Summary:")
    print(df)

    # Acceptance criteria check
    for _, row in df.iterrows():
        if row["mode"] != "none":
            if not (0.02 <= row["fpr_05"] <= 0.08):
                print(
                    f"WARNING: FPR@0.05 for {row['shape']}/{row['mode']} is {row['fpr_05']:.3f}, outside [0.02, 0.08]"
                )


if __name__ == "__main__":
    main()
