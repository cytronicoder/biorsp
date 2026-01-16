"""Debug script for selection bias analysis.

Tests that empty_fg_policy='zero' correctly handles empty sectors and avoids
selection bias. Compares global_rim vs wedge_rim scenarios.

Usage:
    python dev/debug/debug_selection_bias.py
    python dev/debug/debug_selection_bias.py --smoke --outdir /tmp/test

Requires:
    Package installation: pip install -e .
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from biorsp.core.engine import compute_rsp_radar
    from biorsp.core.geometry import polar_coordinates
    from biorsp.core.summaries import compute_scalar_summaries
    from biorsp.utils.config import BioRSPConfig
except ImportError as e:
    print("ERROR: Cannot import biorsp. Please install the package first:")
    print("  pip install -e .")
    print(f"Details: {e}")
    exit(1)


def make_synthetic_data(scenario="wedge"):
    np.random.seed(42)
    n_bg = 2000
    n_fg = 500

    r_bg = np.sqrt(np.random.uniform(0, 1, n_bg))
    theta_bg = np.random.uniform(-np.pi, np.pi, n_bg)
    bg_x = r_bg * np.cos(theta_bg)
    bg_y = r_bg * np.sin(theta_bg)

    if scenario == "wedge_rim":
        r_fg = np.sqrt(np.random.uniform(0.8, 1.0, n_fg))
        theta_fg = np.random.normal(np.pi / 2, 0.2, n_fg)

    elif scenario == "global_rim":
        r_fg = np.sqrt(np.random.uniform(0.8, 1.0, n_fg))
        theta_fg = np.random.uniform(-np.pi, np.pi, n_fg)

    elif scenario == "null":
        r_fg = np.sqrt(np.random.uniform(0, 1, n_fg))
        theta_fg = np.random.uniform(-np.pi, np.pi, n_fg)

    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    fg_x = r_fg * np.cos(theta_fg)
    fg_y = r_fg * np.sin(theta_fg)

    z = np.column_stack([np.concatenate([bg_x, fg_x]), np.concatenate([bg_y, fg_y])])
    is_fg = np.concatenate([np.zeros(n_bg, dtype=bool), np.ones(n_fg, dtype=bool)])

    return z, is_fg


def run_debug_session(outdir=None, smoke=False):
    """Run selection bias debug session.

    Key test: Under empty_fg_policy='zero', wedge_rim should have smaller
    |R_mean| than global_rim, since fewer sectors have foreground.
    This verifies the selection bias fix.

    Args:
        outdir: Output directory. If None, uses scripts/debug/
        smoke: If True, run in fast smoke test mode (fewer scenarios/policies)
    """
    print("Running Selection Bias Debug Session...")
    print("Testing empty_fg_policy behavior for rim patterns.\n")

    debug_dir = Path(outdir) if outdir else Path("scripts") / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    if smoke:
        scenarios = ["global_rim", "wedge_rim"]  # Skip null in smoke mode
        policies = ["zero"]  # Only test the fixed policy
    else:
        scenarios = ["global_rim", "wedge_rim", "null"]
        policies = ["nan", "zero"]

    results_table = []

    for sc in scenarios:
        print(f"\n--- Running Scenario: {sc} ---")
        z, y = make_synthetic_data(sc)
        v = np.array([0.0, 0.0])

        r_raw, theta = polar_coordinates(z, v)

        r_med = np.median(r_raw)
        r_iqr = np.percentile(r_raw, 75) - np.percentile(r_raw, 25)
        if r_iqr < 1e-8:
            r_iqr = 1.0
        r = (r_raw - r_med) / r_iqr

        B = 36
        delta_deg = 180.0

        for policy in policies:
            print(f"  Policy: {policy}")
            config = BioRSPConfig(B=B, delta_deg=delta_deg, empty_fg_policy=policy)

            res = compute_rsp_radar(r, theta, y, config=config, debug=True)
            summ = compute_scalar_summaries(res)

            results_table.append(
                {
                    "Scenario": sc,
                    "Policy": policy,
                    "R_mean": summ.r_mean,
                    "Anisotropy": summ.anisotropy,
                    "Cov_Geom": summ.coverage_geom,
                    "Cov_FG": summ.coverage_fg,
                    "Valid_Sectors": summ.m_valid_sectors,
                }
            )

            if policy == "zero":
                centers_deg = np.degrees(res.centers)

                fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

                ax = axes[0]

                ax.plot(centers_deg, res.rsp, "o-", label="RSP")
                ax.set_ylabel("RSP")
                ax.set_title(f"RSP Profile ({sc}, policy={policy})")
                ax.grid(True)

                ax = axes[1]
                ax.plot(centers_deg, res.counts_fg, "g.-", label="nF (Foreground)")
                ax.plot(centers_deg, res.counts_bg, "k--", label="nB (Background)")
                ax.set_ylabel("Count")
                ax.set_title("Sector Counts")
                ax.legend()
                ax.grid(True)

                ax = axes[2]
                valid_mask = ~np.isnan(res.rsp)
                ax.plot(
                    centers_deg, valid_mask.astype(int), "r-", drawstyle="steps-mid", label="Valid"
                )
                ax.set_ylabel("Valid (1/0)")
                ax.set_xlabel("Theta (degrees)")
                ax.set_ylim(-0.1, 1.1)
                ax.grid(True)

                outpath = debug_dir / f"debug_selection_bias_{sc}_{policy}.png"
                plt.tight_layout()
                plt.savefig(outpath)
                print(f"    Saved plot to {outpath}")

    print("\n=== Summary Results ===")
    print("Expected: wedge_rim should have lower |R_mean| than global_rim under policy='zero'")
    print("(because fewer sectors have foreground, avoiding selection bias)\n")
    print(
        f"{'Scenario':<15} | {'Policy':<6} | {'R_mean':>8} | {'Aniso':>8} | {'Cov_Geom':>8} | {'Cov_FG':>6} | {'Valid':>5}"
    )
    print("-" * 75)
    for row in results_table:
        print(
            f"{row['Scenario']:<15} | {row['Policy']:<6} | {row['R_mean']:8.3f} | {row['Anisotropy']:8.3f} | {row['Cov_Geom']:8.2f} | {row['Cov_FG']:6.2f} | {row['Valid_Sectors']:5d}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Debug selection bias with synthetic data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run in fast smoke test mode (fewer scenarios/policies)",
    )
    parser.add_argument("--outdir", type=str, help="Output directory for figures")
    args = parser.parse_args()

    run_debug_session(outdir=args.outdir, smoke=args.smoke)
