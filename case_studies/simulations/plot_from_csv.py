"""
DEPRECATED: Thin wrapper around simlib.plotting for legacy CSV plotting.

This module provides backward-compatible plotting functions for simulation CSVs.
New code should import directly from simlib.plotting.

Usage:
    python plot_from_csv.py <csv_path> <output_dir> --plot_type <calibration|power|robustness>
    OR (new style):
    python plot_from_csv.py --input-dir <results_dir> --outdir <output_dir> --which all
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Path bootstrap
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)


def _save_fig(fig, path):
    """Save figure to path as PNG and close."""
    fig.savefig(f"{path}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _set_common_axes(ax, xlabel, ylabel):
    """Set common axis properties."""
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.3)


def plot_calibration(csv_path: Path, outdir: Path):
    """Plot calibration results (QQ plot)."""
    from simlib import io, metrics, plotting

    df = pd.read_csv(csv_path)

    if "p_value" in df.columns:
        p_values = df["p_value"].dropna().values
        expected, observed = metrics.qq_quantiles(p_values)
        fig = plotting.plot_qq(expected, observed, title="Calibration: QQ Plot")
        io.save_figure(fig, Path(outdir), "calibration_qq.png")
    else:
        raise ValueError("CSV must contain 'p_value' column")


def plot_power(csv_path: Path, outdir: Path):
    """Plot power curves."""
    from simlib import io, plotting

    df = pd.read_csv(csv_path)

    if "N" in df.columns and "power" in df.columns:
        fig = plotting.plot_power_curve(df, x_var="N", title="Power vs Sample Size")
        io.save_figure(fig, Path(outdir), "power_vs_N.png")
    elif "N" in df.columns and "power_mean" in df.columns:
        # Legacy name support
        df = df.rename(columns={"power_mean": "power"})
        fig = plotting.plot_power_curve(df, x_var="N", title="Power vs Sample Size")
        io.save_figure(fig, Path(outdir), "power_vs_N.png")
    else:
        logger.warning(
            f"CSV {csv_path} does not contain expected power columns. Found: {df.columns.tolist()}"
        )


# Alias for newer main
plot_power_vs_N = plot_power


def plot_robustness(csv_path: Path, outdir: Path):
    """Plot robustness or parameter sensitivity curves."""
    from simlib import io, plotting

    df = pd.read_csv(csv_path)

    if "distortion_strength" in df.columns and "median_abs_delta" in df.columns:
        # Noise robustness
        for dist_kind in df["distortion_kind"].unique():
            subset = df[df["distortion_kind"] == dist_kind]
            fig = plotting.plot_robustness_delta(
                subset,
                x_var="distortion_strength",
                y_var="median_abs_delta",
                title=f"Robustness: {dist_kind}",
            )
            io.save_figure(fig, Path(outdir), f"robustness_{dist_kind}.png")

    elif "param" in df.columns and "value" in df.columns and "similarity" in df.columns:
        # Parameter sensitivity
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Grid Size B
        grid_data = df[df["param"] == "theta_grid_size"]
        if not grid_data.empty:
            sns.lineplot(data=grid_data, x="value", y="similarity", hue="type", ax=axes[0])
            axes[0].set_title("Sensitivity to Grid Size (B)")
            _set_common_axes(axes[0], "Value", "Similarity")

        # Sector Width delta
        width_data = df[df["param"] == "sector_width"]
        if not width_data.empty:
            sns.lineplot(data=width_data, x="value", y="similarity", hue="type", ax=axes[1])
            axes[1].set_title("Sensitivity to Sector Width (delta)")
            _set_common_axes(axes[1], "Value", "Similarity")

        _save_fig(fig, outdir / "robustness_sensitivity")
    else:
        # Fallback: boxplot of similarity by type if it exists
        if "type" in df.columns and "similarity" in df.columns:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.boxplot(data=df, x="type", y="similarity", ax=ax)
            _set_common_axes(ax, "Type", "Similarity")
            ax.set_title("Robustness: Similarity by Type")
            _save_fig(fig, outdir / "robustness_summary")
        else:
            logger.warning(
                f"CSV {csv_path} columns not recognized for robustness plotting: {df.columns.tolist()}"
            )


def plot_baselines(csv_path: Path, outdir: Path):
    """Plot comparison against baseline methods."""
    df = pd.read_csv(csv_path)
    metrics_list = [c for c in df.columns if c not in ["type", "gene", "replicate", "seed"]]
    # Limit to top 3 metrics for visibility
    plot_metrics = metrics_list[:3]

    fig, axes = plt.subplots(1, max(1, len(plot_metrics)), figsize=(5 * len(plot_metrics), 4))
    if len(plot_metrics) == 1:
        axes = [axes]

    for i, metric in enumerate(plot_metrics):
        sns.boxplot(data=df, x="type", y=metric, ax=axes[i])
        axes[i].set_title(metric.replace("_", " ").title())
        _set_common_axes(axes[i], "Type", metric.title())
        axes[i].tick_params(axis="x", rotation=45)

    _save_fig(fig, outdir / "baselines")


def plot_failure_modes(csv_path: Path, outdir: Path):
    """Plot failure mode analysis/abstention rates."""
    df = pd.read_csv(csv_path)
    if "abstain_flag" in df.columns and "case" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        stats = df.groupby("case")["abstain_flag"].mean().reset_index()
        sns.barplot(data=stats, x="case", y="abstain_flag", ax=ax)
        ax.set_ylabel("Abstention Rate")
        ax.set_title("Abstention Rates by Stress Condition")
        _set_common_axes(ax, "Case", "Abstention Rate")
        _save_fig(fig, outdir / "failure_modes")
    else:
        # Fallback: simple histogram of first numeric column
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if not numeric_cols.empty:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(df[numeric_cols[0]].dropna(), bins=20)
            _set_common_axes(ax, numeric_cols[0], "Frequency")
            ax.set_title(f"Distribution: {numeric_cols[0]}")
            _save_fig(fig, outdir / f"failure_mode_{numeric_cols[0]}")


def plot_separability(csv_path: Path, outdir: Path):
    """Plot ROC curves for type separability."""
    df = pd.read_csv(csv_path)
    if set(["fpr", "tpr"]).issubset(df.columns):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(df["fpr"], df["tpr"], lw=2)
        ax.plot([0, 1], [0, 1], "k--")
        _set_common_axes(ax, "False Positive Rate", "True Positive Rate")
        ax.set_title("ROC Curve")
        _save_fig(fig, outdir / "separability_roc")
    elif set(["y_true", "y_score"]).issubset(df.columns):
        from sklearn.metrics import roc_curve

        fpr, tpr, _ = roc_curve(df["y_true"], df["y_score"])
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(fpr, tpr, lw=2)
        ax.plot([0, 1], [0, 1], "k--")
        _set_common_axes(ax, "False Positive Rate", "True Positive Rate")
        ax.set_title("ROC Curve")
        _save_fig(fig, outdir / "separability_roc")
    else:
        logger.warning(f"CSV {csv_path} does not match expected separability structure.")


def main():
    parser = argparse.ArgumentParser(description="Plot simulation results from CSV files")
    # Support for legacy positional args if needed, or structured dirs
    parser.add_argument(
        "--input-dir",
        type=str,
        default="results/simulations_phase3",
        help="Base directory with tables/*.csv",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="results/simulations_phase3/figures_from_csv",
        help="Output figures directory",
    )
    parser.add_argument(
        "--which",
        type=str,
        default="all",
        help="Which plots: all|calibration|power|robustness|baselines|separability|failure",
    )

    # Also support positional for backward compatibility if 2 or 3 args provided
    if len(sys.argv) >= 3 and not sys.argv[1].startswith("-"):
        parser = argparse.ArgumentParser()
        parser.add_argument("csv_path", type=Path)
        parser.add_argument("outdir", type=Path)
        parser.add_argument("--plot_type", type=str, required=True)
        args = parser.parse_args()

        args.outdir.mkdir(parents=True, exist_ok=True)
        if args.plot_type == "calibration":
            plot_calibration(args.csv_path, args.outdir)
        elif args.plot_type == "power":
            plot_power(args.csv_path, args.outdir)
        elif args.plot_type == "robustness":
            plot_robustness(args.csv_path, args.outdir)
        return

    args = parser.parse_args()
    base = Path(args.input_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    plots = (
        args.which.split(",")
        if args.which != "all"
        else ["calibration", "power", "robustness", "baselines", "separability", "failure"]
    )

    mapping = {
        "calibration": ("calibration_summary.csv", plot_calibration),
        "power": ("power_vs_N.csv", plot_power_vs_N),
        "robustness": ("param_sweep_runs.csv", plot_robustness),
        "baselines": ("baseline_comparison.csv", plot_baselines),
        "separability": ("type_separability.csv", plot_separability),
        "failure": ("failure_modes_runs.csv", plot_failure_modes),
    }

    for key, (filename, func) in mapping.items():
        if key in plots:
            csv_path = base / "tables" / filename
            if csv_path.exists():
                logger.info(f"Plotting {key} from {csv_path}")
                try:
                    func(csv_path, outdir)
                except Exception as e:
                    logger.error(f"Error plotting {key}: {e}")
            else:
                logger.debug(f"Skipping {key}: {csv_path} not found")

    print(f"✅ Plotting complete. Figures in: {outdir}")


if __name__ == "__main__":
    main()
