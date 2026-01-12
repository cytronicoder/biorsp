"""Plot simulation results from CSV files with robust schema validation.

This module provides plotting functions for various simulation result types.
Each function validates the CSV schema before plotting and provides actionable
error messages if columns are missing or data is empty after filtering.

Usage:
    python plot_benchmarks.py --input-dir <results_dir> --outdir <output_dir> --which all

Schema Validation:
    - Validates required columns before plotting
    - Provides actionable error messages listing missing columns
    - Shows dataframe shape and sample data on validation failures
    - Prevents silent empty plot generation
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Set

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def validate_columns(
    df: pd.DataFrame, required_cols: Set[str], csv_path: Path, plot_type: str
) -> None:
    """Validate that dataframe contains required columns.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to validate
    required_cols : Set[str]
        Set of required column names
    csv_path : Path
        Path to CSV file (for error messages)
    plot_type : str
        Type of plot being generated (for error messages)

    Raises
    ------
    ValueError
        If required columns are missing
    """
    missing = required_cols - set(df.columns)
    if missing:
        error_msg = (
            f"\n{'=' * 60}\n"
            f"SCHEMA VALIDATION ERROR: {plot_type}\n"
            f"{'=' * 60}\n"
            f"CSV: {csv_path}\n\n"
            f"Missing columns: {sorted(missing)}\n"
            f"Found columns: {sorted(df.columns.tolist())}\n\n"
            f"Dataframe shape: {df.shape}\n"
            f"First few rows:\n{df.head(3)}\n"
            f"{'=' * 60}\n"
        )
        raise ValueError(error_msg)


def check_empty_after_filter(
    df: pd.DataFrame, original_shape: tuple, filter_desc: str, csv_path: Path, plot_type: str
) -> None:
    """Check if dataframe is empty after filtering and raise informative error.

    Parameters
    ----------
    df : pd.DataFrame
        Filtered dataframe
    original_shape : tuple
        Shape of dataframe before filtering
    filter_desc : str
        Description of filter applied
    csv_path : Path
        Path to CSV file
    plot_type : str
        Type of plot being generated

    Raises
    ------
    ValueError
        If dataframe is empty after filtering
    """
    if df.empty:
        error_msg = (
            f"\n{'=' * 60}\n"
            f"EMPTY DATA ERROR: {plot_type}\n"
            f"{'=' * 60}\n"
            f"CSV: {csv_path}\n\n"
            f"Filter: {filter_desc}\n"
            f"Original shape: {original_shape}\n"
            f"After filter: {df.shape}\n\n"
            f"This would produce an empty plot. Check:\n"
            f"1. Are the filter column values correct?\n"
            f"2. Does the data contain the expected categories?\n"
            f"3. Are column names matching expected format?\n"
            f"{'=' * 60}\n"
        )
        raise ValueError(error_msg)


def _save_fig(fig, path):
    """Save figure to path as PNG and PDF, then close."""
    path = Path(path)
    fig.savefig(f"{path}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{path}.pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {path}.png and {path}.pdf")


def _set_common_axes(ax, xlabel, ylabel):
    """Set common axis properties."""
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.3)


def plot_calibration(csv_path: Path, outdir: Path):
    """Plot calibration results (QQ plot) with schema validation."""
    df = pd.read_csv(csv_path)

    validate_columns(df, {"p_value"}, csv_path, "Calibration QQ Plot")

    p_values = df["p_value"].dropna()
    if len(p_values) == 0:
        raise ValueError(f"No valid p-values in {csv_path}. All {len(df)} rows have NaN p_value.")

    try:
        from biorsp.simulations import io, metrics, plotting

        p_values_arr = p_values.values
        expected, observed = metrics.qq_quantiles(p_values_arr)
        fig = plotting.plot_qq(expected, observed, title="Calibration: QQ Plot")
        io.save_figure(fig, Path(outdir), "calibration_qq.png")
    except ImportError:
        fig, ax = plt.subplots(figsize=(6, 6))
        sorted_p = np.sort(p_values.values)
        expected_quantiles = np.linspace(0, 1, len(sorted_p))
        ax.plot([0, 1], [0, 1], "k--", label="Uniform")
        ax.plot(expected_quantiles, sorted_p, "o", alpha=0.5, label="Observed")
        ax.set_xlabel("Expected Quantile")
        ax.set_ylabel("Observed P-value")
        ax.set_title("Calibration QQ Plot")
        ax.legend()
        _save_fig(fig, outdir / "calibration_qq")


def plot_power(csv_path: Path, outdir: Path):
    """Plot power curves with schema validation."""
    df = pd.read_csv(csv_path)

    if "N" not in df.columns:
        validate_columns(df, {"N"}, csv_path, "Power Curve")

    if "power_mean" in df.columns:
        df = df.rename(columns={"power_mean": "power"})

    validate_columns(df, {"N", "power"}, csv_path, "Power Curve")

    df_clean = df[["N", "power"]].dropna()
    if df_clean.empty:
        raise ValueError(
            f"No valid (N, power) pairs in {csv_path}. "
            f"Original shape: {df.shape}, after dropna: {df_clean.shape}"
        )

    try:
        from biorsp.simulations import io, plotting

        fig = plotting.plot_power_curve(df, x_var="N", title="Power vs Sample Size")
        io.save_figure(fig, Path(outdir), "power_vs_N.png")
    except ImportError:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(df_clean["N"], df_clean["power"], "o-", lw=2)
        ax.set_xlabel("Sample Size (N)")
        ax.set_ylabel("Power")
        ax.set_title("Power vs Sample Size")
        ax.grid(alpha=0.3)
        _save_fig(fig, outdir / "power_vs_N")


plot_power_vs_N = plot_power


def plot_robustness(csv_path: Path, outdir: Path):
    """Plot robustness or parameter sensitivity curves with schema validation."""
    df = pd.read_csv(csv_path)
    original_shape = df.shape

    if "distortion_strength" in df.columns and "median_abs_delta" in df.columns:
        validate_columns(
            df,
            {"distortion_strength", "median_abs_delta", "distortion_kind"},
            csv_path,
            "Robustness (Distortion)",
        )

        try:
            from biorsp.simulations import io, plotting

            for dist_kind in df["distortion_kind"].unique():
                subset = df[df["distortion_kind"] == dist_kind]
                check_empty_after_filter(
                    subset,
                    original_shape,
                    f"distortion_kind == '{dist_kind}'",
                    csv_path,
                    "Robustness",
                )
                fig = plotting.plot_robustness_delta(
                    subset,
                    x_var="distortion_strength",
                    y_var="median_abs_delta",
                    title=f"Robustness: {dist_kind}",
                )
                io.save_figure(fig, Path(outdir), f"robustness_{dist_kind}.png")
        except ImportError:
            logger.warning("simlib not available, skipping robustness distortion plots")

    elif "param" in df.columns and "value" in df.columns and "similarity" in df.columns:
        validate_columns(
            df, {"param", "value", "similarity", "type"}, csv_path, "Robustness (Param Sweep)"
        )

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        grid_data = df[df["param"] == "theta_grid_size"]
        if not grid_data.empty:
            sns.lineplot(data=grid_data, x="value", y="similarity", hue="type", ax=axes[0])
            axes[0].set_title("Sensitivity to Grid Size (B)")
            _set_common_axes(axes[0], "Value", "Similarity")
        else:
            axes[0].text(
                0.5, 0.5, "No theta_grid_size data", ha="center", transform=axes[0].transAxes
            )

        width_data = df[df["param"] == "sector_width"]
        if not width_data.empty:
            sns.lineplot(data=width_data, x="value", y="similarity", hue="type", ax=axes[1])
            axes[1].set_title("Sensitivity to Sector Width (delta)")
            _set_common_axes(axes[1], "Value", "Similarity")
        else:
            axes[1].text(0.5, 0.5, "No sector_width data", ha="center", transform=axes[1].transAxes)

        _save_fig(fig, outdir / "robustness_sensitivity")

    elif "type" in df.columns and "similarity" in df.columns:
        validate_columns(df, {"type", "similarity"}, csv_path, "Robustness (Summary)")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.boxplot(data=df, x="type", y="similarity", ax=ax)
        _set_common_axes(ax, "Type", "Similarity")
        ax.set_title("Robustness: Similarity by Type")
        _save_fig(fig, outdir / "robustness_summary")
    else:
        raise ValueError(
            f"Unrecognized robustness schema in {csv_path}. \n"
            f"Found columns: {df.columns.tolist()}\n"
            f"Expected one of:\n"
            f"  - (distortion_strength, median_abs_delta, distortion_kind)\n"
            f"  - (param, value, similarity, type)\n"
            f"  - (type, similarity)"
        )


def plot_baselines(csv_path: Path, outdir: Path):
    """Plot comparison against baseline methods."""
    df = pd.read_csv(csv_path)
    metrics_list = [c for c in df.columns if c not in ["type", "gene", "replicate", "seed"]]

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
