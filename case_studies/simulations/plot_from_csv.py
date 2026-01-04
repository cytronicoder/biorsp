"""
Read simulation CSV outputs (tables/*.csv) and generate standardized figures with
consistent axis labeling, tick formatting, limits, and file outputs.

Usage: import and call the specific plotting functions, or use the CLI script
`scripts/plot_simulation_csv.py`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set(style="whitegrid")


def _save_fig(fig: plt.Figure, outpath: Path, dpi: int = 200):
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath.with_suffix(".png"), bbox_inches="tight", dpi=dpi)
    fig.savefig(outpath.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _set_common_axes(ax: plt.Axes, xlabel: Optional[str] = None, ylabel: Optional[str] = None):
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10)
    ax.tick_params(axis="both", which="major", labelsize=9)
    ax.grid(True, which="both", ls="-", alpha=0.2)


def plot_calibration(csv_path: Path, outdir: Path):
    df = pd.read_csv(csv_path)
    # Expect columns e.g., ['shape','p_val','expected'] or calibration summary files
    fig, ax = plt.subplots(figsize=(6, 6))

    if set(["expected", "observed"]).issubset(df.columns):
        ax.scatter(df["expected"], df["observed"], s=15)
        ax.plot([0, 1], [0, 1], "r--")
        _set_common_axes(ax, "Expected P-value", "Observed P-value")
        ax.set_title("P-value Calibration (Null)")
    else:
        # fallback: histogram of anisotropy / p-values
        if "p_val" in df.columns:
            ax.hist(df["p_val"].dropna(), bins=20, alpha=0.8)
            _set_common_axes(ax, "P-value", "Frequency")
            ax.set_title("P-value Distribution (Null)")
        elif "anisotropy" in df.columns:
            ax.hist(df["anisotropy"].dropna(), bins=20, alpha=0.8)
            _set_common_axes(ax, "RMS Anisotropy", "Frequency")
            ax.set_title("Null Anisotropy Distribution")
        else:
            raise ValueError("Unrecognized calibration CSV structure")

    _save_fig(fig, outdir / "calibration")


def plot_power_vs_N(csv_path: Path, outdir: Path):
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(8, 5))

    if "N" in df.columns and "detected" in df.columns:
        sns.lineplot(
            data=df,
            x="N",
            y="detected",
            hue=(
                df.columns[df.columns.str.lower().str.contains("q")].tolist()[0]
                if any(df.columns.str.lower().str.contains("q"))
                else None
            ),
            marker="o",
            ax=ax,
        )
        ax.set_xscale("log")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title("Detection Power vs N")
        _set_common_axes(ax, "N (log scale)", "Power")
        # add horizontal baseline at alpha if present
        if "alpha" in df.columns:
            alpha = df["alpha"].iloc[0]
            ax.axhline(alpha, ls="--", color="gray", alpha=0.5, label=f"alpha={alpha}")
            ax.legend(title="q")
    else:
        raise ValueError("Unrecognized power CSV structure")

    _save_fig(fig, outdir / "power_vs_N")


def plot_robustness(csv_path: Path, outdir: Path):
    df = pd.read_csv(csv_path)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    if "similarity" in df.columns and "param" in df.columns:
        sns.lineplot(
            data=df[df["param"] == "theta_grid_size"],
            x="value",
            y="similarity",
            hue="type",
            ax=axes[0],
        )
        axes[0].set_title("Sensitivity to Grid Size (B)")
        _set_common_axes(axes[0], "Value", "Similarity")

        sns.lineplot(
            data=df[df["param"] == "sector_width"],
            x="value",
            y="similarity",
            hue="type",
            ax=axes[1],
        )
        axes[1].set_title("Sensitivity to Sector Width (delta)")
        _set_common_axes(axes[1], "Value", "Similarity")
    else:
        # fallback: boxplot of similarity by type
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.boxplot(data=df, x="type", y="similarity", ax=ax)
        _set_common_axes(ax, "Type", "Similarity")
        ax.set_title("Robustness: Similarity by Type")

    _save_fig(fig, outdir / "robustness")


def plot_baselines(csv_path: Path, outdir: Path):
    df = pd.read_csv(csv_path)
    metrics = [c for c in df.columns if c not in ["type"]][:3]
    fig, axes = plt.subplots(1, max(1, len(metrics)), figsize=(5 * len(metrics), 4))
    if len(metrics) == 1:
        axes = [axes]
    for i, metric in enumerate(metrics):
        sns.boxplot(data=df, x="type", y=metric, ax=axes[i])
        axes[i].set_title(metric.replace("_", " ").title())
        _set_common_axes(axes[i], "Type", metric.title())
        axes[i].tick_params(axis="x", rotation=45)
    _save_fig(fig, outdir / "baselines")


def plot_failure_modes(csv_path: Path, outdir: Path):
    df = pd.read_csv(csv_path)
    # if abstain_flag column exists
    if "abstain_flag" in df.columns and "case" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(
            data=df.groupby("case")["abstain_flag"].mean().reset_index(),
            x="case",
            y="abstain_flag",
            ax=ax,
        )
        ax.set_ylabel("Abstention Rate")
        ax.set_title("Abstention Rates by Stress Condition")
        _set_common_axes(ax, "Case", "Abstention Rate")
        _save_fig(fig, outdir / "failure_modes")
    else:
        # fallback: simple histogram of some metric
        fig, ax = plt.subplots(figsize=(6, 4))
        cols = [c for c in df.columns if df[c].dtype in ["int64", "float64"]]
        if not cols:
            raise ValueError("No numeric columns found for failure modes plotting")
        ax.hist(df[cols[0]].dropna(), bins=20)
        _set_common_axes(ax, cols[0], "Frequency")
        ax.set_title("Failure Modes Metric Distribution")
        _save_fig(fig, outdir / "failure_modes_metric")


def plot_separability(csv_path: Path, outdir: Path):
    df = pd.read_csv(csv_path)
    # Expect 'fpr','tpr' for ROC or 'y_true','y_score' for AUC
    if set(["fpr", "tpr"]).issubset(df.columns):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(df["fpr"], df["tpr"], lw=2)
        ax.plot([0, 1], [0, 1], "k--")
        _set_common_axes(ax, "False Positive Rate", "True Positive Rate")
        ax.set_title("ROC")
        _save_fig(fig, outdir / "separability_roc")
    elif set(["y_true", "y_score"]).issubset(df.columns):
        # compute ROC points
        from sklearn.metrics import roc_curve

        fpr, tpr, _ = roc_curve(df["y_true"], df["y_score"])
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(fpr, tpr, lw=2)
        ax.plot([0, 1], [0, 1], "k--")
        _set_common_axes(ax, "False Positive Rate", "True Positive Rate")
        ax.set_title("ROC")
        _save_fig(fig, outdir / "separability_roc")
    else:
        raise ValueError("Unrecognized separability CSV structure")


def main():
    parser = argparse.ArgumentParser(
        description="Plot simulation results from CSV files with revised axes"
    )
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
        help="Which plots to create: all|calibration|power|robustness|baselines|separability|failure",
    )

    args = parser.parse_args()
    base = Path(args.input_dir)
    outdir = Path(args.outdir)
    plots = (
        args.which.split(",")
        if args.which != "all"
        else ["calibration", "power", "robustness", "baselines", "separability", "failure"]
    )

    if "calibration" in plots:
        cal_csv = base / "tables" / "calibration_summary.csv"
        if cal_csv.exists():
            plot_calibration(cal_csv, outdir)
        else:
            print(f"Calibration CSV not found at {cal_csv}")

    if "power" in plots:
        power_csv = base / "tables" / "power_vs_N.csv"
        if power_csv.exists():
            plot_power_vs_N(power_csv, outdir)
        else:
            print(f"Power CSV not found at {power_csv}")

    if "robustness" in plots:
        robustness_csv = base / "tables" / "param_sweep_runs.csv"
        if robustness_csv.exists():
            plot_robustness(robustness_csv, outdir)
        else:
            print(f"Robustness CSV not found at {robustness_csv}")

    if "baselines" in plots:
        baselines_csv = base / "tables" / "baseline_comparison.csv"
        if baselines_csv.exists():
            plot_baselines(baselines_csv, outdir)
        else:
            print(f"Baselines CSV not found at {baselines_csv}")

    if "separability" in plots:
        separability_csv = base / "tables" / "type_separability.csv"
        if separability_csv.exists():
            plot_separability(separability_csv, outdir)
        else:
            print(f"Separability CSV not found at {separability_csv}")

    if "failure" in plots:
        failure_csv = base / "tables" / "failure_modes_runs.csv"
        if failure_csv.exists():
            plot_failure_modes(failure_csv, outdir)
        else:
            print(f"Failure modes CSV not found at {failure_csv}")


if __name__ == "__main__":
    main()
