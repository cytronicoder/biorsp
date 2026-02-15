"""Classification figure factories for gene-level score outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from biorsp.plotting.styles import DEFAULT_PLOT_STYLE, PlotStyle
from biorsp.plotting.utils import sanitize_feature_label, save_figure

CLASS_ORDER = [
    "Underpowered",
    "Ubiquitous (non-localized)",
    "Localized–unimodal",
    "Localized–multimodal",
    "QC-driven",
    "Uncertain",
]

CLASS_COLORS = {
    "Underpowered": "#a6a6a6",
    "Ubiquitous (non-localized)": "#2ca02c",
    "Localized–unimodal": "#1f77b4",
    "Localized–multimodal": "#ff7f0e",
    "QC-driven": "#d62728",
    "Uncertain": "#9467bd",
}


def _safe_series(df: pd.DataFrame, key: str) -> np.ndarray:
    return pd.to_numeric(df[key], errors="coerce").fillna(0.0).to_numpy(dtype=float)


def plot_classification_suite(
    metrics_df: pd.DataFrame,
    out_dir: Path,
    *,
    z_strong_threshold: float,
    coverage_strong_threshold: float,
    style: PlotStyle = DEFAULT_PLOT_STYLE,
) -> dict[str, str | None]:
    """Render score/classification panels from a scored metrics table."""
    out_dir.mkdir(parents=True, exist_ok=True)
    score_scatter_path = out_dir / "score1_score2_scatter.png"
    class_counts_path = out_dir / "class_counts.png"
    class_scatter_path = out_dir / "classification_scatter.png"

    artifacts: dict[str, str | None] = {
        "score1_score2_scatter": None,
        "class_counts": None,
        "classification_scatter": None,
        "class_distribution_plots": None,
    }

    if metrics_df.empty:
        return artifacts

    # score scatter
    fig_sc, ax_sc = plt.subplots(figsize=style.figsize_classification)
    for cls in CLASS_ORDER:
        sub = metrics_df.loc[metrics_df["class_label"] == cls]
        if sub.shape[0] == 0:
            continue
        ax_sc.scatter(
            _safe_series(sub, "score_1"),
            _safe_series(sub, "score_2"),
            s=28,
            alpha=0.85,
            color=CLASS_COLORS.get(cls, "#333333"),
            label=f"{cls} (n={sub.shape[0]})",
        )

    ann_df = (
        metrics_df.sort_values(
            by=["q_T", "score_1", "gene"],
            ascending=[True, False, True],
            kind="mergesort",
        )
        .head(10)
        .reset_index(drop=True)
    )
    ann_offsets = [(0.10, 0.012), (-0.10, 0.012), (0.10, -0.012), (-0.10, -0.012), (0.00, 0.018)]
    for i, row in ann_df.iterrows():
        dx, dy = ann_offsets[i % len(ann_offsets)]
        ax_sc.text(
            float(pd.to_numeric(row.get("score_1"), errors="coerce")) + dx,
            float(pd.to_numeric(row.get("score_2"), errors="coerce")) + dy,
            str(row.get("gene", "")),
            fontsize=7,
            color="black",
        )

    ax_sc.set_xlabel("score_1 (Z_T)")
    ax_sc.set_ylabel("score_2 (coverage_C)")
    ax_sc.set_title("Gene score_1 vs score_2")
    ax_sc.legend(loc="best", fontsize=style.legend_fontsize, frameon=True)
    ax_sc.grid(alpha=0.25, linewidth=0.6)
    fig_sc.tight_layout()
    save_figure(fig_sc, score_scatter_path, style=style)
    artifacts["score1_score2_scatter"] = score_scatter_path.as_posix()

    # class counts
    class_counts = metrics_df["class_label"].value_counts().reindex(CLASS_ORDER, fill_value=0).astype(int)
    fig_cc, ax_cc = plt.subplots(figsize=style.figsize_class_counts)
    x = np.arange(len(class_counts), dtype=float)
    ax_cc.bar(
        x,
        class_counts.to_numpy(dtype=float),
        color=[CLASS_COLORS.get(c, "#333333") for c in class_counts.index],
        edgecolor="black",
        linewidth=0.6,
    )
    ax_cc.set_xticks(x)
    ax_cc.set_xticklabels(class_counts.index, rotation=25, ha="right", fontsize=8)
    ax_cc.set_ylabel("Gene count")
    ax_cc.set_title("Classification counts")
    ax_cc.grid(axis="y", alpha=0.25, linewidth=0.6)
    fig_cc.tight_layout()
    save_figure(fig_cc, class_counts_path, style=style)
    artifacts["class_counts"] = class_counts_path.as_posix()

    # classification map
    fig_cls, ax_cls = plt.subplots(figsize=style.figsize_classification)
    for cls in CLASS_ORDER:
        sub = metrics_df.loc[metrics_df["class_label"] == cls]
        if sub.shape[0] == 0:
            continue
        ax_cls.scatter(
            _safe_series(sub, "score_1"),
            _safe_series(sub, "score_2"),
            s=28,
            alpha=0.85,
            color=CLASS_COLORS.get(cls, "#333333"),
            label=f"{cls} (n={sub.shape[0]})",
        )
    ax_cls.axvline(z_strong_threshold, color="black", linestyle="--", linewidth=1.0)
    ax_cls.axhline(coverage_strong_threshold, color="black", linestyle="--", linewidth=1.0)
    ax_cls.set_xlabel("score_1 (Z_T)")
    ax_cls.set_ylabel("score_2 (coverage_C)")
    ax_cls.set_title("Gene classification map")
    ax_cls.legend(loc="best", fontsize=style.legend_fontsize, frameon=True)
    ax_cls.grid(alpha=0.25, linewidth=0.6)
    fig_cls.tight_layout()
    save_figure(fig_cls, class_scatter_path, style=style)
    artifacts["classification_scatter"] = class_scatter_path.as_posix()

    # per-class distributions
    class_hist_paths: list[str] = []
    for cls in CLASS_ORDER:
        sub = metrics_df.loc[metrics_df["class_label"] == cls]
        if sub.shape[0] < 2:
            continue
        n_bins = int(min(24, max(6, np.ceil(np.sqrt(sub.shape[0])))))
        fig_h, axes_h = plt.subplots(1, 2, figsize=(8.8, 3.6))
        axes_h[0].hist(
            _safe_series(sub, "score_1"),
            bins=n_bins,
            color=CLASS_COLORS.get(cls, "#333333"),
            alpha=0.85,
            edgecolor="black",
            linewidth=0.6,
        )
        axes_h[0].set_title(f"{cls}: score_1")
        axes_h[0].set_xlabel("score_1 (Z_T)")
        axes_h[0].set_ylabel("Count")
        axes_h[1].hist(
            _safe_series(sub, "score_2"),
            bins=n_bins,
            color=CLASS_COLORS.get(cls, "#333333"),
            alpha=0.85,
            edgecolor="black",
            linewidth=0.6,
        )
        axes_h[1].set_title(f"{cls}: score_2")
        axes_h[1].set_xlabel("score_2 (coverage_C)")
        axes_h[1].set_ylabel("Count")
        fig_h.tight_layout()
        hist_path = out_dir / f"class_{sanitize_feature_label(cls)}_score_distributions.png"
        save_figure(fig_h, hist_path, style=style)
        class_hist_paths.append(hist_path.as_posix())

    artifacts["class_distribution_plots"] = ";".join(class_hist_paths) if class_hist_paths else None
    return artifacts


def class_counts_dict(metrics_df: pd.DataFrame) -> dict[str, int]:
    """Get deterministic class counts dictionary for metadata."""
    if metrics_df.empty or "class_label" not in metrics_df.columns:
        return {cls: 0 for cls in CLASS_ORDER}
    return (
        metrics_df["class_label"]
        .value_counts()
        .reindex(CLASS_ORDER, fill_value=0)
        .astype(int)
        .to_dict()
    )


def plot_score_qc_suite(
    metrics_df: pd.DataFrame,
    out_dir: Path,
    *,
    primary_score_col: str = "score_1",
    style: PlotStyle = DEFAULT_PLOT_STYLE,
) -> dict[str, str | None]:
    """Render score/QC summary plots used by the two-stage case-study workflow."""
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, str | None] = {
        "score_distribution": None,
        "score_vs_mean_expr": None,
        "score_vs_foreground_frac": None,
        "score_component_corr": None,
    }
    if metrics_df.empty or primary_score_col not in metrics_df.columns:
        return artifacts

    score = pd.to_numeric(metrics_df[primary_score_col], errors="coerce").to_numpy(dtype=float)
    finite_score = score[np.isfinite(score)]
    if finite_score.size > 0:
        fig_hist, ax_hist = plt.subplots(figsize=(6.8, 4.2))
        bins = int(min(60, max(12, np.ceil(np.sqrt(finite_score.size)))))
        ax_hist.hist(finite_score, bins=bins, color="#2f6fb0", alpha=0.85, edgecolor="black", linewidth=0.5)
        ax_hist.set_xlabel(primary_score_col)
        ax_hist.set_ylabel("Gene count")
        ax_hist.set_title(f"{primary_score_col} distribution")
        ax_hist.grid(alpha=0.20, linewidth=0.6)
        fig_hist.tight_layout()
        p_hist = out_dir / "score_distribution_hist.png"
        save_figure(fig_hist, p_hist, style=style)
        artifacts["score_distribution"] = p_hist.as_posix()

    if "mean_expr" in metrics_df.columns:
        mean_expr = pd.to_numeric(metrics_df["mean_expr"], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(score) & np.isfinite(mean_expr)
        if int(mask.sum()) > 1:
            fig_me, ax_me = plt.subplots(figsize=(6.8, 4.2))
            ax_me.scatter(mean_expr[mask], score[mask], s=20, alpha=0.65, color="#c4503f", linewidths=0)
            ax_me.set_xlabel("mean_expr")
            ax_me.set_ylabel(primary_score_col)
            ax_me.set_title(f"{primary_score_col} vs mean_expr")
            ax_me.grid(alpha=0.20, linewidth=0.6)
            fig_me.tight_layout()
            p_me = out_dir / "score_vs_mean_expr.png"
            save_figure(fig_me, p_me, style=style)
            artifacts["score_vs_mean_expr"] = p_me.as_posix()

    fg_col = "foreground_frac" if "foreground_frac" in metrics_df.columns else "prevalence"
    if fg_col in metrics_df.columns:
        fg = pd.to_numeric(metrics_df[fg_col], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(score) & np.isfinite(fg)
        if int(mask.sum()) > 1:
            fig_fg, ax_fg = plt.subplots(figsize=(6.8, 4.2))
            ax_fg.scatter(fg[mask], score[mask], s=20, alpha=0.65, color="#2b8f7b", linewidths=0)
            ax_fg.set_xlabel(fg_col)
            ax_fg.set_ylabel(primary_score_col)
            ax_fg.set_title(f"{primary_score_col} vs {fg_col}")
            ax_fg.grid(alpha=0.20, linewidth=0.6)
            fig_fg.tight_layout()
            p_fg = out_dir / "score_vs_foreground_frac.png"
            save_figure(fig_fg, p_fg, style=style)
            artifacts["score_vs_foreground_frac"] = p_fg.as_posix()

    corr_cols = [
        c
        for c in [
            "score_1",
            "score_2",
            "T_obs",
            "Z_T",
            "coverage_C",
            "strength_proxy",
            "specificity_proxy",
            "E_max",
            "entropy",
            "mean_expr",
            "foreground_frac",
        ]
        if c in metrics_df.columns
    ]
    if len(corr_cols) >= 2:
        corr_df = metrics_df[corr_cols].apply(pd.to_numeric, errors="coerce")
        corr = corr_df.corr(method="spearman", min_periods=3).to_numpy(dtype=float)
        fig_corr, ax_corr = plt.subplots(figsize=(8.0, 6.4))
        im = ax_corr.imshow(corr, vmin=-1.0, vmax=1.0, cmap="coolwarm")
        ax_corr.set_xticks(np.arange(len(corr_cols)))
        ax_corr.set_xticklabels(corr_cols, rotation=40, ha="right", fontsize=8)
        ax_corr.set_yticks(np.arange(len(corr_cols)))
        ax_corr.set_yticklabels(corr_cols, fontsize=8)
        ax_corr.set_title("Score component correlation (Spearman)")
        cbar = fig_corr.colorbar(im, ax=ax_corr, shrink=0.8)
        cbar.set_label("rho", fontsize=9)
        fig_corr.tight_layout()
        p_corr = out_dir / "score_component_correlation_heatmap.png"
        save_figure(fig_corr, p_corr, style=style)
        artifacts["score_component_corr"] = p_corr.as_posix()

    return artifacts
