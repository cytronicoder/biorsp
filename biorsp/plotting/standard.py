"""Standard plotting API shared by simulation benchmarks and kidney case studies."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from biorsp.utils.labels import (
    ABSTAIN_LABEL,
    CANONICAL_ARCHETYPES,
    assert_archetype_labels,
    classify_from_thresholds,
    label_order,
    label_palette,
    normalize_archetype_series,
)

DEFAULT_THRESHOLDS = {"C_cut": 0.3, "S_cut": 0.15}


def get_archetype_palette() -> dict[str, str]:
    """Stable color mapping for canonical archetypes (no Abstain)."""

    return label_palette(include_abstain=False)


def _require_columns(df: pd.DataFrame, cols: Iterable[str], context: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for {context}: {missing}")


def plot_cs_scatter(
    df: pd.DataFrame,
    C_col: str = "Coverage",
    S_col: str = "Spatial_Score",
    true_col: str | None = None,
    pred_col: str = "Archetype_pred",
    C_cut: float = 0.3,
    S_cut: float = 0.15,
    palette: Mapping[str, str] | None = None,
    annotate_quadrants: bool = True,
):
    """Coverage vs spatial score scatter with archetype coloring.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing scores and optional truth/prediction columns.
    C_col : str, optional
        Name of the coverage column, by default 'Coverage'.
    S_col : str, optional
        Name of the spatial score column, by default 'Spatial_Score'.
    true_col : str | None, optional
        Column containing ground-truth archetype labels, by default None.
    pred_col : str, optional
        Column used for predicted archetypes, by default 'Archetype_pred'.
    C_cut : float, optional
        Vertical threshold line for coverage, by default 0.3.
    S_cut : float, optional
        Horizontal threshold line for spatial score, by default 0.15.
    palette : Mapping[str, str] | None, optional
        Color mapping for archetypes. If None, default palette is used.
    annotate_quadrants : bool, optional
        Whether to annotate quadrant labels on the plot, by default True.

    Returns
    -------
    matplotlib.figure.Figure
        Created scatter figure.
    """

    _require_columns(df, [C_col, S_col], "Coverage/Spatial scatter")
    palette = dict(get_archetype_palette() if palette is None else palette)

    plot_df = df.copy()

    if true_col and true_col in plot_df.columns:
        plot_df[true_col] = normalize_archetype_series(plot_df[true_col], allow_abstain=True)
        assert_archetype_labels(plot_df, true_col, allow_abstain=True)

    if pred_col not in plot_df.columns:
        plot_df[pred_col] = [
            classify_from_thresholds(c, s, C_cut, S_cut)
            for c, s in zip(plot_df[C_col], plot_df[S_col])
        ]
    else:
        plot_df[pred_col] = normalize_archetype_series(plot_df[pred_col], allow_abstain=True)

    assert_archetype_labels(plot_df, pred_col, allow_abstain=True)

    fig, ax = plt.subplots(figsize=(6, 6))

    display_labels = label_order(include_abstain=ABSTAIN_LABEL in plot_df[pred_col].unique())

    for archetype in display_labels:
        mask = plot_df[pred_col] == archetype
        if not mask.any():
            continue
        color = palette.get(archetype, "#BBBBBB")
        ax.scatter(
            plot_df.loc[mask, C_col],
            plot_df.loc[mask, S_col],
            s=35,
            color=color,
            edgecolors="white",
            linewidths=0.5,
            alpha=0.8,
            label=archetype,
        )

    ax.axvline(C_cut, color="black", linestyle="--", linewidth=1.2)
    ax.axhline(S_cut, color="black", linestyle="--", linewidth=1.2)

    if annotate_quadrants:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        annotations = [
            ((xlim[0] + C_cut) / 2, (ylim[0] + S_cut) / 2, "Basal"),
            ((xlim[1] + C_cut) / 2, (ylim[0] + S_cut) / 2, "Ubiquitous"),
            ((xlim[0] + C_cut) / 2, (ylim[1] + S_cut) / 2, "Patchy"),
            ((xlim[1] + C_cut) / 2, (ylim[1] + S_cut) / 2, "Gradient"),
        ]
        for x, y, label in annotations:
            ax.text(
                x,
                y,
                label,
                color=palette.get(label, "black"),
                fontsize=9,
                ha="center",
                va="center",
                alpha=0.7,
            )

    ax.set_xlabel("Coverage")
    ax.set_ylabel("Spatial Score")
    ax.legend(title="Archetype", frameon=True)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_confusion_matrix(
    cm: np.ndarray, labels: Sequence[str], normalize: str | None = "true", title: str = "Confusion"
):
    """Plot a confusion matrix heatmap.

    Parameters
    ----------
    cm : np.ndarray
        Square confusion matrix counts.
    labels : Sequence[str]
        Label names in the same order as rows/columns of ``cm``.
    normalize : str | None, optional
        Normalization mode: 'true' (row-wise), 'pred' (col-wise), 'all' or None
        for raw counts, by default 'true'.
    title : str, optional
        Plot title, by default 'Confusion'.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the confusion matrix visualization.
    """
    cm = np.asarray(cm, dtype=float)
    fig, ax = plt.subplots(figsize=(6, 5))

    plot_cm = cm.copy()
    if normalize == "true":
        row_sums = plot_cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        plot_cm = plot_cm / row_sums
    elif normalize == "pred":
        col_sums = plot_cm.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1
        plot_cm = plot_cm / col_sums
    elif normalize == "all":
        total = plot_cm.sum()
        if total > 0:
            plot_cm = plot_cm / total

    im = ax.imshow(plot_cm, cmap="Blues", vmin=0, vmax=plot_cm.max() if plot_cm.size else 1)
    for i in range(plot_cm.shape[0]):
        for j in range(plot_cm.shape[1]):
            val = plot_cm[i, j]
            disp = f"{val:.2f}" if normalize else f"{int(cm[i, j])}"
            color = "white" if val > plot_cm.max() * 0.6 else "black"
            ax.text(j, i, disp, ha="center", va="center", color=color, fontsize=9)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def plot_marker_recovery(
    df_ranked: pd.DataFrame,
    k_list: Sequence[int] = (10, 20, 50, 100),
    baseline: float | None = None,
):
    _require_columns(df_ranked, ["label_true"], "marker recovery")
    fig, ax = plt.subplots(figsize=(6, 4))
    n = len(df_ranked)
    for k in k_list:
        k_eff = min(k, n)
        precision = df_ranked.iloc[:k_eff]["label_true"].mean()
        ax.bar(str(k_eff), precision, color="#4CAF50", alpha=0.8)
    if baseline is not None:
        ax.axhline(baseline, color="gray", linestyle="--", label="baseline")
        ax.legend()
    ax.set_ylim(0, 1)
    ax.set_ylabel("Precision")
    ax.set_xlabel("Top-k")
    fig.tight_layout()
    return fig


def plot_genegene_metrics(summary: Mapping[str, float]):
    keys = list(summary.keys())
    values = [summary[k] for k in keys]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(len(keys)), values, color="#2196F3")
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=45, ha="right")
    ax.set_ylabel("Value")
    fig.tight_layout()
    return fig


def plot_calibration_qq(
    pvals: Iterable[float],
    alpha: float = 0.05,
    perm_floor: float | None = None,
    title: str = "QQ plot",
    zoom_alpha: bool = True,
):
    pvals = np.asarray(list(pvals), dtype=float)
    pvals = pvals[~np.isnan(pvals)]
    if pvals.size == 0:
        raise ValueError("No p-values provided for QQ plot")
    expected = np.linspace(0, 1, len(pvals), endpoint=False) + 1 / (len(pvals) + 1)
    observed = np.sort(pvals)

    if zoom_alpha:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        ax_small, ax_full = axes
    else:
        fig, ax_full = plt.subplots(figsize=(6, 5))
        ax_small = None

    def _plot(ax):
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax.scatter(expected, observed, s=12, color="#2196F3", alpha=0.7)
        ax.axvline(alpha, color="red", linestyle=":", linewidth=1)
        ax.axhline(alpha, color="red", linestyle=":", linewidth=1)
        if perm_floor is not None:
            ax.axhline(perm_floor, color="orange", linestyle=":", linewidth=1)
        ax.set_xlabel("Expected")
        ax.set_ylabel("Observed")
        ax.grid(alpha=0.3)

    _plot(ax_full)
    ax_full.set_title(title)
    if zoom_alpha and ax_small is not None:
        _plot(ax_small)
        ax_small.set_xlim(0, alpha * 1.1)
        ax_small.set_ylim(0, alpha * 1.1)
        ax_small.set_title(f"Zoom near alpha={alpha}")

    fig.tight_layout()
    return fig


def plot_fpr_grid(df_summary: pd.DataFrame, alpha: float = 0.05, title: str = "FPR grid"):
    _require_columns(df_summary, ["metric", "mean", "ci_low", "ci_high"], "FPR grid")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.axhline(alpha, color="red", linestyle=":", linewidth=1.2, label=f"alpha={alpha}")
    x = np.arange(len(df_summary))
    ax.errorbar(
        x,
        df_summary["mean"],
        yerr=[
            df_summary["mean"] - df_summary["ci_low"],
            df_summary["ci_high"] - df_summary["mean"],
        ],
        fmt="o",
        color="#2196F3",
        ecolor="#90CAF9",
        capsize=4,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(df_summary["metric"], rotation=45, ha="right")
    ax.set_ylabel("FPR")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_embedding_with_archetypes(
    embedding_xy: np.ndarray,
    labels: Sequence[str],
    palette: Mapping[str, str] | None = None,
    title: str = "Embedding",
):
    if embedding_xy.shape[0] != len(labels):
        raise ValueError("embedding_xy and labels must have the same length")
    palette = dict(get_archetype_palette() if palette is None else palette)
    labels_series = normalize_archetype_series(pd.Series(labels))
    fig, ax = plt.subplots(figsize=(6, 6))
    for archetype in CANONICAL_ARCHETYPES:
        mask = labels_series == archetype
        if not mask.any():
            continue
        coords = embedding_xy[mask.to_numpy()]
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            s=20,
            color=palette.get(archetype, "#BBBBBB"),
            label=archetype,
            alpha=0.8,
        )
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def _ensure_figdir(outdir: Path) -> Path:
    """Ensure output directory exists and return it (no subdirectory)."""
    figdir = Path(outdir)
    figdir.mkdir(parents=True, exist_ok=True)
    return figdir


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def _classify_if_missing(
    df: pd.DataFrame, pred_col: str, C_cut: float, S_cut: float
) -> pd.DataFrame:
    if pred_col in df.columns:
        return df

    preds = []
    for c_val, s_val in zip(df["Coverage"], df["Spatial_Score"]):
        if np.isnan(c_val) or np.isnan(s_val):
            preds.append(ABSTAIN_LABEL)
        else:
            preds.append(classify_from_thresholds(c_val, s_val, C_cut, S_cut))
    df_out = df.copy()
    df_out[pred_col] = preds
    return df_out


def make_standard_plot_set(
    scores_df: pd.DataFrame,
    outdir: Path,
    *,
    thresholds: Mapping[str, float] | None = None,
    truth_col: str | None = None,
    pred_col: str | None = None,
    gene_col: str = "gene",
    title: str | None = None,
    debug: bool = False,
    embedding: np.ndarray | None = None,
    expr_matrix: np.ndarray | None = None,
    fg_masks: Mapping[str, np.ndarray] | None = None,
    rng_seed: int = 0,
) -> dict[str, Path]:
    """Generate the canonical plot set for simulation and kidney outputs.

    Returns a mapping from figure identifier to saved path. The function is
    resilient to missing optional inputs (truth labels, embeddings) and will
    annotate plots with "no data" when appropriate instead of failing.
    """

    outdir = Path(outdir)
    figdir = _ensure_figdir(outdir)
    thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    C_cut = float(thresholds.get("C_cut", DEFAULT_THRESHOLDS["C_cut"]))
    S_cut = float(thresholds.get("S_cut", DEFAULT_THRESHOLDS["S_cut"]))

    df = scores_df.copy()
    if pred_col is None:
        pred_col = "Archetype_pred" if "Archetype_pred" in df.columns else "archetype_pred"
    df = _classify_if_missing(df, pred_col, C_cut, S_cut)

    palette = label_palette(include_abstain=True)

    if truth_col and truth_col in df.columns:
        df[truth_col] = normalize_archetype_series(df[truth_col], allow_abstain=True)
        assert_archetype_labels(df, truth_col, allow_abstain=True)
    df[pred_col] = normalize_archetype_series(df[pred_col], allow_abstain=True)
    assert_archetype_labels(df, pred_col, allow_abstain=True)

    figures: dict[str, Path] = {}

    # Scatter
    fig_scatter = plot_cs_scatter(
        df,
        C_cut=C_cut,
        S_cut=S_cut,
        pred_col=pred_col,
        true_col=truth_col,
        palette=palette,
    )
    if title:
        fig_scatter.suptitle(title)
    figures["fig_cs_scatter"] = _save(fig_scatter, figdir / "fig_cs_scatter.png")

    # Marginals
    fig_marg, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(df["Coverage"].dropna(), bins=30, color="#4CAF50", alpha=0.8)
    axes[0].axvline(C_cut, color="black", linestyle="--", linewidth=1.2)
    axes[0].set_title("Coverage distribution")
    axes[0].set_xlabel("Coverage")
    axes[0].set_ylabel("Count")

    axes[1].hist(df["Spatial_Score"].dropna(), bins=30, color="#2196F3", alpha=0.8)
    axes[1].axvline(S_cut, color="black", linestyle="--", linewidth=1.2)
    axes[1].set_title("Spatial score distribution")
    axes[1].set_xlabel("Spatial Score")
    axes[1].set_ylabel("Count")
    fig_marg.tight_layout()
    figures["fig_cs_marginals"] = _save(fig_marg, figdir / "fig_cs_marginals.png")

    # Confusion or composition
    if truth_col and truth_col in df.columns:
        scored = df[df[pred_col] != ABSTAIN_LABEL]
        if scored.empty:
            fig_empty, ax = plt.subplots(figsize=(5, 4))
            ax.text(0.5, 0.5, "No non-abstained predictions", ha="center", va="center")
            ax.axis("off")
            figures["fig_confusion_or_composition"] = _save(
                fig_empty, figdir / "fig_confusion_or_composition.png"
            )
        else:
            cm_counts = pd.crosstab(scored[truth_col], scored[pred_col], dropna=False)
            cm_counts = cm_counts.reindex(
                index=label_order(include_abstain=False),
                columns=label_order(include_abstain=False),
                fill_value=0,
            )
            fig_cm = plot_confusion_matrix(
                cm_counts.values,
                labels=label_order(include_abstain=False),
                normalize="true",
                title="Confusion",
            )
            figures["fig_confusion_or_composition"] = _save(
                fig_cm, figdir / "fig_confusion_or_composition.png"
            )
    else:
        counts = (
            df[pred_col].value_counts().reindex(label_order(include_abstain=True), fill_value=0)
        )
        fig_comp, ax = plt.subplots(figsize=(6, 5))
        bars = ax.bar(
            range(len(counts)),
            counts.values,
            color=[palette.get(k, "#999999") for k in counts.index],
        )
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels(counts.index, rotation=45, ha="right")
        ax.set_ylabel("Genes")
        ax.set_title("Archetype composition")
        for bar, count in zip(bars, counts.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.2,
                str(int(count)),
                ha="center",
                va="bottom",
                fontsize=8,
            )
        fig_comp.tight_layout()
        figures["fig_confusion_or_composition"] = _save(
            fig_comp, figdir / "fig_confusion_or_composition.png"
        )

    archetype_examples: list[dict[str, object]] = []
    fig_examples, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    for idx, archetype in enumerate(CANONICAL_ARCHETYPES):
        ax = axes[idx]
        subset = df[df[pred_col] == archetype]
        if subset.empty:
            ax.text(0.5, 0.5, "No genes", ha="center", va="center")
            ax.axis("off")
            continue
        # Select up to 5 examples by highest spatial score
        subset_sorted = subset.sort_values("Spatial_Score", ascending=False)
        top = subset_sorted.head(5)
        genes = (
            top[gene_col]
            if gene_col in top.columns
            else pd.Series([f"gene_{i}" for i in top.index])
        )
        scores = top["Spatial_Score"]
        bars = ax.bar(range(len(top)), scores, color=palette.get(archetype, "#777777"))
        ax.set_xticks(range(len(top)))
        ax.set_xticklabels(list(genes), rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Spatial Score")
        ax.set_title(archetype)
        for g, s in zip(genes, scores):
            archetype_examples.append({"archetype": archetype, "gene": g, "Spatial_Score": s})
    fig_examples.tight_layout()
    figures["fig_archetype_examples"] = _save(fig_examples, figdir / "fig_archetype_examples.png")

    if archetype_examples:
        examples_df = pd.DataFrame(archetype_examples)
        examples_dir = outdir / "examples"
        examples_dir.mkdir(parents=True, exist_ok=True)
        examples_df.to_csv(examples_dir / "example_metadata.csv", index=False)

    # Top tables panel (bar plot of top metrics)
    fig_tables, axes_tables = plt.subplots(1, 2, figsize=(12, 5))
    axes_tables = axes_tables.flatten()
    top_cov = df.sort_values("Coverage", ascending=False).head(10)
    top_spatial = df.sort_values("Spatial_Score", ascending=False).head(10)
    cov_genes = (
        top_cov[gene_col]
        if gene_col in top_cov.columns
        else pd.Series([f"gene_{i}" for i in top_cov.index])
    )
    spatial_genes = (
        top_spatial[gene_col]
        if gene_col in top_spatial.columns
        else pd.Series([f"gene_{i}" for i in top_spatial.index])
    )

    axes_tables[0].barh(range(len(top_cov)), top_cov["Coverage"], color="#4CAF50")
    axes_tables[0].set_yticks(range(len(top_cov)))
    axes_tables[0].set_yticklabels(list(cov_genes))
    axes_tables[0].invert_yaxis()
    axes_tables[0].set_title("Top Coverage")

    axes_tables[1].barh(range(len(top_spatial)), top_spatial["Spatial_Score"], color="#2196F3")
    axes_tables[1].set_yticks(range(len(top_spatial)))
    axes_tables[1].set_yticklabels(list(spatial_genes))
    axes_tables[1].invert_yaxis()
    axes_tables[1].set_title("Top Spatial Score")

    fig_tables.tight_layout()
    figures["fig_top_tables"] = _save(fig_tables, figdir / "fig_top_tables.png")

    if debug and embedding is not None:
        fig_debug, ax_debug = plt.subplots(figsize=(6, 5))
        if fg_masks and gene_col in df.columns:
            # Use first available mask for visualization
            gene_name = df[gene_col].iloc[0]
            mask = fg_masks.get(str(gene_name)) if isinstance(fg_masks, Mapping) else None
        else:
            mask = None
        ax_debug.scatter(embedding[:, 0], embedding[:, 1], s=5, color="#cccccc", alpha=0.6)
        if mask is not None:
            ax_debug.scatter(
                embedding[mask, 0], embedding[mask, 1], s=6, color="#ff5722", alpha=0.8
            )
        ax_debug.set_title("Debug: embedding examples")
        fig_debug.tight_layout()
        figures["fig_debug_embedding"] = _save(fig_debug, figdir / "fig_debug_embedding.png")

    return figures
