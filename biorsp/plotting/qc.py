"""QC and metadata embedding figure factories."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.axes
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from biorsp.plotting.styles import DEFAULT_PLOT_STYLE, PlotStyle
from biorsp.plotting.utils import save_figure


def _is_rgba(value: Any) -> bool:
    if isinstance(value, (str, bytes)) or value is None:
        return False
    arr = np.asarray(value, dtype=object)
    if arr.ndim != 1 or arr.size not in (3, 4):
        return False
    try:
        _ = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return False
    return True


def plot_embedding_scatter(
    ax: matplotlib.axes.Axes,
    xy: np.ndarray,
    *,
    color: Any = None,
    cmap: str | None = None,
    s: float | None = None,
    alpha: float | None = None,
    label: str | None = None,
    rasterized: bool = True,
    linewidths: float = 0.0,
    edgecolors: str | None = None,
    marker: str = "o",
    style: PlotStyle = DEFAULT_PLOT_STYLE,
) -> Any:
    """Plot embedding points with shared defaults."""
    scatter_kwargs: dict[str, Any] = {
        "s": style.s_fg if s is None else s,
        "alpha": style.alpha_fg if alpha is None else alpha,
        "linewidths": linewidths,
        "edgecolors": edgecolors,
        "marker": marker,
        "rasterized": rasterized,
        "label": label,
    }
    if isinstance(color, str) or _is_rgba(color):
        scatter_kwargs["color"] = color
    else:
        scatter_kwargs["c"] = color
        if cmap is not None:
            scatter_kwargs["cmap"] = cmap
    return ax.scatter(xy[:, 0], xy[:, 1], **scatter_kwargs)


def finalize_embedding_axes(
    ax: matplotlib.axes.Axes,
    title: str,
    *,
    show_ticks: bool = False,
    xlabel: str = "UMAP1",
    ylabel: str = "UMAP2",
    style: PlotStyle = DEFAULT_PLOT_STYLE,
) -> None:
    ax.set_title(title, fontsize=style.title_fontsize)
    ax.set_xlabel(xlabel, fontsize=style.axis_label_fontsize)
    ax.set_ylabel(ylabel, fontsize=style.axis_label_fontsize)
    if not show_ticks:
        ax.set_xticks([])
        ax.set_yticks([])


def add_vantage_marker(
    ax: matplotlib.axes.Axes,
    vantage_point: tuple[float, float] | None,
    *,
    in_legend: bool = True,
    style: PlotStyle = DEFAULT_PLOT_STYLE,
) -> None:
    if vantage_point is None:
        return
    plot_embedding_scatter(
        ax=ax,
        xy=np.array([[vantage_point[0], vantage_point[1]]], dtype=float),
        color="black",
        s=style.vantage_size,
        alpha=1.0,
        label="vantage point" if in_legend else None,
        rasterized=False,
        linewidths=1.0,
        edgecolors="white",
        marker=style.vantage_marker,
        style=style,
    )


def save_numeric_umap(
    umap_xy: np.ndarray,
    values: np.ndarray,
    out_png: Path,
    title: str,
    *,
    cmap: str = "viridis",
    vantage_point: tuple[float, float] | None = None,
    colorbar_label: str | None = None,
    show_vantage_in_legend: bool = True,
    style: PlotStyle = DEFAULT_PLOT_STYLE,
) -> None:
    """Save deterministic numeric UMAP overlay plot."""
    order = np.argsort(values, kind="mergesort")
    xy = umap_xy[order]
    vals = values[order]

    fig, ax = plt.subplots(figsize=style.figsize_umap)
    pts = plot_embedding_scatter(
        ax=ax,
        xy=xy,
        color=vals,
        cmap=cmap,
        s=style.s_bg,
        alpha=style.alpha_fg,
        rasterized=True,
        style=style,
    )
    finalize_embedding_axes(ax=ax, title=title, show_ticks=False, style=style)
    add_vantage_marker(
        ax=ax,
        vantage_point=vantage_point,
        in_legend=show_vantage_in_legend,
        style=style,
    )
    if vantage_point is not None and show_vantage_in_legend:
        ax.legend(loc="best", frameon=True, fontsize=style.legend_fontsize)
    cbar = fig.colorbar(
        pts,
        ax=ax,
        shrink=style.colorbar_shrink,
        pad=style.colorbar_pad,
    )
    cbar.set_label(colorbar_label or "value", fontsize=style.axis_label_fontsize)
    fig.tight_layout()
    save_figure(fig, out_png, style=style)


def plot_umap_vantage_diagnostic(
    umap_xy: np.ndarray,
    vantage_point: tuple[float, float],
    out_png: Path,
    *,
    style: PlotStyle = DEFAULT_PLOT_STYLE,
) -> None:
    fig, ax = plt.subplots(figsize=style.figsize_umap)
    plot_embedding_scatter(
        ax=ax,
        xy=umap_xy,
        color="lightgray",
        s=style.s_bg,
        alpha=style.alpha_bg,
        rasterized=True,
        style=style,
    )
    add_vantage_marker(ax=ax, vantage_point=vantage_point, in_legend=True, style=style)
    finalize_embedding_axes(
        ax=ax, title="UMAP vantage point diagnostic", show_ticks=False, style=style
    )
    ax.legend(loc="best", frameon=True, fontsize=style.legend_fontsize)
    fig.tight_layout()
    save_figure(fig, out_png, style=style)


def _maybe_numeric_sort(categories: list[str]) -> list[str]:
    parsed: list[tuple[float, str] | None] = []
    for cat in categories:
        try:
            parsed.append((float(cat), cat))
        except ValueError:
            parsed.append(None)
    if any(item is None for item in parsed):
        return sorted(categories)
    pairs = [item for item in parsed if item is not None]
    pairs.sort(key=lambda x: (x[0], x[1]))
    return [cat for _, cat in pairs]


def _compressed_categorical_labels(
    labels: pd.Series,
    *,
    style: PlotStyle = DEFAULT_PLOT_STYLE,
    numeric_categories: bool = False,
) -> tuple[pd.Series, list[str]]:
    labels_str = labels.astype("string").fillna("NA").astype(str)
    counts = (
        labels_str.value_counts(sort=False)
        .rename_axis("category")
        .reset_index(name="count")
        .sort_values(
            by=["count", "category"], ascending=[False, True], kind="mergesort"
        )
    )
    ordered_categories = counts["category"].astype(str).tolist()
    if numeric_categories:
        ordered_categories = _maybe_numeric_sort(ordered_categories)
    if int(len(ordered_categories)) <= style.categorical_legend_trigger:
        return labels_str, ordered_categories
    top = ordered_categories[: style.categorical_legend_top_k]
    n_more = int(len(ordered_categories) - len(top))
    other_label = f"Other ({n_more} categories)"
    compressed = labels_str.where(labels_str.isin(top), other_label)
    return compressed, top + [other_label]


def _palette_for_categories(
    categories: list[str],
) -> dict[str, tuple[float, float, float, float]]:
    cmap = plt.get_cmap("tab20")
    return {cat: cmap(i % 20) for i, cat in enumerate(categories)}


def _annotate_medians_with_spacing(
    ax: matplotlib.axes.Axes,
    xy: np.ndarray,
    labels: pd.Series,
    categories: list[str],
    *,
    style: PlotStyle = DEFAULT_PLOT_STYLE,
) -> None:
    if len(categories) > style.categorical_annotation_max:
        return
    x_span = float(np.ptp(xy[:, 0])) if xy.shape[0] > 0 else 1.0
    y_span = float(np.ptp(xy[:, 1])) if xy.shape[0] > 0 else 1.0
    min_d2 = (max(x_span, y_span) * style.annotation_min_dist_frac) ** 2
    placed: list[tuple[float, float]] = []
    labels_np = labels.to_numpy()
    for cat in categories:
        mask = labels_np == cat
        sub = xy[mask]
        if sub.shape[0] == 0:
            continue
        cx = float(np.median(sub[:, 0]))
        cy = float(np.median(sub[:, 1]))
        for _ in range(30):
            if all(((cx - px) ** 2 + (cy - py) ** 2) >= min_d2 for px, py in placed):
                break
            cy += 0.01 * max(y_span, 1.0)
        placed.append((cx, cy))
        txt = ax.text(
            cx, cy, str(cat), fontsize=9, color="black", ha="center", va="center"
        )
        txt.set_path_effects([pe.withStroke(linewidth=2.0, foreground="white")])


def plot_categorical_umap(
    umap_xy: np.ndarray,
    labels: pd.Series,
    title: str,
    outpath: Path,
    *,
    vantage_point: tuple[float, float] | None = None,
    annotate_cluster_medians: bool = False,
    numeric_categories: bool = False,
    style: PlotStyle = DEFAULT_PLOT_STYLE,
) -> None:
    """Plot deterministic categorical UMAP panels."""
    labels_str, categories = _compressed_categorical_labels(
        labels=labels,
        style=style,
        numeric_categories=numeric_categories,
    )
    color_map = _palette_for_categories(categories)
    if categories and categories[-1].startswith("Other ("):
        color_map[categories[-1]] = (0.7, 0.7, 0.7, 1.0)

    fig, ax = plt.subplots(figsize=style.figsize_categorical)
    for cat in categories:
        mask = labels_str == cat
        xy = umap_xy[mask.to_numpy()]
        if xy.shape[0] == 0:
            continue
        plot_embedding_scatter(
            ax=ax,
            xy=xy,
            color=color_map[cat],
            s=style.s_fg,
            alpha=style.alpha_fg,
            linewidths=0,
            rasterized=True,
            label=cat,
            style=style,
        )

    if annotate_cluster_medians:
        _annotate_medians_with_spacing(
            ax=ax,
            xy=umap_xy,
            labels=labels_str,
            categories=categories,
            style=style,
        )

    add_vantage_marker(
        ax=ax,
        vantage_point=vantage_point,
        in_legend=vantage_point is not None,
        style=style,
    )
    finalize_embedding_axes(ax=ax, title=title, show_ticks=False, style=style)

    handles, legend_labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        legend_labels,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        fontsize=style.legend_fontsize,
        frameon=True,
        ncol=1,
    )
    fig.tight_layout()
    save_figure(fig, outpath, style=style, bbox_tight=True)


def write_cluster_celltype_counts(
    cluster_labels: pd.Series,
    celltype_labels: pd.Series,
    *,
    cluster_key: str,
    celltype_key: str,
    out_csv: Path,
) -> pd.DataFrame:
    """Write deterministic cluster/celltype contingency table used in metadata panels."""
    counts_df = (
        pd.DataFrame(
            {
                cluster_key: cluster_labels.astype("string"),
                celltype_key: celltype_labels.astype("string"),
            }
        )
        .groupby([cluster_key, celltype_key], observed=False)
        .size()
        .reset_index(name="n_cells")
        .sort_values(
            by=[cluster_key, "n_cells"], ascending=[True, False], kind="mergesort"
        )
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    counts_df.to_csv(out_csv, index=False)
    return counts_df
