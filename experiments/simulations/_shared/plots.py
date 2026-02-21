"""Shared matplotlib plotting utilities for simulation reports."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .io import ensure_dir


def savefig(fig: plt.Figure, path: str | Path, dpi: int = 200) -> Path:
    """Save figure to disk and close layout cleanly."""
    out = Path(path)
    ensure_dir(out.parent)
    fig.savefig(out, dpi=int(dpi), bbox_inches="tight")
    return out


def save_fig(fig: plt.Figure, path: str | Path, dpi: int = 220) -> Path:
    """Standardized save wrapper alias."""
    return savefig(fig, path, dpi=int(dpi))


def annotate_n(ax: plt.Axes, n: int, loc: str = "topright") -> None:
    """Annotate sample size on an axis."""
    loc_map = {
        "topright": (0.98, 0.95, "right", "top"),
        "topleft": (0.02, 0.95, "left", "top"),
        "bottomright": (0.98, 0.05, "right", "bottom"),
        "bottomleft": (0.02, 0.05, "left", "bottom"),
    }
    x, y, ha, va = loc_map.get(loc, loc_map["topright"])
    ax.text(x, y, f"n={int(n)}", transform=ax.transAxes, ha=ha, va=va, fontsize=7)


def qq_plot(ax: plt.Axes, pvals: np.ndarray, label: str | None = None) -> None:
    """Draw Uniform(0,1) QQ plot for p-values."""
    p = np.asarray(pvals, dtype=float)
    p = p[np.isfinite(p)]
    if p.size >= 3:
        p_sorted = np.sort(p)
        expected = (np.arange(1, p_sorted.size + 1, dtype=float) - 0.5) / p_sorted.size
        ax.plot(expected, p_sorted, ".", markersize=2.0, alpha=0.8, label=label)
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="#444444", linewidth=0.8)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)


def p_hist(
    ax: plt.Axes, pvals: np.ndarray, bins: int = 20, density: bool = True
) -> None:
    """Draw a p-value histogram with optional uniform-density reference line."""
    p = np.asarray(pvals, dtype=float)
    p = p[np.isfinite(p)]
    if p.size > 0:
        edges = np.linspace(0.0, 1.0, int(bins) + 1)
        ax.hist(
            p,
            bins=edges,
            density=bool(density),
            color="#4C78A8",
            alpha=0.85,
            edgecolor="white",
            linewidth=0.3,
        )
    if density:
        ax.axhline(1.0, linestyle="--", color="#555555", linewidth=0.7)
    ax.set_xlim(0.0, 1.0)


def heatmap(
    ax: plt.Axes,
    mat: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    value_fmt: str = "{:.3f}",
    cmap: str | None = None,
) -> None:
    """Plot an annotated heatmap."""
    arr = np.asarray(mat, dtype=float)
    im = ax.imshow(arr, aspect="auto", cmap=cmap or "viridis")
    ax.set_title(title)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            txt = "NA" if not np.isfinite(v) else value_fmt.format(v)
            color = "white" if np.isfinite(v) and v > np.nanmean(arr) else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7, color=color)
    return im


def confusion_matrix_plot(
    ax: plt.Axes,
    mat: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    cmap: str = "Blues",
) -> None:
    """Render a confusion matrix with integer annotations."""
    arr = np.asarray(mat, dtype=float)
    im = ax.imshow(arr, cmap=str(cmap), aspect="auto")
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title(title)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val = arr[i, j]
            txt = "NA" if not np.isfinite(val) else str(int(round(val)))
            ax.text(j, i, txt, ha="center", va="center", fontsize=7)
    plt.colorbar(im, ax=ax, shrink=0.8)


def scatter_diagnostic(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    color: str = "#4C78A8",
) -> None:
    """Simple standardized scatter diagnostic."""
    xx = np.asarray(x, dtype=float)
    yy = np.asarray(y, dtype=float)
    m = np.isfinite(xx) & np.isfinite(yy)
    ax.scatter(xx[m], yy[m], s=10, alpha=0.6, color=color, linewidths=0)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def plot_embedding_with_foreground(
    X: np.ndarray,
    f: np.ndarray,
    *,
    ax: plt.Axes | None = None,
    title: str | None = None,
    s: float = 8.0,
    alpha_bg: float = 0.25,
    alpha_fg: float = 0.85,
) -> tuple[plt.Figure, plt.Axes]:
    """Scatter embedding coordinates colored by foreground membership."""
    coords = np.asarray(X, dtype=float)
    fg = np.asarray(f, dtype=bool).ravel()
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError("X must have shape (n_cells, 2+) for embedding plotting.")
    if coords.shape[0] != fg.size:
        raise ValueError("X and foreground mask must have matching length.")

    fig: plt.Figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.2, 4.6), constrained_layout=False)
    else:
        fig = ax.figure

    bg = ~fg
    ax.scatter(
        coords[bg, 0],
        coords[bg, 1],
        s=s,
        c="#BDBDBD",
        alpha=float(alpha_bg),
        linewidths=0.0,
        label="background",
    )
    ax.scatter(
        coords[fg, 0],
        coords[fg, 1],
        s=s,
        c="#D62728",
        alpha=float(alpha_fg),
        linewidths=0.0,
        label="foreground",
    )
    ax.set_xlabel("embedding-1")
    ax.set_ylabel("embedding-2")
    ax.set_title(title or "Embedding colored by foreground")
    ax.legend(loc="best", frameon=False, markerscale=1.3)
    return fig, ax


def plot_rsp_polar(
    theta_centers: np.ndarray,
    rsp_values: np.ndarray,
    *,
    ax: plt.Axes | None = None,
    title: str | None = None,
    color: str = "#1F77B4",
    linewidth: float = 1.6,
    label: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot RSP profile on a polar axis with wraparound continuity."""
    theta = np.asarray(theta_centers, dtype=float).ravel()
    rsp = np.asarray(rsp_values, dtype=float).ravel()
    if theta.size != rsp.size or theta.size < 2:
        raise ValueError("theta_centers and rsp_values must have same length >= 2.")

    order = np.argsort(theta)
    theta = theta[order]
    rsp = rsp[order]
    theta_wrapped = np.concatenate([theta, [theta[0] + 2.0 * np.pi]])
    rsp_wrapped = np.concatenate([rsp, [rsp[0]]])

    fig: plt.Figure
    if ax is None:
        fig, ax = plt.subplots(
            figsize=(5.2, 4.8),
            subplot_kw={"projection": "polar"},
            constrained_layout=False,
        )
    else:
        fig = ax.figure
        if getattr(ax, "name", "") != "polar":
            raise ValueError("plot_rsp_polar requires a polar-projection axis.")

    ax.plot(
        theta_wrapped, rsp_wrapped, color=color, linewidth=float(linewidth), label=label
    )
    if label is not None:
        ax.legend(loc="upper right", frameon=False)
    ax.set_theta_direction(-1)
    ax.set_theta_zero_location("N")
    ax.set_title(title or "RSP profile (polar)")
    return fig, ax


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson confidence interval for a proportion."""
    if int(n) <= 0:
        return float("nan"), float("nan")
    from scipy.stats import norm

    z = float(norm.ppf(1.0 - float(alpha) / 2.0))
    p_hat = float(k) / float(n)
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p_hat + z2 / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt((p_hat * (1.0 - p_hat) / n) + (z2 / (4.0 * n * n)))
    return max(0.0, center - half), min(1.0, center + half)
