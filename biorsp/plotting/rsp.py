"""RSP plotting primitives bound to `RSPResult` objects."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from biorsp.core.compute import compute_rsp
from biorsp.core.types import RSPConfig, RSPResult
from biorsp.plotting.styles import DEFAULT_PLOT_STYLE, PlotStyle


def _configure_polar_axis(ax: plt.Axes, r_values: np.ndarray) -> None:
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    # Keep cardinal ticks only to reduce clutter while preserving orientation cues.
    ax.set_thetagrids(np.arange(0, 360, 90))
    r_min = float(np.nanmin(r_values))
    r_max = float(np.nanmax(r_values))
    if np.isfinite(r_min) and np.isfinite(r_max):
        if np.isclose(r_min, r_max):
            ax.set_rticks([r_min])
        else:
            ax.set_rticks(np.linspace(r_min, r_max, num=5))
    ax.yaxis.set_tick_params(labelsize=8)
    ax.xaxis.set_tick_params(labelsize=9)
    ax.grid(True, alpha=0.4)


def plot_rsp(
    result: RSPResult,
    *,
    title: str | None = None,
    ax: plt.Axes | None = None,
    line_color: str = "#8B0000",
    fill_color: str = "#F08080",
    fill_alpha: float = 0.30,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot RSP curve from precomputed `RSPResult` (no recomputation)."""
    theta = np.asarray(result.theta, dtype=float).ravel()
    r_theta = np.asarray(result.R_theta, dtype=float).ravel()
    if theta.size != r_theta.size:
        raise ValueError("result.theta and result.R_theta length mismatch.")

    theta_closed = np.concatenate([theta, theta[:1]])
    r_closed = np.concatenate([r_theta, r_theta[:1]])

    if ax is None:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="polar")
    else:
        fig = ax.figure
        if ax.name != "polar":
            raise ValueError("Provided axis must be polar.")

    ax.plot(theta_closed, r_closed, lw=2, color=line_color)
    ax.fill_between(theta_closed, 0.0, r_closed, color=fill_color, alpha=fill_alpha)
    _configure_polar_axis(ax, r_closed)
    ax.set_title(title or f"RSP: {result.feature_label}")
    fig.tight_layout()
    return fig, ax


def plot_rsp_to_file(
    result: RSPResult,
    out_png: str | Path,
    *,
    title: str | None = None,
    style: PlotStyle = DEFAULT_PLOT_STYLE,
) -> None:
    out_path = Path(out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, _ = plot_rsp(result, title=title)
    fig.savefig(out_path.as_posix(), dpi=style.dpi)
    plt.close(fig)


def plot_umap_rsp_pair(
    *,
    embedding_xy: np.ndarray,
    expr: np.ndarray,
    result: RSPResult,
    out_png: str | Path | None = None,
    title_prefix: str | None = None,
    style: PlotStyle = DEFAULT_PLOT_STYLE,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]] | None:
    """Create side-by-side UMAP (left) + RSP (right) panel."""
    xy = np.asarray(embedding_xy, dtype=float)
    x = np.asarray(expr, dtype=float).ravel()
    if xy.ndim != 2 or xy.shape[1] < 2:
        raise ValueError("embedding_xy must have shape (N, 2+).")
    if x.size != xy.shape[0]:
        raise ValueError("expr length must match embedding rows.")

    fig = plt.figure(figsize=(12, 5))
    ax_l = fig.add_subplot(1, 2, 1)
    ax_r = fig.add_subplot(1, 2, 2, projection="polar")

    ax_l.scatter(
        xy[:, 0],
        xy[:, 1],
        c="lightgray",
        s=style.s_bg,
        alpha=style.alpha_bg,
        linewidths=0,
    )
    order = np.argsort(x)
    sc = ax_l.scatter(
        xy[order, 0],
        xy[order, 1],
        c=x[order],
        s=style.s_fg,
        alpha=style.alpha_fg,
        linewidths=0,
        cmap="Reds",
    )
    ctr = result.metadata.get("center_xy", None)
    if isinstance(ctr, list) and len(ctr) == 2:
        ax_l.scatter(
            [float(ctr[0])],
            [float(ctr[1])],
            marker=style.vantage_marker,
            s=style.vantage_size,
            color="black",
            edgecolors="white",
            linewidths=1.0,
            zorder=10,
        )
    cbar = fig.colorbar(sc, ax=ax_l, fraction=0.046, pad=0.04)
    cbar.set_label("expression")
    ax_l.set_xticks([])
    ax_l.set_yticks([])
    ax_l.set_title(
        f"{title_prefix + ': ' if title_prefix else ''}UMAP: {result.feature_label}"
    )

    plot_rsp(result, title=f"RSP: {result.feature_label}", ax=ax_r)
    fig.tight_layout()

    if out_png is not None:
        out_path = Path(out_png)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path.as_posix(), dpi=style.dpi)
        plt.close(fig)
        return None
    return fig, (ax_l, ax_r)


def compute_and_plot_pair(
    *,
    expr: np.ndarray,
    embedding_xy: np.ndarray,
    config: RSPConfig,
    feature_label: str,
    feature_index: int | None = None,
    out_png: str | Path,
) -> RSPResult:
    """Convenience function used by pipeline code."""
    result = compute_rsp(
        expr=np.asarray(expr, dtype=float),
        embedding_xy=np.asarray(embedding_xy, dtype=float),
        config=config,
        feature_label=feature_label,
        feature_index=feature_index,
    )
    plot_umap_rsp_pair(
        embedding_xy=np.asarray(embedding_xy, dtype=float),
        expr=np.asarray(expr, dtype=float),
        result=result,
        out_png=out_png,
    )
    return result
