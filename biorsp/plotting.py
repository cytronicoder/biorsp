"""
Plotting module for BioRSP.

Provides visualization functions:
- Radar plots for RSP profiles
- Embedding scatter plots
- Summary visualizations
"""

from __future__ import annotations

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .radar import RadarResult
from .summaries import ScalarSummaries


def _as_1d(a) -> np.ndarray:
    return np.asarray(a).reshape(-1)


def _maybe_degrees_to_radians(theta: np.ndarray) -> np.ndarray:
    """Heuristic: if angles look like degrees, convert to radians."""
    finite = np.isfinite(theta)
    if not np.any(finite):
        return theta
    tmax = float(np.nanmax(theta[finite]))
    # If values exceed ~2.4*pi (~432 deg), they are almost surely degrees
    if tmax > (2.0 * np.pi * 1.2):
        return np.deg2rad(theta)
    return theta


def _prepare_polar(theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize angles to [0, 2pi), sort, and compute sector widths (one per theta).
    Handles singleton theta and degenerate/duplicate angles robustly.
    """
    theta = _maybe_degrees_to_radians(theta.astype(float, copy=False))
    theta = np.mod(theta, 2 * np.pi)

    if theta.size == 0:
        return theta, theta

    if theta.size == 1:
        return theta, np.array([2 * np.pi], dtype=float)

    order = np.argsort(theta)
    theta = theta[order]

    # Compute widths via midpoint method; fallback to uniform widths if degenerate
    theta_ext = np.concatenate([theta, [theta[0] + 2 * np.pi]])
    mid_right = 0.5 * (theta_ext[:-1] + theta_ext[1:])
    left_edges = np.concatenate([[mid_right[-1] - 2 * np.pi], mid_right[:-1]])
    widths = mid_right - left_edges

    # Guard against duplicates / numerical pathologies
    if not np.all(np.isfinite(widths)) or np.any(widths <= 0):
        widths = np.full(theta.size, 2 * np.pi / theta.size, dtype=float)

    return theta, widths


def _ensure_polar_ax(ax: Optional[plt.Axes]) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(subplot_kw={"projection": "polar"})
        return ax
    # If a non-polar axis is passed, fail early with a clear message.
    if getattr(ax, "name", "") != "polar":
        raise ValueError("plot_radar requires a polar axis (projection='polar').")
    return ax


def _set_default_polar_style(ax: plt.Axes) -> None:
    # Reasonable defaults; do not enforce colors/styles beyond readability.
    ax.set_theta_direction(-1)  # clockwise
    ax.set_theta_zero_location("N")  # 0° at top


def plot_radar(
    radar: RadarResult,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    color: str = "b",
    alpha: float = 0.5,
    mode: str = "combined",
    **kwargs,
) -> plt.Axes:
    """
    Plot RSP radar profile.

    Args:
        radar: RadarResult object.
        ax: Matplotlib axes (must be polar). If None, created.
        title: Plot title.
        color: Line/Fill color.
        alpha: Fill transparency.
        mode: Plotting mode.
            - "combined": Enrichment (R>0) and depletion (R<0 shown as positive magnitude) together.
            - "enrichment": Only positive RSP values (R > 0).
            - "depletion": Only negative RSP values (R < 0) as positive magnitudes.
            - "relative": Signed RSP visualized robustly as magnitude wedges colored by sign.
        **kwargs: Additional arguments forwarded to matplotlib (e.g., linewidth).

    Returns:
        ax: The axes object.
    """
    ax = _ensure_polar_ax(ax)
    _set_default_polar_style(ax)

    theta_raw = _as_1d(radar.centers)
    r_raw = _as_1d(radar.rsp)

    if theta_raw.size == 0 or r_raw.size == 0:
        if title:
            ax.set_title(f"{title} (No data)")
        ax.set_ylim(0, 1)
        return ax

    if theta_raw.size != r_raw.size:
        raise ValueError(
            f"radar.centers (n={theta_raw.size}) and radar.rsp (n={r_raw.size}) must have the same length."
        )

    # Compute y-limit using finite values only
    finite_r = np.isfinite(r_raw)
    if not np.any(finite_r):
        msg = "No valid sectors (insufficient foreground/background counts)"
        ax.set_title(f"{title}\n{msg}" if title else msg, fontsize=20)
        ax.set_ylim(0, 1)
        return ax

    max_mag = float(np.nanmax(np.abs(r_raw[finite_r])))
    y_top = max(1e-6, max_mag * 1.05)

    # Normalize/sort theta and carry r accordingly
    theta_norm = _maybe_degrees_to_radians(theta_raw.astype(float, copy=False))
    theta_norm = np.mod(theta_norm, 2 * np.pi)

    if theta_norm.size == 1:
        order = np.array([0], dtype=int)
    else:
        order = np.argsort(theta_norm)

    theta = theta_norm[order]
    r = r_raw[order]

    # Compute widths on the sorted theta
    theta_sorted, widths = _prepare_polar(theta)
    # _prepare_polar returns theta already sorted; keep alignment
    theta = theta_sorted  # clarity

    # If _prepare_polar had to fall back or adjust sorting, keep r aligned with theta.
    # (It only sorts theta; we already sorted theta and r together.)
    # widths corresponds to current theta ordering.

    # Mask invalid values consistently
    valid = np.isfinite(r)
    theta_v = theta[valid]
    widths_v = widths[valid]
    r_v = r[valid]

    if theta_v.size == 0:
        msg = "No valid sectors (insufficient foreground/background counts)"
        ax.set_title(f"{title}\n{msg}" if title else msg, fontsize=20)
        ax.set_ylim(0, 1)
        return ax

    lw = float(kwargs.pop("linewidth", 1.25))

    def _outline(ax_obj: plt.Axes, th: np.ndarray, vals: np.ndarray, **plot_kw) -> None:
        if th.size == 0:
            return
        if th.size == 1:
            # One point: draw a short radial marker to make it visible
            ax_obj.plot([th[0], th[0]], [0.0, float(vals[0])], **plot_kw)
            return
        th_closed = np.concatenate([th, [th[0] + 2 * np.pi]])
        v_closed = np.concatenate([vals, [vals[0]]])
        ax_obj.plot(th_closed, v_closed, **plot_kw)

    mode = mode.lower().strip()

    if mode == "relative":
        heights = np.abs(r_v)
        bar_colors = np.where(r_v >= 0, color, "r")

        ax.bar(
            theta_v,
            heights,
            width=widths_v,
            bottom=0.0,
            align="center",
            color=bar_colors,
            alpha=alpha,
            edgecolor=bar_colors,
            linewidth=1.0,
        )
        _outline(ax, theta_v, heights, color="k", linewidth=lw)

        ax.set_ylim(0, y_top)
        if title:
            ax.set_title(f"{title} (relative)", fontsize=20)

        # Legend only if both signs exist
        try:
            import matplotlib.patches as mpatches

            handles, labels = [], []
            if np.any(r_v > 0):
                handles.append(mpatches.Patch(color=color, alpha=alpha))
                labels.append("Enrichment (R > 0)")
            if np.any(r_v < 0):
                handles.append(mpatches.Patch(color="r", alpha=alpha))
                labels.append("Depletion (R < 0)")
            if handles:
                ax.legend(handles, labels, loc="lower left", fontsize=18)
        except Exception:
            pass

        return ax

    if mode == "enrichment":
        r_pos = np.maximum(0.0, r_v)
        mask = r_pos > 0
        theta_m = theta_v[mask]
        widths_m = widths_v[mask]
        r_pos_m = r_pos[mask]

        if theta_m.size:
            ax.bar(
                theta_m,
                r_pos_m,
                width=widths_m,
                bottom=0.0,
                align="center",
                color=color,
                alpha=alpha,
                edgecolor=color,
                linewidth=1.0,
            )
            _outline(ax, theta_m, r_pos_m, color=color, linewidth=lw)
        ax.set_ylim(0, y_top)
        if title:
            ax.set_title(title + " RSP" if "RSP" not in title else title, fontsize=20)
        return ax

    if mode == "depletion":
        r_neg = np.maximum(0.0, -r_v)
        mask = r_neg > 0
        theta_m = theta_v[mask]
        widths_m = widths_v[mask]
        r_neg_m = r_neg[mask]

        if theta_m.size:
            ax.bar(
                theta_m,
                r_neg_m,
                width=widths_m,
                bottom=0.0,
                align="center",
                color="r",
                alpha=alpha,
                edgecolor="r",
                linewidth=1.0,
            )
            _outline(ax, theta_m, r_neg_m, color="r", linewidth=lw, linestyle="--")
        ax.set_ylim(0, y_top)
        if title:
            ax.set_title(title + " RSP" if "RSP" not in title else title, fontsize=20)
        return ax

    if mode == "combined":
        r_pos = np.maximum(0.0, r_v)
        r_neg = np.maximum(0.0, -r_v)

        mask_pos = r_pos > 0
        mask_neg = r_neg > 0

        if np.any(mask_pos):
            ax.bar(
                theta_v[mask_pos],
                r_pos[mask_pos],
                width=widths_v[mask_pos],
                bottom=0.0,
                align="center",
                color=color,
                alpha=alpha,
                edgecolor=color,
                linewidth=1.0,
            )
            _outline(ax, theta_v[mask_pos], r_pos[mask_pos], color=color, linewidth=lw)

        if np.any(mask_neg):
            ax.bar(
                theta_v[mask_neg],
                r_neg[mask_neg],
                width=widths_v[mask_neg],
                bottom=0.0,
                align="center",
                color="r",
                alpha=alpha,
                edgecolor="r",
                linewidth=1.0,
            )
            _outline(
                ax,
                theta_v[mask_neg],
                r_neg[mask_neg],
                color="r",
                linewidth=lw,
                linestyle="--",
            )

        ax.set_ylim(0, y_top)
        if title:
            ax.set_title(title + " RSP" if "RSP" not in title else title, fontsize=20)

        # Legend if any plotted
        try:
            import matplotlib.patches as mpatches

            handles, labels = [], []
            if np.any(mask_pos):
                handles.append(mpatches.Patch(color=color, alpha=alpha))
                labels.append("Enrichment")
            if np.any(mask_neg):
                handles.append(mpatches.Patch(color="r", alpha=alpha))
                labels.append("Depletion")
            if handles:
                ax.legend(handles, labels, loc="lower left", fontsize=18)
        except Exception:
            pass

        return ax

    raise ValueError(f"Unknown mode: {mode}")


def plot_radar_absolute(
    radar: RadarResult,
    fig: Optional[plt.Figure] = None,
    title: Optional[str] = None,
    color: str = "b",
    alpha: float = 0.5,
    **kwargs,
) -> plt.Figure:
    """
    Generate both enrichment and depletion radar plots side-by-side.

    Args:
        radar: RadarResult object.
        fig: Matplotlib figure. If None, created.
        title: Overall title.
        color: Line/Fill color for enrichment.
        alpha: Fill transparency.
        **kwargs: Additional arguments for plot_radar.

    Returns:
        fig: The figure object.
    """
    if fig is None:
        fig = plt.figure(figsize=(10, 5))

    ax1 = fig.add_subplot(121, projection="polar")
    plot_radar(
        radar,
        ax=ax1,
        title="Enrichment RSP ($R > 0$)",
        color=color,
        alpha=alpha,
        mode="enrichment",
        **kwargs,
    )

    ax2 = fig.add_subplot(122, projection="polar")
    plot_radar(
        radar,
        ax=ax2,
        title="Depletion RSP ($R < 0$)",
        color="r",
        alpha=alpha,
        mode="depletion",
        **kwargs,
    )

    if title:
        fig.suptitle(title + " RSP" if "RSP" not in title else title, fontsize=20)

    # Tight layout without crushing suptitle
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


def plot_embedding(
    Z: np.ndarray,
    c: Optional[np.ndarray] = None,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    cmap: str = "viridis",
    s: int = 10,
    **kwargs,
) -> plt.Axes:
    """
    Scatter plot of embedding.

    Args:
        Z: (N, 2) embedding coordinates.
        c: (N,) color values (e.g. expression).
        ax: Matplotlib axes. If None, created.
        title: Plot title.
        cmap: Colormap.
        s: Marker size.
        **kwargs: Additional arguments for scatter.

    Returns:
        ax: The axes object.
    """
    Z = np.asarray(Z)
    if Z.ndim != 2 or Z.shape[1] != 2:
        raise ValueError(f"Z must have shape (N, 2); got {Z.shape}")

    if c is not None:
        c = _as_1d(c)
        if c.size != Z.shape[0]:
            raise ValueError(f"c must have length N={Z.shape[0]}; got {c.size}")

    if ax is None:
        _, ax = plt.subplots()

    sc = ax.scatter(Z[:, 0], Z[:, 1], c=c, s=s, cmap=cmap, **kwargs)

    if c is not None:
        plt.colorbar(sc, ax=ax)

    if title:
        ax.set_title(title)

    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")

    return ax


def plot_summary(summary: ScalarSummaries, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Display scalar summaries as text.

    Args:
        summary: ScalarSummaries object.
        ax: Matplotlib axes. If None, created.

    Returns:
        ax: The axes object.
    """
    if ax is None:
        _, ax = plt.subplots()

    ax.axis("off")

    text = (
        f"Max RSP: {summary.max_rsp:.3f}\n"
        f"Min RSP: {summary.min_rsp:.3f}\n"
        f"RMS Anisotropy: {summary.anisotropy:.3f}\n"
        f"Integrated RSP: {summary.integrated_rsp:.3f}\n"
        f"Peak Distal Angle: {np.degrees(summary.peak_distal_angle):.1f}°\n"
        f"Peak Proximal Angle: {np.degrees(summary.peak_proximal_angle):.1f}°\n"
        f"Peak Extremal Angle: {np.degrees(summary.peak_extremal_angle):.1f}°"
    )

    ax.text(0.1, 0.5, text, fontsize=20, transform=ax.transAxes, va="center")
    return ax


__all__ = ["plot_radar", "plot_radar_absolute", "plot_embedding", "plot_summary"]
