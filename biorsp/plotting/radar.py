"""
Plotting module for BioRSP.

Provides visualization functions:
- Radar plots for RSP profiles
- Embedding scatter plots
- Summary visualizations
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from biorsp.core.summaries import ScalarSummaries
from biorsp.core.typing import RadarResult
from biorsp.utils.helpers import _as_1d


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


def _prepare_polar(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Normalize angles to [0, 2pi), sort, and compute sector widths (one per theta).

    Handles edge cases: singleton arrays, degenerate/duplicate angles via fallback
    to uniform widths when midpoint method produces non-positive or non-finite values.
    """
    theta = _maybe_degrees_to_radians(theta.astype(float, copy=False))
    theta = np.mod(theta, 2 * np.pi)

    if theta.size == 0:
        return theta, theta

    if theta.size == 1:
        return theta, np.array([2 * np.pi], dtype=float)

    order = np.argsort(theta)
    theta = theta[order]

    theta_ext = np.concatenate([theta, [theta[0] + 2 * np.pi]])
    mid_right = 0.5 * (theta_ext[:-1] + theta_ext[1:])
    left_edges = np.concatenate([[mid_right[-1] - 2 * np.pi], mid_right[:-1]])
    widths = mid_right - left_edges

    if not np.all(np.isfinite(widths)) or np.any(widths <= 0):
        widths = np.full(theta.size, 2 * np.pi / theta.size, dtype=float)

    return theta, widths


def _ensure_polar_ax(ax: plt.Axes | None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(subplot_kw={"projection": "polar"})
        return ax
    # If a non-polar axis is passed, fail early with a clear message.
    if getattr(ax, "name", "") != "polar":
        raise ValueError("plot_radar requires a polar axis (projection='polar').")
    return ax


def _set_default_polar_style(ax: plt.Axes) -> None:
    # Set mathematical convention: 0° at East, increasing counter-clockwise.
    ax.set_theta_direction(1)
    ax.set_theta_zero_location("E")


def _draw_segmented_rsp(
    ax: plt.Axes,
    th: np.ndarray,
    vals: np.ndarray,
    fill: bool = True,
    color: str = "b",
    alpha: float = 0.5,
    line_color: str | None = None,
    **kwargs,
) -> None:
    """
    Draw RSP outline and fill, handling NaN gaps and wrap-around.

    Splits continuous segments at NaN boundaries. Merges segments across
    the 0/2pi boundary if both ends are finite (polar wrap-around).
    """
    if th.size == 0:
        return

    mask = np.isfinite(vals)
    if not np.any(mask):
        return

    idx = np.where(mask)[0]
    if len(idx) == 0:
        return

    splits = np.where(np.diff(idx) > 1)[0] + 1
    seg_indices = np.split(idx, splits)
    segments = [s.tolist() for s in seg_indices]

    if len(segments) > 1 and mask[0] and mask[-1]:
        last_seg = segments.pop(-1)
        first_seg = segments.pop(0)
        merged_seg = last_seg + first_seg
        segments.append(merged_seg)
    elif len(segments) == 1 and mask.all():
        segments[0].append(segments[0][0])

    for seg in segments:
        th_seg = th[seg].astype(float, copy=True)
        v_seg = vals[seg].astype(float, copy=True)

        for i in range(1, len(th_seg)):
            if th_seg[i] < th_seg[i - 1]:
                th_seg[i:] += 2 * np.pi

        l_color = line_color if line_color is not None else color
        line_kw = kwargs.copy()
        line_kw.pop("hatch", None)
        ax.plot(th_seg, v_seg, color=l_color, **line_kw)

        if fill:
            fill_kw = {"color": color, "alpha": alpha}
            if "hatch" in kwargs:
                fill_kw["hatch"] = kwargs["hatch"]

            th_fill = np.concatenate([[th_seg[0]], th_seg, [th_seg[-1]]])
            v_fill = np.concatenate([[0], v_seg, [0]])
            ax.fill(th_fill, v_fill, edgecolor="none", **fill_kw)


def plot_localization_scatter(
    feature_results: dict,
    ax: plt.Axes | None = None,
    show_archetypes: bool = True,
    color_by_sign: bool = True,
    delta_deg: float | None = None,
    **kwargs,
) -> plt.Axes:
    """
    Plot anisotropy (A_g) vs localization (L_g) to distinguish spatial phenotypes.

    Parameters
    ----------
    feature_results : dict
        Mapping of feature name to FeatureResult.
    ax : plt.Axes, optional
        Axes to plot on.
    show_archetypes : bool, optional
        Whether to annotate archetypes (rim/core, wedge, null).
    color_by_sign : bool, optional
        Whether to color points by the sign of the extremal peak.
    delta_deg : float, optional
        The sector width used. If >= 90, "Wedge" labels are marked as potential artifacts.
    **kwargs
        Passed to ax.scatter.

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    anisotropy = []
    localization = []
    signs = []

    for fr in feature_results.values():
        if fr.adequacy.is_adequate and np.isfinite(fr.summaries.anisotropy):
            anisotropy.append(fr.summaries.anisotropy)
            localization.append(fr.summaries.localization_entropy)
            signs.append(np.sign(fr.summaries.peak_extremal))

    anisotropy = np.array(anisotropy)
    localization = np.array(localization)
    signs = np.array(signs)

    if len(anisotropy) == 0:
        ax.text(0.5, 0.5, "No adequate features", ha="center", va="center")
        return ax

    if color_by_sign:
        colors = np.where(signs > 0, "firebrick", "royalblue")
        ax.scatter(anisotropy, localization, c=colors, alpha=0.6, **kwargs)
    else:
        ax.scatter(anisotropy, localization, alpha=0.6, **kwargs)

    ax.set_xlabel("Anisotropy ($A_g$)")
    ax.set_ylabel("Localization ($L_g$)")
    ax.set_title("Spatial Phenotype Landscape")

    if show_archetypes:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        ax.text(
            xlim[0] + 0.05 * (xlim[1] - xlim[0]),
            ylim[0] + 0.05 * (ylim[1] - ylim[0]),
            "Null",
            fontstyle="italic",
            alpha=0.5,
        )

        ax.text(
            xlim[1] - 0.2 * (xlim[1] - xlim[0]),
            ylim[0] + 0.05 * (ylim[1] - ylim[0]),
            "Rim/Core\n(Global)",
            ha="center",
            fontstyle="italic",
            alpha=0.5,
        )

        wedge_label = "Wedge\n(Localized)"
        if delta_deg is not None and delta_deg >= 90:
            wedge_label = "Wedge\n(Artifact?)"

        ax.text(
            xlim[1] - 0.2 * (xlim[1] - xlim[0]),
            ylim[1] - 0.15 * (ylim[1] - ylim[0]),
            wedge_label,
            ha="center",
            fontstyle="italic",
            alpha=0.5,
        )

    return ax


def plot_phenotype_map(
    feature_results: dict,
    ax: plt.Axes | None = None,
    y_axis: str = "polarity",
    color_by: str = "localization_entropy",
    show_archetypes: bool = True,
    delta_deg: float | None = None,
    **kwargs,
) -> plt.Axes:
    """
    Plot magnitude (A_g) vs directionality (polarity/R_mean) to distinguish core vs rim.

    Parameters
    ----------
    feature_results : dict
        Mapping of feature name to FeatureResult.
    ax : plt.Axes, optional
        Axes to plot on.
    y_axis : str, optional
        Field to use for y-axis ('polarity' or 'r_mean').
    color_by : str, optional
        Field to use for coloring points (e.g., 'localization_entropy').
    show_archetypes : bool, optional
        Whether to annotate archetypes (rim, core, wedge, null).
    delta_deg : float, optional
        The sector width used. If >= 90, "Wedge" labels are marked as potential artifacts.
    **kwargs
        Passed to ax.scatter.

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))

    anisotropy = []
    y_vals = []
    c_vals = []

    for fr in feature_results.values():
        if fr.adequacy.is_adequate and np.isfinite(fr.summaries.anisotropy):
            anisotropy.append(fr.summaries.anisotropy)
            y_vals.append(getattr(fr.summaries, y_axis))
            c_vals.append(getattr(fr.summaries, color_by))

    anisotropy = np.array(anisotropy)
    y_vals = np.array(y_vals)
    c_vals = np.array(c_vals)

    if len(anisotropy) == 0:
        ax.text(0.5, 0.5, "No adequate features", ha="center", va="center")
        return ax

    scatter = ax.scatter(anisotropy, y_vals, c=c_vals, cmap="viridis", alpha=0.7, **kwargs)
    plt.colorbar(scatter, ax=ax, label=color_by.replace("_", " ").title())

    ax.set_xlabel("Anisotropy ($A_g$)")
    ax.set_ylabel(y_axis.replace("_", " ").title())
    ax.set_title(f"Phenotype Map: Magnitude vs {y_axis.title()}")

    if show_archetypes:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        ax.text(
            xlim[1] - 0.05 * (xlim[1] - xlim[0]),
            ylim[1] - 0.1 * (ylim[1] - ylim[0]),
            "Core-enriched",
            ha="right",
            fontstyle="italic",
            alpha=0.6,
            color="firebrick",
        )

        ax.text(
            xlim[1] - 0.05 * (xlim[1] - xlim[0]),
            ylim[0] + 0.1 * (ylim[1] - ylim[0]),
            "Rim-enriched",
            ha="right",
            fontstyle="italic",
            alpha=0.6,
            color="royalblue",
        )

        wedge_label = "Mixed / Wedge"
        if delta_deg is not None and delta_deg >= 90:
            wedge_label = "Mixed / Wedge (Artifact?)"

        ax.text(
            xlim[1] - 0.05 * (xlim[1] - xlim[0]),
            0,
            wedge_label,
            ha="right",
            va="center",
            fontstyle="italic",
            alpha=0.6,
        )

    return ax


def plot_radar(
    radar: RadarResult,
    ax: plt.Axes | None = None,
    title: str | None = None,
    color: str = "b",
    alpha: float = 0.5,
    mode: str = "signed",
    radial_max: float | None = None,
    summaries: ScalarSummaries | None = None,
    show_anchors: bool = False,
    **kwargs,
) -> plt.Axes:
    """
    Plot RSP radar profile.

    Args:
        radar: RadarResult object.
        ax: Matplotlib axes (must be polar). If None, created.
        title: Plot title.
        color: Line/Fill color for proximal (positive) values.
        alpha: Fill transparency.
        mode: Plotting mode (semantic):
            - "signed": canonical signed view; radial length = |R|, color/style encodes sign (proximal vs distal).
            - "proximal": only proximal (R > 0) magnitudes.
            - "distal": only distal (R < 0) magnitudes.
        Note: legacy aliases supported: 'enrichment' -> 'proximal', 'depletion' -> 'distal',
        'combined'/'relative' -> 'signed'.
        radial_max: Optional override for the radial axis upper limit. If provided, used directly.
        summaries: Optional `ScalarSummaries` object to annotate peak angles and statistics.
        show_anchors: When True and `summaries` provided, plot peak markers and an annotation box.
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

    finite_r = np.isfinite(r_raw)
    if not np.any(finite_r):
        msg = "No valid sectors (insufficient foreground/background counts)"
        ax.set_title(f"{title}\n{msg}" if title else msg, fontsize=20)
        ax.set_ylim(0, 1)
        return ax

    max_mag = float(np.nanmax(np.abs(r_raw[finite_r])))
    y_top = float(radial_max) if radial_max is not None else max(1e-06, max_mag * 1.05)

    if y_top <= 0:
        y_top = 1e-6

    theta_norm = _maybe_degrees_to_radians(theta_raw.astype(float, copy=False))
    theta_norm = np.mod(theta_norm, 2 * np.pi)

    order = np.array([0], dtype=int) if theta_norm.size == 1 else np.argsort(theta_norm)

    theta = theta_norm[order]
    r = r_raw[order]
    counts_fg = radar.counts_fg[order] if hasattr(radar, "counts_fg") else None

    theta_sorted, widths = _prepare_polar(theta)
    theta = theta_sorted

    valid = np.isfinite(r)
    theta_v = theta[valid]

    if theta_v.size == 0:
        msg = "No valid sectors (insufficient foreground/background counts)"
        ax.set_title(f"{title}\n{msg}" if title else msg, fontsize=20)
        ax.set_ylim(0, 1)
        return ax

    lw = float(kwargs.pop("linewidth", 1.25))

    # Mark underpowered sectors with faint ticks.
    invalid_theta = theta[~valid]
    if invalid_theta.size:
        for t in invalid_theta:
            ax.plot([t, t], [y_top * 0.95, y_top], color="gray", linewidth=1.0, alpha=0.6)

    # Mark zero-filled sectors (empty foreground) with distinct marker
    if counts_fg is not None:
        # Zero-filled sectors are valid (finite r), have r=0, and counts_fg=0
        zero_filled_mask = valid & (r == 0) & (counts_fg == 0)
        zero_filled_theta = theta[zero_filled_mask]
        if zero_filled_theta.size:
            # Plot a small circle at the rim to indicate "forced zero"
            ax.scatter(
                zero_filled_theta,
                np.full_like(zero_filled_theta, y_top),
                marker="o",
                color="gray",
                s=15,
                alpha=0.5,
                zorder=10,
            )

    def _maybe_add_anchors(ax_obj: plt.Axes, summaries_obj: ScalarSummaries | None):
        if not summaries_obj or not show_anchors:
            return
        p_prox = getattr(summaries_obj, "peak_proximal_angle", None)
        p_dist = getattr(summaries_obj, "peak_distal_angle", None)
        if p_prox is not None:
            ax_obj.plot(p_prox, y_top * 0.9, marker="^", color=color, markersize=8)
        if p_dist is not None:
            ax_obj.plot(p_dist, y_top * 0.9, marker="v", color="r", markersize=8)
        lines = []
        anis = getattr(summaries_obj, "anisotropy", None)
        integ = getattr(summaries_obj, "integrated_rsp", None)
        if anis is not None:
            lines.append(f"Anisotropy: {anis:.3f}")
        if integ is not None:
            lines.append(f"Integrated RSP: {integ:.3f}")
        if lines:
            ax_obj.text(
                0.02,
                0.95,
                "\n".join(lines),
                transform=ax_obj.transAxes,
                va="top",
                fontsize=12,
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
            )

    mode = mode.lower().strip()
    mode_map = {
        "enrichment": "proximal",
        "depletion": "distal",
        "combined": "signed",
        "relative": "signed",
    }
    mode = mode_map.get(mode, mode)
    if mode not in {"signed", "proximal", "distal"}:
        raise ValueError(f"Unknown mode: {mode}")

    if mode == "signed":
        r_pos = np.where(r >= 0, np.abs(r), np.nan)
        r_neg = np.where(r < 0, np.abs(r), np.nan)

        if np.any(np.isfinite(r_pos)):
            _draw_segmented_rsp(
                ax,
                theta,
                r_pos,
                fill=True,
                color=color,
                alpha=alpha,
                line_color="k",
                linewidth=lw,
            )

        if np.any(np.isfinite(r_neg)):
            _draw_segmented_rsp(
                ax,
                theta,
                r_neg,
                fill=True,
                color="r",
                alpha=alpha * 0.9,
                hatch="//",
                linewidth=lw,
                linestyle="--",
            )

        ax.set_ylim(0, y_top)
        if title:
            ax.set_title(title + " RSP (signed)" if "RSP" not in title else title, fontsize=20)

        try:
            import matplotlib.patches as mpatches

            handles, labels = [], []
            if np.any(np.isfinite(r_pos)):
                handles.append(mpatches.Patch(color=color, alpha=alpha))
                labels.append("Proximal bias (R > 0)")
            if np.any(np.isfinite(r_neg)):
                p = mpatches.Patch(facecolor="r", hatch="//", edgecolor="r", alpha=alpha)
                handles.append(p)
                labels.append("Distal bias (R < 0)")
            if handles:
                ax.legend(handles, labels, loc="lower left", fontsize=18)
        except Exception:
            pass

        _maybe_add_anchors(ax, summaries)
        return ax

    if mode == "proximal":
        r_pos = np.where(r > 0, r, np.nan)
        if np.any(np.isfinite(r_pos)):
            _draw_segmented_rsp(ax, theta, r_pos, fill=True, color=color, alpha=alpha, linewidth=lw)
        ax.set_ylim(0, y_top)
        if title:
            ax.set_title(title + " RSP (proximal)", fontsize=20)
        _maybe_add_anchors(ax, summaries)
        return ax

    if mode == "distal":
        r_neg = np.where(r < 0, np.abs(r), np.nan)
        if np.any(np.isfinite(r_neg)):
            _draw_segmented_rsp(
                ax,
                theta,
                r_neg,
                fill=True,
                color="r",
                alpha=alpha * 0.9,
                hatch="//",
                linewidth=lw,
                linestyle="--",
            )
        ax.set_ylim(0, y_top)
        if title:
            ax.set_title(title + " RSP (distal)", fontsize=20)
        _maybe_add_anchors(ax, summaries)
        return ax

    if mode == "combined":
        r_pos = np.where(r > 0, r, np.nan)
        r_neg = np.where(r < 0, np.abs(r), np.nan)

        if np.any(np.isfinite(r_pos)):
            _draw_segmented_rsp(ax, theta, r_pos, fill=True, color=color, alpha=alpha, linewidth=lw)

        if np.any(np.isfinite(r_neg)):
            _draw_segmented_rsp(
                ax,
                theta,
                r_neg,
                fill=True,
                color="r",
                alpha=alpha,
                linewidth=lw,
                linestyle="--",
            )

        ax.set_ylim(0, y_top)
        if title:
            ax.set_title(title + " RSP (combined)", fontsize=20)

        try:
            import matplotlib.patches as mpatches

            handles, labels = [], []
            if np.any(np.isfinite(r_pos)):
                handles.append(mpatches.Patch(color=color, alpha=alpha))
                labels.append("Enrichment")
            if np.any(np.isfinite(r_neg)):
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
    fig: plt.Figure | None = None,
    title: str | None = None,
    color: str = "b",
    alpha: float = 0.5,
    radial_max: float | None = None,
    summaries: ScalarSummaries | None = None,
    show_anchors: bool = False,
    **kwargs,
) -> plt.Figure:
    """
    Generate both proximal and distal radar plots side-by-side with shared radial scale.

    Args:
        radar: RadarResult object.
        fig: Matplotlib figure. If None, created.
        title: Overall title.
        color: Line/Fill color for proximal.
        alpha: Fill transparency.
        radial_max: Optional override for shared radial axis upper limit.
        summaries: Optional `ScalarSummaries` object to annotate peaks and stats.
        show_anchors: Whether to draw anchors/annotations when `summaries` is provided.
        **kwargs: Additional arguments for plot_radar.

    Returns:
        fig: The figure object.
    """
    if fig is None:
        fig = plt.figure(figsize=(10, 5))

    finite = np.isfinite(radar.rsp)
    if radial_max is None:
        if np.any(finite):
            radial_max_use = float(np.nanmax(np.abs(radar.rsp[finite]))) * 1.05
        else:
            radial_max_use = 1.0
    else:
        radial_max_use = float(radial_max)

    ax1 = fig.add_subplot(121, projection="polar")
    plot_radar(
        radar,
        ax=ax1,
        title="Proximal RSP ($R > 0$)",
        color=color,
        alpha=alpha,
        mode="proximal",
        radial_max=radial_max_use,
        summaries=summaries,
        show_anchors=show_anchors,
        **kwargs,
    )

    ax2 = fig.add_subplot(122, projection="polar")
    plot_radar(
        radar,
        ax=ax2,
        title="Distal RSP ($R < 0$)",
        color="r",
        alpha=alpha,
        mode="distal",
        radial_max=radial_max_use,
        summaries=summaries,
        show_anchors=show_anchors,
        **kwargs,
    )

    if title:
        fig.suptitle(title + " RSP" if "RSP" not in title else title, fontsize=20)

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


# Alias with clearer name
def plot_radar_split(*args, **kwargs):
    """Alias for `plot_radar_absolute` to emphasize split/proximal-vs-distal layout."""
    return plot_radar_absolute(*args, **kwargs)


def plot_summary(summary: ScalarSummaries, ax: plt.Axes | None = None) -> plt.Axes:
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
        f"Coverage (BG/FG): {summary.coverage_bg:.2f} / {summary.coverage_fg:.2f}\n"
        f"Peak Distal Angle: {np.degrees(summary.peak_distal_angle):.1f}°\n"
        f"Peak Proximal Angle: {np.degrees(summary.peak_proximal_angle):.1f}°\n"
        f"Peak Extremal Angle: {np.degrees(summary.peak_extremal_angle):.1f}°"
    )

    ax.text(0.1, 0.5, text, fontsize=20, transform=ax.transAxes, va="center")
    return ax


__all__ = ["plot_radar", "plot_radar_absolute", "plot_summary"]
