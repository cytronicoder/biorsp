"""Plotting module for BioRSP.

Provides visualization functions:
- Radar plots for RSP profiles
- Embedding scatter plots
- Summary visualizations

Theta Convention
----------------
BioRSP computation uses mathematical convention from atan2:
- 0 radians at +x axis (east)
- Increases counterclockwise
- Range: [-π, π) or [0, 2π)

Plotting supports two conventions:
- "math" (default): matches computation, 0 at east, CCW
- "compass": 0 at north, clockwise (visual transform only)
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from biorsp.core.summaries import ScalarSummaries
from biorsp.core.typing import RadarResult
from biorsp.utils.helpers import _as_1d


def transform_theta(
    theta: np.ndarray, from_convention: str = "math", to_convention: str = "math"
) -> np.ndarray:
    """Transform angles between theta conventions.

    Args:
        theta: Array of angles in radians.
        from_convention: Source convention ("math" or "compass").
        to_convention: Target convention ("math" or "compass").

    Returns:
        Transformed angles in radians, normalized to [0, 2π).

    Conventions:
        - "math": 0 at +x (east), increases counterclockwise (atan2 convention)
        - "compass": 0 at +y (north), increases clockwise

    Transform: compass = (π/2 - math) mod 2π
               math = (π/2 - compass) mod 2π
    """
    if from_convention == to_convention:
        return np.mod(theta, 2 * np.pi)

    if (
        from_convention == "math"
        and to_convention == "compass"
        or from_convention == "compass"
        and to_convention == "math"
    ):
        return np.mod(np.pi / 2 - theta, 2 * np.pi)
    else:
        raise ValueError(
            f"Unsupported convention pair: from={from_convention}, to={to_convention}. "
            f"Supported: 'math', 'compass'."
        )


def _maybe_degrees_to_radians(theta: np.ndarray) -> np.ndarray:
    """Heuristic: if angles look like degrees, convert to radians."""
    finite = np.isfinite(theta)
    if not np.any(finite):
        return theta
    tmax = float(np.nanmax(theta[finite]))

    if tmax > (2.0 * np.pi * 1.2):
        return np.deg2rad(theta)
    return theta


def _split_into_finite_segments(theta: np.ndarray, vals: np.ndarray) -> list[list[int]]:
    """Split sorted arrays into contiguous segments of finite values.

    Args:
        theta: Sorted angles in radians.
        vals: Corresponding values.

    Returns:
        List of index lists, each representing a contiguous finite segment.
    """
    if theta.size == 0:
        return []

    mask = np.isfinite(vals)
    if not np.any(mask):
        return []

    idx = np.where(mask)[0]
    if len(idx) == 0:
        return []

    splits = np.where(np.diff(idx) > 1)[0] + 1
    seg_indices = np.split(idx, splits)
    return [s.tolist() for s in seg_indices]


def _merge_wraparound_segments(segments: list[list[int]], vals: np.ndarray) -> list[list[int]]:
    """Merge first and last segments if they represent circular continuity.

    Args:
        segments: List of index lists from _split_into_finite_segments.
        vals: Original value array to check finite status.

    Returns:
        Merged segment list with wraparound handled.
    """
    if len(segments) == 0:
        return segments

    n = vals.size

    if len(segments) == 1:
        first_seg = segments[0]
        if len(first_seg) == n and np.all(np.isfinite(vals)):
            first_seg.append(first_seg[0])
        return segments

    if len(segments) > 1:
        first_indices = segments[0]
        last_indices = segments[-1]

        if (
            len(first_indices) > 0
            and len(last_indices) > 0
            and first_indices[0] == 0
            and last_indices[-1] == n - 1
            and np.isfinite(vals[0])
            and np.isfinite(vals[n - 1])
        ):
            merged = last_indices + first_indices
            return [merged] + segments[1:-1]

    return segments


def _prepare_polar(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalize angles to [0, 2pi), sort, and compute sector widths (one per theta).

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

    if getattr(ax, "name", "") != "polar":
        raise ValueError("plot_radar requires a polar axis (projection='polar').")
    return ax


def _set_default_polar_style(ax: plt.Axes, theta_convention: str = "math") -> None:
    """Configure polar axis style based on theta convention.

    Args:
        ax: Polar axes object.
        theta_convention: "math" (default, 0 at east, CCW) or "compass" (0 at north, CW).
    """
    if theta_convention == "math":
        pass
    elif theta_convention == "compass":
        ax.set_theta_direction(-1)
        ax.set_theta_zero_location("N")
    else:
        raise ValueError(f"Unknown theta_convention: {theta_convention}")


def _draw_debug_overlays(
    ax: plt.Axes,
    theta: np.ndarray,
    radar: RadarResult,
    theta_convention: str,
    y_top: float,
) -> None:
    """Add diagnostic overlays showing sector validity and counts.

    Displays:
    - nF (foreground count) per sector as text
    - nB (background count) per sector as text
    - Whether sector was forced-zero (empty FG)
    - Whether sector is geom-supported (valid)
    """
    if radar.counts_fg is None or radar.counts_bg is None:
        return

    theta_centers_math = _as_1d(radar.centers)
    theta_display = transform_theta(
        theta_centers_math, from_convention="math", to_convention=theta_convention
    )

    for i, (th, nf, nb) in enumerate(zip(theta_display, radar.counts_fg, radar.counts_bg)):
        is_valid = True
        if radar.geom_supported_mask is not None:
            is_valid = radar.geom_supported_mask[i]

        is_zero_filled = False
        if radar.forced_zero_mask is not None:
            is_zero_filled = radar.forced_zero_mask[i]

        if is_zero_filled:
            color_text = "orange"
            marker = "◉"
        elif not is_valid:
            color_text = "gray"
            marker = "○"
        else:
            color_text = "black"
            marker = "●"

        r_text = y_top * 0.85
        ax.text(
            th,
            r_text,
            f"{marker}\nnF={int(nf)}\nnB={int(nb)}",
            ha="center",
            va="center",
            fontsize=7,
            color=color_text,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1),
        )


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
    """Draw RSP outline and fill, handling NaN gaps and circular wrap-around.

    Splits continuous segments at NaN boundaries. Merges segments across
    the 0/2pi boundary if both ends are finite (polar wrap-around).
    """
    if th.size == 0:
        return

    segments = _split_into_finite_segments(th, vals)
    if not segments:
        return

    segments = _merge_wraparound_segments(segments, vals)

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
    """Plot anisotropy (A_g) vs localization (L_g) to distinguish spatial phenotypes.

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
    """Plot magnitude (A_g) vs directionality (polarity/R_mean) to distinguish core vs rim.

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
    radial_max: float | str | None = None,
    summaries: ScalarSummaries | None = None,
    show_anchors: bool = False,
    theta_convention: str = "math",
    debug_overlay: bool = False,
    **kwargs,
) -> plt.Axes:
    """Plot radar profile with proper handling of supported sectors.

    This function ensures:
    - Geometry-supported sectors (finite RSP) are plotted
    - Invalid sectors (NaN RSP from insufficient background) shown as gaps
    - Zero-filled sectors (empty FG, forced to 0) marked distinctly
    - No smoothing applied to metrics (display-only smoothing would be explicit)
    - Correct theta convention handling

    Args:
        radar: RadarResult object with centers in math convention (0 at +x, CCW).
               Must contain rsp, centers, and optionally geom_supported_mask, forced_zero_mask.
        ax: Matplotlib axes (must be polar). If None, created.
        title: Plot title.
        color: Line/Fill color for proximal (positive) values (default: "b").
        alpha: Fill transparency (default: 0.5).
        mode: Plotting mode (semantic):
            - "signed" (default): canonical signed view; |R| as radius, sign determines color/style
            - "proximal": only positive (R > 0) magnitudes
            - "distal": only negative (R < 0) magnitudes
        radial_max: Radial axis upper limit control:
            - None (default): robust 99th percentile * 1.1 (avoids outlier flattening)
            - "max": use absolute maximum (may be affected by outliers)
            - float: explicit value
        summaries: Optional `ScalarSummaries` object to annotate peak angles and statistics.
        show_anchors: When True and `summaries` provided, plot peak markers and annotation box.
        theta_convention: Axis convention for visualization:
            - "math" (default): 0 at +x (east), increases CCW (matches computation)
            - "compass": 0 at +y (north), increases CW (visual transform only)
        debug_overlay: If True, add diagnostic overlays showing sector validity and counts.
        **kwargs: Additional arguments forwarded to matplotlib (e.g., linewidth).

    Returns:
        ax: The axes object.

    Note:
        BioRSP computes angles using mathematical convention (atan2: 0 at +x, CCW).
        The theta_convention parameter only affects visualization. Data integrity
        is preserved regardless of display convention.

    Example:
        >>> radar = compute_rsp_radar(r_norm, theta, fg_mask, config=config)
        >>> plot_radar(radar, theta_convention="math", radial_max=None, debug_overlay=False)
    """
    ax = _ensure_polar_ax(ax)
    _set_default_polar_style(ax, theta_convention)

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

    if radial_max is None:
        max_mag = float(np.nanpercentile(np.abs(r_raw[finite_r]), 99))
        y_top = max(1e-06, max_mag * 1.1)
    elif radial_max == "max":
        max_mag = float(np.nanmax(np.abs(r_raw[finite_r])))
        y_top = max(1e-06, max_mag * 1.05)
    else:
        y_top = float(radial_max)

    if y_top <= 0:
        y_top = 1e-6

    theta_norm = _maybe_degrees_to_radians(theta_raw.astype(float, copy=False))
    theta_norm = np.mod(theta_norm, 2 * np.pi)

    theta_plot = transform_theta(theta_norm, from_convention="math", to_convention=theta_convention)

    order = np.array([0], dtype=int) if theta_plot.size == 1 else np.argsort(theta_plot)

    theta = theta_plot[order]
    r = r_raw[order]

    theta_sorted, _ = _prepare_polar(theta)
    theta = theta_sorted

    valid = np.isfinite(r)
    theta_v = theta[valid]

    if theta_v.size == 0:
        msg = "No valid sectors (insufficient foreground/background counts)"
        ax.set_title(f"{title}\n{msg}" if title else msg, fontsize=20)
        ax.set_ylim(0, 1)
        return ax

    lw = float(kwargs.pop("linewidth", 1.25))

    invalid_theta = theta[~valid]
    if invalid_theta.size:
        for t in invalid_theta:
            ax.plot([t, t], [y_top * 0.95, y_top], color="gray", linewidth=1.0, alpha=0.6)

    if radar.forced_zero_mask is not None:
        zero_filled_ordered = radar.forced_zero_mask[order]
        zero_filled_theta = theta[zero_filled_ordered]
        if zero_filled_theta.size:
            ax.scatter(
                zero_filled_theta,
                np.full_like(zero_filled_theta, y_top),
                marker="o",
                color="gray",
                s=15,
                alpha=0.5,
                zorder=10,
            )

    def _add_anchors_if_requested(ax_obj: plt.Axes, summaries_obj: ScalarSummaries | None):
        if not summaries_obj or not show_anchors:
            return
        p_prox = getattr(summaries_obj, "peak_proximal_angle", None)
        p_dist = getattr(summaries_obj, "peak_distal_angle", None)

        if p_prox is not None:
            p_prox_plot = transform_theta(
                np.array([p_prox]), from_convention="math", to_convention=theta_convention
            )[0]
            ax_obj.plot(p_prox_plot, y_top * 0.9, marker="^", color=color, markersize=8, zorder=15)

        if p_dist is not None:
            p_dist_plot = transform_theta(
                np.array([p_dist]), from_convention="math", to_convention=theta_convention
            )[0]
            ax_obj.plot(p_dist_plot, y_top * 0.9, marker="v", color="r", markersize=8, zorder=15)

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

        if debug_overlay:
            _draw_debug_overlays(ax, theta, radar, theta_convention, y_top)

        if title:
            ax.set_title(title + " RSP (signed)" if "RSP" not in title else title, fontsize=20)

        try:
            import matplotlib.patches as mpatches

            handles, labels = [], []
            if np.any(np.isfinite(r_pos)):
                handles.append(mpatches.Patch(color=color, alpha=alpha))
                labels.append("Proximal shift (core-biased)")
            if np.any(np.isfinite(r_neg)):
                p = mpatches.Patch(facecolor="r", hatch="//", edgecolor="r", alpha=alpha)
                handles.append(p)
                labels.append("Distal shift (rim-biased)")
            if handles:
                ax.legend(handles, labels, loc="lower left", fontsize=18)
        except Exception:
            pass

        _add_anchors_if_requested(ax, summaries)
        return ax

    if mode == "proximal":
        r_pos = np.where(r > 0, r, np.nan)
        if np.any(np.isfinite(r_pos)):
            _draw_segmented_rsp(ax, theta, r_pos, fill=True, color=color, alpha=alpha, linewidth=lw)
        ax.set_ylim(0, y_top)
        if title:
            ax.set_title(title + " RSP (proximal)", fontsize=20)
        _add_anchors_if_requested(ax, summaries)
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
        _add_anchors_if_requested(ax, summaries)
        return ax

    raise ValueError(f"Unknown mode: {mode}")


def plot_radar_absolute(
    radar: RadarResult,
    fig: plt.Figure | None = None,
    title: str | None = None,
    color: str = "b",
    alpha: float = 0.5,
    radial_max: float | str | None = None,
    summaries: ScalarSummaries | None = None,
    show_anchors: bool = False,
    theta_convention: str = "math",
    **kwargs,
) -> plt.Figure:
    """Generate both proximal and distal radar plots side-by-side with shared radial scale.

    Args:
        radar: RadarResult object.
        fig: Matplotlib figure. If None, created.
        title: Overall title.
        color: Line/Fill color for proximal.
        alpha: Fill transparency.
        radial_max: Radial limit control (None=robust, "max"=absolute, float=explicit).
        summaries: Optional `ScalarSummaries` object to annotate peaks and stats.
        show_anchors: Whether to draw anchors/annotations when `summaries` is provided.
        theta_convention: Axis convention ("math" or "compass").
        **kwargs: Additional arguments for plot_radar.

    Returns:
        fig: The figure object.

    """
    if fig is None:
        fig = plt.figure(figsize=(10, 5))

    finite = np.isfinite(radar.rsp)
    if radial_max is None:
        if np.any(finite):
            radial_max_use = float(np.nanpercentile(np.abs(radar.rsp[finite]), 99)) * 1.1
        else:
            radial_max_use = 1.0
    elif radial_max == "max":
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
        theta_convention=theta_convention,
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
        theta_convention=theta_convention,
        **kwargs,
    )

    if title:
        fig.suptitle(title + " RSP" if "RSP" not in title else title, fontsize=20)

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


def plot_radar_split(*args, **kwargs):
    """Alias for `plot_radar_absolute` to emphasize split/proximal-vs-distal layout."""
    return plot_radar_absolute(*args, **kwargs)


def plot_summary(summary: ScalarSummaries, ax: plt.Axes | None = None) -> plt.Axes:
    """Display scalar summaries as text.

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
        f"Coverage (geom/FG): {summary.coverage_geom:.2f} / {summary.coverage_fg:.2f}\n"
        f"Peak Distal Angle: {np.degrees(summary.peak_distal_angle):.1f}°\n"
        f"Peak Proximal Angle: {np.degrees(summary.peak_proximal_angle):.1f}°\n"
        f"Peak Extremal Angle: {np.degrees(summary.peak_extremal_angle):.1f}°"
    )

    ax.text(0.1, 0.5, text, fontsize=20, transform=ax.transAxes, va="center")
    return ax


__all__ = ["plot_radar", "plot_radar_absolute", "plot_summary"]
