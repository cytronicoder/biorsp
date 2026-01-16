from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from biorsp.core.engine import compute_rsp_radar
from biorsp.core.geometry import get_sector_indices, polar_coordinates
from biorsp.plotting.style import (
    COLORS,
    add_panel_label,
    get_column_width,
    save_figure,
)
from biorsp.preprocess.normalization import normalize_radii
from biorsp.utils.config import BioRSPConfig


def interpret_pattern(
    S_g: float,
    R_mean: float,
    coverage_geom: float,
    delta_deg: float,
) -> str:
    """Interpret spatial pattern using angular resolution-dependent rules.

    Args:
        S_g: Spatial organization score (RMS of radar profile).
        R_mean: Mean signed radar profile (positive = core bias, negative = rim bias).
        coverage_geom: Fraction of geometry-supported sectors.
        delta_deg: Angular resolution in degrees.

    Returns:
        Interpretation string with conservative labels appropriate for Δ.

    Note:
        Δ ≥ 90°: Cannot make wedge/directional claims (half-plane sectors).
        Use only global core/rim bias or diffuse/weak labels.
    """
    if coverage_geom < 0.5:
        return "Low angular coverage (unreliable)"

    if S_g < 0.1:
        return "Diffuse / Weak signal"

    if delta_deg >= 90:
        if abs(R_mean) < 0.15:
            return "Global mixed signal"
        elif R_mean > 0:
            return "Global core bias"
        else:
            return "Global rim bias"
    elif delta_deg >= 60:
        if abs(R_mean) < 0.15:
            return "Mixed / Sector-level variation"
        elif R_mean > 0:
            return "Sector localization (core-biased)"
        else:
            return "Sector localization (rim-biased)"
    else:
        if abs(R_mean) < 0.15:
            return "Localized mixed pattern"
        elif R_mean > 0:
            return "Wedge localization (core direction)"
        else:
            return "Wedge localization (rim direction)"


def make_end_to_end_figure(
    z: np.ndarray,
    y: np.ndarray,
    v: np.ndarray,
    theta_grid: np.ndarray,
    delta_deg: float,
    outpath: str,
    feature_name: str = "Feature",
    seed: int = 42,
    coverage: Optional[float] = None,
    expr_threshold: Optional[float] = None,
    foreground_fraction: Optional[float] = None,
    debug: bool = False,
):
    """Generate end-to-end workflow figure illustrating BioRSP scoring.

    This figure illustrates the BioRSP two-score framework:
    - Coverage score C_g: fraction of cells expressing >= biological threshold
    - Spatial organization score S_g: RMS magnitude of radar profile

    Key Design:
    - coverage (C_g) is distinct from foreground_fraction (internal quantile)
    - Radii are normalized before radar computation
    - Interpretation uses Δ-dependent rules (no wedge claims when Δ ≥ 90°)

    Parameters
    ----------
    z : np.ndarray
        (N, 2) coordinates.
    y : np.ndarray
        (N,) binary foreground labels (1=fg, 0=bg). This is the INTERNAL
        foreground for spatial scoring (from quantile), NOT coverage threshold.
    v : np.ndarray
        (2,) vantage point.
    theta_grid : np.ndarray
        (B,) angular grid centers in radians.
    delta_deg : float
        Sector width in degrees.
    outpath : str
        Path to save the figure.
    feature_name : str
        Feature name for title.
    seed : int
        Random seed (for reproducibility).
    coverage : float, optional
        Coverage score C_g (fraction >= expr_threshold). If provided, displayed.
    expr_threshold : float, optional
        Expression threshold used for coverage. Shown if provided.
    foreground_fraction : float, optional
        Internal foreground fraction (quantile). Metadata only, not a biological score.
    debug : bool
        If True, add debug panels showing:
        - Embedding + vantage
        - FG/BG overlays for both thresholds
        - nF/nB per θ
        - bg-supported mask per θ
    """
    if delta_deg > 180:
        raise ValueError(f"Delta {delta_deg} > 180 degrees is forbidden (sector > hemisphere).")

    if delta_deg >= 90:
        print(
            f"WARNING: Δ={delta_deg}° ≥ 90° → half-plane sectors. "
            "Cannot make directional wedge claims. Use global core/rim bias only."
        )

    r, theta = polar_coordinates(z, v)
    r_norm, norm_stats = normalize_radii(r)

    config = BioRSPConfig(delta_deg=delta_deg, B=len(theta_grid))
    res = compute_rsp_radar(r_norm, theta, y, config=config)

    valid_mask = ~np.isnan(res.rsp)
    if not np.any(valid_mask):
        print("No valid sectors. Cannot generate figure.")
        return

    mask_geom = res.geom_supported_mask if res.geom_supported_mask is not None else valid_mask
    weights = res.sector_weights if res.sector_weights is not None else np.ones(len(res.rsp))

    w_valid = weights[mask_geom]
    sum_w = np.sum(w_valid)
    if sum_w > 0:
        w_norm = w_valid / sum_w
        rsp_valid = res.rsp[mask_geom]
        rsp_valid = np.nan_to_num(rsp_valid, nan=0.0)
        S_g = float(np.sqrt(np.sum(w_norm * rsp_valid**2)))
        R_mean = float(np.sum(w_norm * rsp_valid))
    else:
        S_g = 0.0
        R_mean = 0.0

    coverage_geom = float(np.mean(mask_geom))

    peak_idx = np.nanargmax(np.abs(res.rsp))
    theta_star = res.centers[peak_idx]
    R_star = res.rsp[peak_idx]

    fig = plt.figure(figsize=(get_column_width("double"), 7), constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0], projection="polar")
    ax_d = fig.add_subplot(gs[1, 1])

    ax_a.scatter(
        z[y == 0, 0], z[y == 0, 1], c=COLORS["bg_cells"], s=1, alpha=0.2, label="Background"
    )
    ax_a.scatter(
        z[y == 1, 0],
        z[y == 1, 1],
        c=COLORS["fg_cells"],
        s=2,
        alpha=0.5,
        label=f"{feature_name} (internal FG)",
    )
    ax_a.scatter(v[0], v[1], c="black", marker="x", s=50, label="Vantage")

    max_r = np.max(r)
    for phi in res.centers:
        ax_a.plot(
            [v[0], v[0] + 1.2 * max_r * np.cos(phi)],
            [v[1], v[1] + 1.2 * max_r * np.sin(phi)],
            color=COLORS["highlight"],
            alpha=0.05,
            lw=0.5,
        )

    ray_x = v[0] + 1.2 * max_r * np.cos(theta_star)
    ray_y = v[1] + 1.2 * max_r * np.sin(theta_star)
    ax_a.plot(
        [v[0], ray_x], [v[1], ray_y], color=COLORS["highlight"], lw=1.5, label=r"Peak $\theta^*$"
    )

    delta_rad = np.deg2rad(delta_deg)
    wedge_theta = np.linspace(theta_star - delta_rad / 2, theta_star + delta_rad / 2, 20)
    wedge_x = np.concatenate([[v[0]], v[0] + 1.2 * max_r * np.cos(wedge_theta), [v[0]]])
    wedge_y = np.concatenate([[v[1]], v[1] + 1.2 * max_r * np.sin(wedge_theta), [v[1]]])
    ax_a.fill(wedge_x, wedge_y, color=COLORS["highlight"], alpha=0.1)

    ax_a.set_aspect("equal")
    ax_a.set_title(f"{feature_name}: Spatial Distribution")
    ax_a.set_xlabel("Embedding Dimension 1")
    ax_a.set_ylabel("Embedding Dimension 2")
    ax_a.legend(loc="upper right", fontsize=8)
    add_panel_label(ax_a, "A")

    sector_indices_list = get_sector_indices(theta, len(theta_grid), delta_deg)
    indices_star = sector_indices_list[peak_idx]

    r_star_sector = r_norm[indices_star]
    y_star_sector = y[indices_star]

    r_fg = r_star_sector[y_star_sector == 1]
    r_bg = r_star_sector[y_star_sector == 0]

    n_fg = len(r_fg)
    n_bg = len(r_bg)

    if n_fg > 0 and n_bg > 0:
        r_fg_sorted = np.sort(r_fg)
        r_bg_sorted = np.sort(r_bg)
        y_fg = np.linspace(0, 1, n_fg)
        y_bg = np.linspace(0, 1, n_bg)

        ax_b.step(r_bg_sorted, y_bg, color=COLORS["ref_line"], label=r"$P_{\mathrm{bg}}$")
        ax_b.step(r_fg_sorted, y_fg, color=COLORS["fg_cells"], label=r"$P_{\mathrm{fg}}$")

        med_fg = np.median(r_fg)
        med_bg = np.median(r_bg)
        ax_b.axvline(med_bg, color=COLORS["ref_line"], ls="--", lw=1)
        ax_b.axvline(med_fg, color=COLORS["fg_cells"], ls="--", lw=1)

        diff = med_bg - med_fg
        sign_val = np.sign(diff)
        sign_str = "Core (+)" if sign_val > 0 else "Rim (-)"
        if sign_val == 0:
            sign_str = "Neutral (0)"

        ax_b.text(
            0.5,
            0.1,
            f"Sign: {sign_str}\n$R(\\theta^*) = {R_star:.2f}$",
            transform=ax_b.transAxes,
            bbox=dict(
                facecolor="white", edgecolor=COLORS["ref_line"], alpha=0.8, boxstyle="round,pad=0.3"
            ),
            fontsize=8,
        )
    else:
        ax_b.text(0.5, 0.5, "Insufficient points in sector", ha="center")

    ax_b.set_title(r"Radial Distribution at Peak Direction $\theta^*$ (normalized)")
    ax_b.set_xlabel(r"Normalized Radius $\hat{r}$")
    ax_b.set_ylabel("Cumulative Probability")
    ax_b.legend(loc="lower right", fontsize=8)
    add_panel_label(ax_b, "B")

    from biorsp.plotting.radar import plot_radar

    plot_radar(
        res,
        ax=ax_c,
        title=r"Directional Organization Profile $R(\theta)$",
        mode="signed",
        theta_convention="math",
        color="black",
        alpha=0.1,
        linewidth=1.2,
        debug_overlay=debug,
    )
    add_panel_label(ax_c, "C")

    interpretation = interpret_pattern(S_g, R_mean, coverage_geom, delta_deg)

    vantage_str = config.vantage.replace("_", " ").title()

    param_text = (
        r"$\bf{BioRSP\ Parameters}$" + "\n\n"
        f"$B = {config.B}$ | "
        rf"$\Delta = {config.delta_deg:g}^\circ$ | "
        f"Feature: {feature_name}" + "\n\n"
        f"Vantage: {vantage_str}\n"
        f"Normalization: {norm_stats.get('method', 'median-IQR')}"
    )

    stats_text = r"$\bf{Two\text{-}Score\ Framework}$" + "\n\n"

    if coverage is not None:
        stats_text += f"Coverage $C_g$ = {coverage:.2%}"
        if expr_threshold is not None:
            stats_text += f" (≥ {expr_threshold:.2g})"
        stats_text += "\n"

    stats_text += f"Spatial Org. $S_g$ = {S_g:.3f}\n\n"

    stats_text += (
        r"$\bf{Diagnostic\ Metrics}$" + "\n"
        f"$\\bar{{R}}_{{geom}}$ = {R_mean:.3f} | "
        f"$C_{{geom}}$ = {coverage_geom:.2f}\n"
    )

    if foreground_fraction is not None:
        stats_text += f"Internal FG frac = {foreground_fraction:.2%} (not C_g)\n"

    stats_text += f"\nPattern: {interpretation}"

    bbox_props = dict(
        boxstyle="round,pad=0.8", facecolor="#F8F9FA", edgecolor=COLORS["ref_line"], alpha=0.5
    )

    ax_d.axis("off")
    ax_d.text(
        0.5,
        0.7,
        param_text,
        va="center",
        ha="center",
        fontsize=8.5,
        transform=ax_d.transAxes,
        bbox=bbox_props,
    )

    ax_d.text(
        0.5,
        0.3,
        stats_text,
        va="center",
        ha="center",
        fontsize=8.5,
        transform=ax_d.transAxes,
        bbox=bbox_props,
    )
    ax_d.set_title("BioRSP Summary: Coverage & Spatial Organization")
    add_panel_label(ax_d, "D")

    outpath_obj = Path(outpath)
    if outpath_obj.suffix:
        outdir = str(outpath_obj.parent) if outpath_obj.parent != Path(".") else "figures"
        filename = outpath_obj.stem
    else:
        outdir = outpath
        filename = f"end_to_end_{feature_name}"

    save_figure(fig, Path(outdir) / filename)

    plt.close()
