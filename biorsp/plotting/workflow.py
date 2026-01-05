import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from biorsp.core.engine import compute_rsp_radar
from biorsp.core.summaries import compute_scalar_summaries
from biorsp.plotting.style import (
    COLORS,
    add_panel_label,
    get_column_width,
    save_figure,
    set_publication_style,
)
from biorsp.preprocess.geometry import get_sector_indices, polar_coordinates
from biorsp.utils.config import BioRSPConfig


def interpret_pattern(A: float, L: float, R_mean: float, polarity: float) -> str:
    """
    Deterministic interpretation of the pattern.
    """
    if L < 0.1:
        if abs(R_mean) < 0.2:
            return "Diffuse / Uniform"
        elif R_mean > 0:
            return "Global Core Bias"
        else:
            return "Global Rim Bias"
    else:
        base = "Localized"
        if polarity > 0.8:
            if R_mean > 0:
                return f"{base} Core"
            else:
                return f"{base} Rim"
        else:
            return f"{base} Mixed"


def make_end_to_end_figure(
    z: np.ndarray,
    y: np.ndarray,
    v: np.ndarray,
    theta_grid: np.ndarray,
    delta_deg: float,
    outpath: str,
    feature_name: str = "Feature",
    seed: int = 42,
):
    """
    Generate the end-to-end workflow figure.

    Parameters
    ----------
    z : np.ndarray
        (N, 2) coordinates.
    y : np.ndarray
        (N,) binary foreground labels (1=fg, 0=bg).
    v : np.ndarray
        (2,) vantage point.
    theta_grid : np.ndarray
        (B,) angular grid centers in radians.
    delta_deg : float
        Sector width in degrees.
    outpath : str
        Path to save the figure.
    """
    if delta_deg > 180:
        raise ValueError(f"Delta {delta_deg} > 180 degrees is forbidden.")
    if delta_deg >= 180:
        # Exact 180 is allowed per request but loses directionality (covers half-circle)
        print(
            f"WARNING: Delta {delta_deg} >= 180 degrees. Directionality is lost (sector covers half-circle)."
        )
    elif delta_deg >= 90:
        print(f"WARNING: Delta {delta_deg} >= 90 degrees. Directionality may be poor.")

    r, theta = polar_coordinates(z, v)
    config = BioRSPConfig(delta_deg=delta_deg, B=len(theta_grid))
    res = compute_rsp_radar(r, theta, y, config=config)

    valid_mask = ~np.isnan(res.rsp)
    if not np.any(valid_mask):
        print("No valid sectors. Cannot generate figure.")
        return

    # Use argmax of absolute value
    peak_idx = np.nanargmax(np.abs(res.rsp))
    theta_star = res.centers[peak_idx]
    R_star = res.rsp[peak_idx]

    # 3. Compute Summaries
    summaries = compute_scalar_summaries(res, valid_mask=valid_mask)

    # 4. Plotting
    set_publication_style()
    fig = plt.figure(figsize=(get_column_width("double"), 7), constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0], projection="polar")
    ax_d = fig.add_subplot(gs[1, 1])

    # --- Panel A: Embedding ---
    # Scatter points
    ax_a.scatter(
        z[y == 0, 0], z[y == 0, 1], c=COLORS["bg_cells"], s=1, alpha=0.2, label="Background"
    )
    ax_a.scatter(
        z[y == 1, 0], z[y == 1, 1], c=COLORS["fg_cells"], s=2, alpha=0.5, label=feature_name
    )
    ax_a.scatter(v[0], v[1], c="black", marker="x", s=50, label="Vantage")

    # Draw all sector rays
    max_r = np.max(r)
    for phi in res.centers:
        ax_a.plot(
            [v[0], v[0] + 1.2 * max_r * np.cos(phi)],
            [v[1], v[1] + 1.2 * max_r * np.sin(phi)],
            color=COLORS["highlight"],
            alpha=0.05,
            lw=0.5,
        )

    # Draw theta* ray
    ray_x = v[0] + 1.2 * max_r * np.cos(theta_star)
    ray_y = v[1] + 1.2 * max_r * np.sin(theta_star)
    ax_a.plot(
        [v[0], ray_x], [v[1], ray_y], color=COLORS["highlight"], lw=1.5, label=r"Peak $\theta^*$"
    )

    # Draw sector wedge
    delta_rad = np.deg2rad(delta_deg)
    wedge_theta = np.linspace(theta_star - delta_rad / 2, theta_star + delta_rad / 2, 20)
    wedge_x = np.concatenate([[v[0]], v[0] + 1.2 * max_r * np.cos(wedge_theta), [v[0]]])
    wedge_y = np.concatenate([[v[1]], v[1] + 1.2 * max_r * np.sin(wedge_theta), [v[1]]])
    ax_a.fill(wedge_x, wedge_y, color=COLORS["highlight"], alpha=0.1)

    ax_a.set_aspect("equal")
    ax_a.set_title("Embedding & Radar Sweep")
    ax_a.legend(loc="upper right", fontsize=8)
    add_panel_label(ax_a, "A")

    # --- Panel B: Radial ECDFs ---
    # Task 2 & 3: Verify sector membership is correct and matches theta*
    sector_indices_list = get_sector_indices(theta, len(theta_grid), delta_deg)
    indices_star = sector_indices_list[peak_idx]

    r_star_sector = r[indices_star]
    y_star_sector = y[indices_star]

    r_fg = r_star_sector[y_star_sector == 1]
    r_bg = r_star_sector[y_star_sector == 0]

    n_fg = len(r_fg)
    n_bg = len(r_bg)

    if n_fg > 0 and n_bg > 0:
        # Normalize r for display
        r_med = np.median(r)
        r_iqr = np.percentile(r, 75) - np.percentile(r, 25)
        scale_denom = r_iqr if r_iqr > 1e-8 else 1.0

        r_fg_hat = (r_fg - r_med) / scale_denom
        r_bg_hat = (r_bg - r_med) / scale_denom

        # Sort for ECDF
        r_fg_sorted = np.sort(r_fg_hat)
        r_bg_sorted = np.sort(r_bg_hat)
        y_fg = np.linspace(0, 1, n_fg)
        y_bg = np.linspace(0, 1, n_bg)

        ax_b.step(r_bg_sorted, y_bg, color=COLORS["ref_line"], label=r"$P_{\mathrm{bg}}$")
        ax_b.step(r_fg_sorted, y_fg, color=COLORS["fg_cells"], label=r"$P_{\mathrm{fg}}$")

        # Medians
        med_fg = np.median(r_fg_hat)
        med_bg = np.median(r_bg_hat)
        ax_b.axvline(med_bg, color=COLORS["ref_line"], ls="--", lw=1)
        ax_b.axvline(med_fg, color=COLORS["fg_cells"], ls="--", lw=1)

        # Task 4: Fix sign convention
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

    ax_b.set_title(r"Radial ECDFs (Sector $\theta^*$)")
    ax_b.set_xlabel(r"Std. Radius $\hat{r}$")
    ax_b.set_ylabel("Cumulative Prob.")
    ax_b.legend(loc="lower right", fontsize=8)
    add_panel_label(ax_b, "B")

    # --- Panel C: Radar Plot ---
    # Plot R(theta)
    theta_plot = np.concatenate([res.centers, [res.centers[0]]])
    rsp_plot = np.concatenate([res.rsp, [res.rsp[0]]])

    r_max = np.nanmax(np.abs(res.rsp))
    r_lim = np.ceil(r_max * 10) / 10 if r_max > 0 else 0.1
    ax_c.set_ylim(-r_lim, r_lim)
    ax_c.set_rticks([-r_lim, 0, r_lim])
    ax_c.set_yticklabels([f"-{r_lim:.1f}", "0", f"{r_lim:.1f}"], fontsize=7)
    ax_c.set_rlabel_position(22.5)

    ax_c.plot(theta_plot, rsp_plot, marker="o", markersize=3, color="black", lw=1.2)
    ax_c.fill(theta_plot, rsp_plot, color="black", alpha=0.1)
    ax_c.plot(
        np.linspace(0, 2 * np.pi, 100), np.zeros(100), color=COLORS["ref_line"], lw=0.8, ls="--"
    )

    ax_c.set_title(r"Radar Profile $R(\theta)$", pad=20)
    ax_c.set_theta_zero_location("E")
    ax_c.set_theta_direction(1)
    add_panel_label(ax_c, "C")

    # --- Panel D: Summary ---
    # Task 6: Localization metric L and interpretation
    L = summaries.localization_entropy
    R_bar = summaries.r_mean
    A = summaries.anisotropy
    polarity = summaries.polarity

    interpretation = interpret_pattern(A, L, R_bar, polarity)

    vantage_str = config.vantage.replace("_", " ").title()
    param_text = (
        r"$\bf{BioRSP\ Parameters}$" + "\n\n"
        f"$B = {config.B}$ | "
        rf"$\Delta = {config.delta_deg:g}^\circ$ | "
        f"Feature: {feature_name}" + "\n\n"
        f"Vantage: {vantage_str}\n"
        f"QC: {config.qc_mode.title()} | "
        f"Perm: {config.perm_mode.title()}"
    )

    stats_text = (
        r"$\bf{Summary\ Statistics}$" + "\n\n"
        f"$A$ = {A:.3f} | "
        f"$L$ = {L:.3f} | "
        rf"$\bar{{R}}$ = {R_bar:.3f}" + "\n\n"
        f"Interpretation: {interpretation}"
    )

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
    ax_d.set_title("Parameters & Summary Statistics")
    add_panel_label(ax_d, "D")

    # Save
    save_figure(
        fig, f"end_to_end_{feature_name}", outpath.split("/")[0] if "/" in outpath else "figures"
    )
    plt.close()
