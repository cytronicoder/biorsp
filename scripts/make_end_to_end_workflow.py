#!/usr/bin/env python3
"""
Generate a comprehensive End-to-End Workflow visualization.
Combines embedding analysis, radial distributions, polar radar profiles,
and summary statistics.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import biorsp
from biorsp.core.engine import compute_rsp_radar
from biorsp.plotting.style import (
    COLORS,
    add_panel_label,
    get_column_width,
    save_figure,
    set_publication_style,
)
from biorsp.preprocess.geometry import compute_vantage, get_sector_indices, polar_coordinates

set_publication_style()


def setup_args():
    parser = argparse.ArgumentParser(description="Generate End-to-End Workflow Figure")
    parser.add_argument("--adata", type=str, help="Path to AnnData h5ad file")
    parser.add_argument("--feature", type=str, default="Enriched_Gene", help="Feature to visualize")
    parser.add_argument("--q", type=float, default=0.9, help="Quantile for foreground")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--outdir", type=str, default="figures", help="Output directory")
    return parser.parse_args()


def generate_demo_data(seed=42):
    rng = np.random.default_rng(seed)
    n_cells = 3000
    r_true = np.sqrt(rng.random(n_cells))
    theta_true = 2 * np.pi * rng.random(n_cells) - np.pi
    coords = np.column_stack([r_true * np.cos(theta_true), r_true * np.sin(theta_true)])
    prob = 0.05 + 0.8 * np.exp(-0.5 * ((theta_true - 0.5) / 0.4) ** 2)
    expr = rng.binomial(1, prob).astype(float)
    return coords, expr


def main():
    args = setup_args()
    np.random.seed(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.adata:
        import scanpy as sc

        adata = sc.read_h5ad(args.adata)
        coords = adata.obsm["X_umap"]
        expr = (
            adata[:, args.feature].X.toarray().flatten()
            if args.feature in adata.var_names
            else adata.obs[args.feature].values
        )
    else:
        coords, expr = generate_demo_data(args.seed)

    v = compute_vantage(coords)
    r, theta = polar_coordinates(coords, v)

    r_med = np.median(r)
    r_iqr = np.percentile(r, 75) - np.percentile(r, 25)
    r_hat = (r - r_med) / (r_iqr + 1e-8)

    config = biorsp.BioRSPConfig()
    thresh = np.quantile(expr, args.q)
    y = (expr >= thresh).astype(float) if thresh > 0 else (expr > 0).astype(float)
    res = compute_rsp_radar(r, theta, y, config=config)

    valid = ~np.isnan(res.rsp)
    if not np.any(valid):
        print("No valid sectors found.")
        return
    peak_idx = np.nanargmax(np.abs(res.rsp))
    peak_theta = res.centers[peak_idx]

    fig = plt.figure(figsize=(get_column_width("double"), 7), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0], projection="polar")
    ax_d = fig.add_subplot(gs[1, 1])

    ax_a.scatter(coords[:, 0], coords[:, 1], c=COLORS["bg_cells"], s=1, alpha=0.2)
    ax_a.scatter(coords[y > 0, 0], coords[y > 0, 1], c=COLORS["fg_cells"], s=2, alpha=0.5)
    ax_a.scatter(v[0], v[1], c="black", marker="x", s=50)

    for phi in res.centers:
        ax_a.plot(
            [v[0], v[0] + 1.2 * np.max(r) * np.cos(phi)],
            [v[1], v[1] + 1.2 * np.max(r) * np.sin(phi)],
            color=COLORS["highlight"],
            alpha=0.05,
            lw=0.5,
        )
    ax_a.plot(
        [v[0], v[0] + 1.2 * np.max(r) * np.cos(peak_theta)],
        [v[1], v[1] + 1.2 * np.max(r) * np.sin(peak_theta)],
        color=COLORS["highlight"],
        lw=1.5,
        label=r"Peak $\theta^*$",
    )
    ax_a.set_title("Embedding & Radar Sweep")
    ax_a.set_aspect("equal")
    ax_a.legend(fontsize=8, loc="upper right")
    add_panel_label(ax_a, "A")

    sector_indices = get_sector_indices(theta, config.B, config.delta_deg)
    idx = sector_indices[peak_idx]
    r_s = r_hat[idx]
    y_s = y[idx]
    r_fg = r_s[y_s > 0]
    r_bg = r_s[y_s == 0]

    if r_fg.size > 1 and r_bg.size > 1:
        ax_b.step(
            np.sort(r_bg),
            np.linspace(0, 1, len(r_bg)),
            color=COLORS["ref_line"],
            label=r"$P_{\mathrm{bg}}$",
        )
        ax_b.step(
            np.sort(r_fg),
            np.linspace(0, 1, len(r_fg)),
            color=COLORS["fg_cells"],
            label=r"$P_{\mathrm{fg}}$",
        )
        med_fg = np.median(r_fg)
        med_bg = np.median(r_bg)
        ax_b.axvline(med_bg, color=COLORS["ref_line"], ls="--", lw=1)
        ax_b.axvline(med_fg, color=COLORS["fg_cells"], ls="--", lw=1)
        ax_b.set_title(r"Radial ECDFs (Sector $\theta^*$)")
        ax_b.set_xlabel(r"Std. Radius $\hat{r}$")
        ax_b.set_ylabel("Cumulative Prob.")
        ax_b.legend(fontsize=8)
        sign_str = "Core (+)" if med_fg < med_bg else "Rim (-)"
        ax_b.text(
            0.5,
            0.1,
            f"Sign: {sign_str}\n$R(\\theta^*) = {res.rsp[peak_idx]:.2f}$",
            transform=ax_b.transAxes,
            bbox=dict(
                facecolor="white", edgecolor=COLORS["ref_line"], alpha=0.8, boxstyle="round,pad=0.3"
            ),
            fontsize=8,
        )
    add_panel_label(ax_b, "B")

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

    ax_d.axis("off")
    A = np.sqrt(np.nanmean(res.rsp**2))
    R_bar = np.nanmean(res.rsp)
    abs_r = np.abs(res.rsp[valid])
    p = abs_r / np.sum(abs_r)
    L = 1 - (-np.sum(p * np.log(p + 1e-12)) / np.log(len(p)))

    vantage_str = config.vantage.replace("_", " ").title()
    param_text = (
        r"$\bf{BioRSP\ Parameters}$" + "\n\n"
        f"$B = {config.B}$ | "
        rf"$\Delta = {config.delta_deg:g}^\circ$ | "
        f"$q = {args.q}$" + "\n\n"
        f"Vantage: {vantage_str}\n"
        f"QC: {config.qc_mode.title()} | "
        f"Perm: {config.perm_mode.title()}"
    )

    stats_text = (
        r"$\bf{Summary\ Statistics}$" + "\n\n"
        f"$A$ = {A:.3f} | "
        f"$L$ = {L:.3f} | "
        rf"$\bar{{R}}$ = {R_bar:.3f}" + "\n\n"
        f"{'Strong' if A > 0.1 else 'Weak'} pattern, "
        f"{'Localized' if L > 0.5 else 'Diffuse'} wedge, "
        f"{'Core' if R_bar > 0 else 'Rim'} bias."
    )

    bbox_props = dict(
        boxstyle="round,pad=0.8", facecolor="#F8F9FA", edgecolor=COLORS["ref_line"], alpha=0.5
    )

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

    save_figure(fig, "fig_end_to_end_workflow", outdir=outdir)


if __name__ == "__main__":
    main()
