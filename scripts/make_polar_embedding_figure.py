#!/usr/bin/env python3
"""
Generate a comprehensive Polar Embedding visualization.
Combines original embedding, polar transformation (theta, r-hat),
polar-reembedding, and radial distribution analysis.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Wedge

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
    parser = argparse.ArgumentParser(description="Generate Comprehensive Polar Embedding Figure")
    parser.add_argument("--adata", type=str, help="Path to AnnData h5ad file")
    parser.add_argument("--coords_csv", type=str, help="Path to coordinates CSV")
    parser.add_argument(
        "--embedding_key", type=str, default="X_umap", help="Embedding key in adata.obsm"
    )
    parser.add_argument(
        "--feature", type=str, default="Enriched_Gene", help="Feature (gene) to visualize"
    )
    parser.add_argument("--q", type=float, default=0.9, help="Quantile for binary foreground")
    parser.add_argument(
        "--abs_threshold", type=float, help="Absolute threshold for binary foreground"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--outdir", type=str, default="figures", help="Output directory")
    parser.add_argument(
        "--sector_idx", type=int, default=0, help="Sector index to highlight in Panel D"
    )
    parser.add_argument(
        "--example_thetas",
        type=float,
        nargs="+",
        default=[0.0, np.pi / 2, -np.pi / 2],
        help="Example angles to show as wedges in Panel A/B",
    )
    return parser.parse_args()


def generate_demo_data(seed=42):
    rng = np.random.default_rng(seed)
    n_cells = 3000

    r_true = np.sqrt(rng.random(n_cells))
    theta_true = 2 * np.pi * rng.random(n_cells) - np.pi
    coords = np.column_stack([r_true * np.cos(theta_true), r_true * np.sin(theta_true)])

    dist_to_v = np.linalg.norm(coords, axis=1)
    prob = 0.05 + 0.8 * np.exp(-0.5 * (theta_true / 0.3) ** 2) * np.exp(
        -0.5 * (dist_to_v / 0.2) ** 2
    )
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
        coords = adata.obsm[args.embedding_key]
        if args.feature in adata.var_names:
            expr = adata[:, args.feature].X.toarray().flatten()
        else:
            expr = adata.obs[args.feature].values
    elif args.coords_csv:
        df = pd.read_csv(args.coords_csv)
        coords = df[["x", "y"]].values
        expr = df[args.feature].values
    else:
        coords, expr = generate_demo_data(args.seed)

    v = compute_vantage(coords, method="geometric_median")
    r, theta = polar_coordinates(coords, v)

    r_med = np.median(r)
    r_iqr = np.percentile(r, 75) - np.percentile(r, 25)
    r_hat = (r - r_med) / (r_iqr + 1e-8)

    if args.abs_threshold is not None:
        y = (expr >= args.abs_threshold).astype(float)
    else:
        thresh = np.quantile(expr, args.q)
        y = (expr >= thresh).astype(float) if thresh > 0 else (expr > 0).astype(float)
    fg_mask = y > 0

    fig, axes = plt.subplots(2, 2, figsize=(get_column_width("double"), 7))

    ax = axes[0, 0]
    ax.scatter(coords[~fg_mask, 0], coords[~fg_mask, 1], c=COLORS["bg_cells"], s=1, alpha=0.2)
    ax.scatter(
        coords[fg_mask, 0],
        coords[fg_mask, 1],
        c=COLORS["fg_cells"],
        s=2,
        alpha=0.5,
        label="Foreground",
    )
    ax.scatter(v[0], v[1], c="black", marker="x", s=50, label="Vantage $v$")

    max_r_plot = np.max(r) * 1.1
    for t in args.example_thetas:
        wedge = Wedge(
            v,
            max_r_plot,
            np.degrees(t) - 10,
            np.degrees(t) + 10,
            facecolor=COLORS["highlight"],
            alpha=0.1,
        )
        ax.add_patch(wedge)

    ax.set_title("Original Embedding $(x,y)$")
    ax.set_aspect("equal")
    add_panel_label(ax, "A")

    ax = axes[0, 1]
    ax.scatter(theta[~fg_mask], r_hat[~fg_mask], c=COLORS["bg_cells"], s=1, alpha=0.2)
    ax.scatter(theta[fg_mask], r_hat[fg_mask], c=COLORS["fg_cells"], s=2, alpha=0.5)

    for t in args.example_thetas:
        ax.axvspan(t - np.radians(10), t + np.radians(10), color=COLORS["highlight"], alpha=0.1)

    ax.set_title(r"Polar Embedding $(\theta, \hat{r})$")
    ax.set_xlabel(r"Angle $\theta$ (rad)")
    ax.set_ylabel(r"Std. Radius $\hat{r}$")
    ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
    add_panel_label(ax, "B")

    ax = axes[1, 0]
    u = r_hat * np.cos(theta)
    vv = r_hat * np.sin(theta)
    ax.scatter(u[~fg_mask], vv[~fg_mask], c=COLORS["bg_cells"], s=1, alpha=0.2)
    ax.scatter(u[fg_mask], vv[fg_mask], c=COLORS["fg_cells"], s=2, alpha=0.5)
    ax.set_title("Polar-reembedded $(u, v)$")
    ax.set_xlabel(r"$u = \hat{r} \cos \theta$")
    ax.set_ylabel(r"$v = \hat{r} \sin \theta$")
    ax.set_aspect("equal")
    add_panel_label(ax, "C")

    ax = axes[1, 1]
    B = 36
    delta_deg = 360 / B
    sector_indices = get_sector_indices(theta, B, delta_deg=delta_deg)
    idx = sector_indices[args.sector_idx]

    if idx.size > 0:
        r_s = r_hat[idx]
        y_s = y[idx]
        r_fg = r_s[y_s > 0]
        r_bg = r_s[y_s == 0]

        if r_fg.size > 1 and r_bg.size > 1:
            ax.step(
                np.sort(r_bg),
                np.linspace(0, 1, len(r_bg)),
                color=COLORS["ref_line"],
                label=r"$P_{\mathrm{bg}}$",
            )
            ax.step(
                np.sort(r_fg),
                np.linspace(0, 1, len(r_fg)),
                color=COLORS["fg_cells"],
                label=r"$P_{\mathrm{fg}}$",
            )
            med_fg = np.median(r_fg)
            med_bg = np.median(r_bg)
            ax.axvline(med_bg, color=COLORS["ref_line"], ls="--", lw=1)
            ax.axvline(med_fg, color=COLORS["fg_cells"], ls="--", lw=1)

            sign_str = "Core (+)" if med_fg < med_bg else "Rim (-)"
            ax.text(
                0.5,
                0.1,
                f"Sector {args.sector_idx}\n{sign_str}",
                transform=ax.transAxes,
                ha="center",
                bbox=dict(
                    facecolor="white",
                    edgecolor=COLORS["ref_line"],
                    alpha=0.8,
                    boxstyle="round,pad=0.3",
                ),
                fontsize=8,
            )

    ax.set_title("Radial ECDFs")
    ax.set_xlabel(r"Std. Radius $\hat{r}$")
    ax.set_ylabel("Cumulative Prob.")
    ax.legend(fontsize=7, loc="upper left")
    add_panel_label(ax, "D")

    save_figure(fig, "fig_polar_embedding_comprehensive", outdir=outdir)


if __name__ == "__main__":
    main()
