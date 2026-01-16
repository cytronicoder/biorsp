"""Generate Polar Re-parameterization Diagnostic Figure.

This figure illustrates how BioRSP transforms Cartesian coordinates (x,y)
into polar coordinates (θ, r) for directional spatial analysis. It does NOT
claim to generate a new embedding, but rather shows the geometric transformation.

Panels:
- A: Original embedding with highlighted example sectors
- B: Polar representation (θ, r̂) showing angular structure
- C: Cartesian projection of polar coords (for visualization)
- D: Radial ECDF comparison in an example sector

Usage:
    python scripts/make_polar_embedding_figure.py --adata data.h5ad --feature CD3D
    python scripts/make_polar_embedding_figure.py  # uses demo data

Requires:
    Package installation: pip install -e .
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Wedge

try:
    from biorsp.core.geometry import compute_vantage, get_sector_indices, polar_coordinates
    from biorsp.plotting.style import (
        COLORS,
        add_panel_label,
        get_column_width,
    )
except ImportError as e:
    print("ERROR: Cannot import biorsp. Please install the package first:")
    print("  pip install -e .")
    print(f"Details: {e}")
    exit(1)


def setup_args():
    """Set up command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate Polar Re-parameterization Diagnostic Figure",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--adata", type=str, help="Path to AnnData h5ad file (optional)")
    parser.add_argument("--coords_csv", type=str, help="Path to coordinates CSV (optional)")
    parser.add_argument(
        "--embedding-key", type=str, default="X_umap", help="Embedding key in adata.obsm"
    )
    parser.add_argument(
        "--feature", type=str, default="Enriched_Gene", help="Gene name to visualize"
    )
    parser.add_argument(
        "--q",
        type=float,
        default=0.9,
        help="Quantile for binary foreground (internal, NOT coverage)",
    )
    parser.add_argument(
        "--abs-threshold", type=float, help="Absolute threshold for binary foreground (overrides q)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--out",
        type=str,
        default="scripts/output/polar_reparameterization.png",
        help="Output file path",
    )
    parser.add_argument(
        "--sector-idx", type=int, default=0, help="Sector index to highlight in Panel D"
    )
    parser.add_argument(
        "--example-thetas",
        type=float,
        nargs="+",
        default=[0.0, np.pi / 2, -np.pi / 2],
        help="Example angles (radians) to show as wedges in Panels A/B",
    )
    parser.add_argument(
        "--smoke", action="store_true", help="Run in fast smoke test mode with demo data"
    )
    parser.add_argument("--outdir", type=str, help="Output directory for smoke test mode")
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
    """Main entry point."""
    args = setup_args()
    np.random.seed(args.seed)

    if args.smoke:
        if args.outdir:
            outpath = Path(args.outdir) / "polar_reparameterization.png"
        else:
            outpath = Path("polar_reparameterization.png")
        # Force demo data in smoke mode
        args.adata = None
        args.coords_csv = None
    else:
        outpath = Path(args.out)

    outpath.parent.mkdir(parents=True, exist_ok=True)

    if args.adata:
        try:
            import scanpy as sc
        except ImportError:
            print("ERROR: scanpy required to load AnnData")
            print("  pip install scanpy")
            exit(1)

        adata = sc.read_h5ad(args.adata)
        coords = adata.obsm[args.embedding_key]
        if args.feature in adata.var_names:
            from scipy import sparse

            X = adata[:, args.feature].X
            expr = X.toarray().flatten() if sparse.issparse(X) else X
        elif args.feature in adata.obs.columns:
            expr = adata.obs[args.feature].values
        else:
            print(f"ERROR: Feature '{args.feature}' not found")
            exit(1)
    elif args.coords_csv:
        df = pd.read_csv(args.coords_csv)
        coords = df[["x", "y"]].values
        expr = df[args.feature].values
    else:
        print("No --adata or --coords-csv provided. Using demo data.")
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
        label="High Expression",
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

    ax.set_title(f"Original Embedding: {args.feature}")
    ax.set_xlabel("Embedding Dim 1")
    ax.set_ylabel("Embedding Dim 2")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=7)
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
    ax.set_title(r"Cartesian Projection of $(\theta, \hat{r})$")
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

    ax.set_title("Radial ECDFs in Example Sector")
    ax.set_xlabel(r"Std. Radius $\hat{r}$")
    ax.set_ylabel("Cumulative Prob.")
    ax.legend(fontsize=7, loc="upper left")
    add_panel_label(ax, "D")

    plt.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    fig.savefig(outpath.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    print(f"✅ Figure saved to: {outpath} and {outpath.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
