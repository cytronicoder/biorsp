#!/usr/bin/env python3
"""
Generate a comprehensive End-to-End Workflow visualization.
Combines embedding analysis, radial distributions, polar radar profiles,
and summary statistics.
"""

import argparse
from pathlib import Path

import numpy as np

from biorsp.plotting.workflow import make_end_to_end_figure
from biorsp.preprocess.geometry import angle_grid, compute_vantage


def setup_args():
    parser = argparse.ArgumentParser(description="Generate End-to-End Workflow Figure")
    parser.add_argument("--adata", type=str, help="Path to AnnData h5ad file")
    parser.add_argument("--feature", type=str, default="Enriched_Gene", help="Feature to visualize")
    parser.add_argument("--q", type=float, default=0.9, help="Quantile for foreground")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--outdir", type=str, default="figures", help="Output directory")
    parser.add_argument("--delta_deg", type=float, default=60.0, help="Sector width in degrees")
    parser.add_argument("--B", type=int, default=72, help="Number of sectors")
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

    thresh = np.quantile(expr, args.q)
    y = (expr >= thresh).astype(float) if thresh > 0 else (expr > 0).astype(float)

    theta_grid = angle_grid(args.B)

    outpath = str(outdir / f"end_to_end_{args.feature}.png")

    make_end_to_end_figure(
        z=coords,
        y=y,
        v=v,
        theta_grid=theta_grid,
        delta_deg=args.delta_deg,
        outpath=outpath,
        feature_name=args.feature,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
