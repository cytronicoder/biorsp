#!/usr/bin/env python3
"""Generate publication-quality End-to-End Workflow visualization.

Creates a comprehensive 4-panel figure showing:
- Panel A: Embedding with sector sweep visualization
- Panel B: Radial ECDFs for peak sector (demonstrating sign rule)
- Panel C: Polar radar profile R_g(θ)
- Panel D: Parameters and summary statistics

The figure illustrates the two-score BioRSP output:
- Coverage score C_g (fraction of cells expressing the gene)
- Spatial organization score S_g (RMS magnitude of radar profile)

Usage:
    # With real data
    python scripts/make_end_to_end_workflow.py --adata data.h5ad --feature CD3D --out scripts/output/cd3d.png

    # With demo data
    python scripts/make_end_to_end_workflow.py --feature Demo --out scripts/output/demo.png

Requires:
    Package installation: pip install -e .
"""

import argparse
from pathlib import Path

import numpy as np

try:
    from biorsp.core.geometry import compute_vantage
    from biorsp.plotting.workflow import make_end_to_end_figure
except ImportError as e:
    print("ERROR: Cannot import biorsp. Please install the package first:")
    print("  pip install -e .")
    print(f"Details: {e}")
    exit(1)


def setup_args():
    """Set up command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate End-to-End Workflow Figure",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--adata", type=str, help="Path to AnnData h5ad file (optional, uses demo data if omitted)"
    )
    parser.add_argument(
        "--feature",
        type=str,
        default="Enriched_Gene",
        help="Gene name or feature to visualize",
    )
    parser.add_argument(
        "--embedding-key",
        type=str,
        default="X_umap",
        help="Key in adata.obsm for embedding coordinates",
    )
    parser.add_argument(
        "--q",
        type=float,
        default=0.9,
        help="Internal quantile for foreground definition (NOT coverage threshold)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--out",
        type=str,
        default="scripts/output/end_to_end.png",
        help="Output file path",
    )
    parser.add_argument("--delta-deg", type=float, default=60.0, help="Sector width Δ in degrees")
    parser.add_argument("--B", type=int, default=72, help="Number of angular sectors")
    return parser.parse_args()


def generate_demo_data(seed=42):
    """Generate synthetic data with localized enrichment pattern.

    Simulates a gene with rim localization at angle ~0.5 radians.
    This demonstrates core vs rim detection.
    """
    rng = np.random.default_rng(seed)
    n_cells = 3000
    r_true = np.sqrt(rng.random(n_cells))
    theta_true = 2 * np.pi * rng.random(n_cells) - np.pi
    coords = np.column_stack([r_true * np.cos(theta_true), r_true * np.sin(theta_true)])
    prob = 0.05 + 0.8 * np.exp(-0.5 * ((theta_true - 0.5) / 0.4) ** 2) * (r_true > 0.7)
    expr = rng.binomial(1, prob).astype(float)
    return coords, expr


def main():
    """Main entry point."""
    args = setup_args()
    np.random.seed(args.seed)

    outpath = Path(args.out)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    if args.adata:
        try:
            import scanpy as sc
        except ImportError:
            print("ERROR: scanpy is required to load AnnData files.")
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
            print(f"ERROR: Feature '{args.feature}' not found in adata.var_names or adata.obs")
            exit(1)
    else:
        print("No --adata provided. Using demo data.")
        coords, expr = generate_demo_data(args.seed)

    v = compute_vantage(coords, method="geometric_median")

    # Distinguish biological threshold (coverage) from internal foreground (quantile)
    # For this demo, we use a simple quantile threshold for both
    thresh = np.quantile(expr, args.q)
    y = (expr >= thresh).astype(float) if thresh > 0 else (expr > 0).astype(float)

    # Compute coverage with same threshold (in real analysis, use _detect_threshold)
    coverage_expr = float(np.mean(expr >= thresh))
    foreground_fraction = float(np.mean(y))

    theta_grid = np.linspace(-np.pi, np.pi, args.B, endpoint=False)

    print(f"Generating end-to-end workflow figure for '{args.feature}'...")
    print(f"  Coverage (≥{thresh:.2g}): {coverage_expr:.1%}")
    print(f"  Internal FG fraction: {foreground_fraction:.1%}")

    make_end_to_end_figure(
        z=coords,
        y=y,
        v=v,
        theta_grid=theta_grid,
        delta_deg=args.delta_deg,
        outpath=str(outpath),
        feature_name=args.feature,
        seed=args.seed,
        coverage_expr=coverage_expr,
        expr_threshold=thresh,
        foreground_fraction=foreground_fraction,
    )
    print(f"✅ Figure saved to: {outpath}")


if __name__ == "__main__":
    main()
