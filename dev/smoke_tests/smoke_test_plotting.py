"""Smoke test CLI for BioRSP plotting.

Generates all plot types deterministically to validate:
- Embedding plots (with scalability features)
- Radar plots (all modes + debug overlay)
- Workflow figures
- Figure metadata outputs

Usage:
    python scripts/smoke_test_plotting.py
    python scripts/smoke_test_plotting.py --outdir outputs/smoke_plots
"""

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

from biorsp.core.engine import compute_rsp_radar
from biorsp.core.geometry import compute_vantage, polar_coordinates
from biorsp.core.summaries import compute_scalar_summaries
from biorsp.plotting.embedding import plot_embedding
from biorsp.plotting.radar import plot_radar, plot_radar_absolute
from biorsp.plotting.style import publication_style, save_figure
from biorsp.plotting.workflow import make_end_to_end_figure
from biorsp.preprocess.foreground import define_foreground
from biorsp.preprocess.normalization import normalize_radii
from biorsp.utils.config import BioRSPConfig


def generate_synthetic_data(n=1000, pattern="rim_wedge", seed=42):
    """Generate synthetic spatial data with known pattern."""
    rng = np.random.default_rng(seed)

    r_true = np.sqrt(rng.random(n))
    theta_true = 2 * np.pi * rng.random(n) - np.pi
    coords = np.column_stack([r_true * np.cos(theta_true), r_true * np.sin(theta_true)])

    if pattern == "rim_wedge":
        prob_base = 0.05
        prob_spatial = 0.8 * (r_true > 0.6) * np.exp(-0.5 * (theta_true / 0.5) ** 2)
        prob = np.clip(prob_base + prob_spatial, 0, 1)
        expr = rng.binomial(10, prob).astype(float)
    elif pattern == "core_global":
        prob_base = 0.05
        prob_spatial = 0.7 * (r_true < 0.5)
        prob = np.clip(prob_base + prob_spatial, 0, 1)
        expr = rng.binomial(10, prob).astype(float)
    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    return coords, expr


def save_metadata(outpath: Path, metadata: dict):
    """Save figure metadata as JSON sidecar."""
    json_path = outpath.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata: {json_path.name}")


def test_embedding_plots(outdir: Path, coords: np.ndarray, expr: np.ndarray):
    """Test embedding plot with scalability features."""
    print("\n1. Testing embedding plots...")

    v = compute_vantage(coords, method="geometric_median", seed=42)

    with publication_style():
        fig, ax = plt.subplots(figsize=(6, 6))
        plot_embedding(coords, c=expr, ax=ax, title="Expression", show_vantage=True, vantage=v)
        paths = save_figure(fig, outdir / "01_embedding_expression", close=True)
        print(f"  ✓ {paths[0].name}")

        fg_mask, _ = define_foreground(expr, mode="quantile", q=0.9)
        fig, ax = plt.subplots(figsize=(6, 6))
        plot_embedding(
            coords, c=fg_mask, ax=ax, title="Foreground/Background", show_vantage=True, vantage=v
        )
        paths = save_figure(fig, outdir / "02_embedding_fg_bg", close=True)
        print(f"  ✓ {paths[0].name}")

        fig, ax = plt.subplots(figsize=(6, 6))
        plot_embedding(
            coords,
            c=expr,
            ax=ax,
            title="Subsampled (500 pts)",
            max_points=500,
            subsample_seed=42,
            rasterized=True,
            show_vantage=True,
            vantage=v,
        )
        paths = save_figure(fig, outdir / "03_embedding_subsampled", close=True)
        print(f"  ✓ {paths[0].name}")


def test_radar_plots(outdir: Path, coords: np.ndarray, expr: np.ndarray, config: BioRSPConfig):
    """Test radar plots (all modes + debug)."""
    print("\n2. Testing radar plots...")

    v = compute_vantage(coords, method="geometric_median", seed=42)
    r, theta = polar_coordinates(coords, v)
    r_norm, norm_stats = normalize_radii(r)

    fg_mask, _ = define_foreground(expr, mode="quantile", q=0.9)
    radar = compute_rsp_radar(r_norm, theta, fg_mask, config=config)
    summaries = compute_scalar_summaries(radar)

    with publication_style():
        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})
        plot_radar(radar, ax=ax, title="Signed RSP", mode="signed", theta_convention="math")
        paths = save_figure(fig, outdir / "04_radar_signed", close=True)
        print(f"  ✓ {paths[0].name}")

        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})
        plot_radar(
            radar,
            ax=ax,
            title="Debug Overlay",
            mode="signed",
            theta_convention="math",
            debug_overlay=True,
        )
        paths = save_figure(fig, outdir / "05_radar_debug", close=True)
        print(f"  ✓ {paths[0].name}")

        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})
        plot_radar(
            radar, ax=ax, title="Compass Convention", mode="signed", theta_convention="compass"
        )
        paths = save_figure(fig, outdir / "06_radar_compass", close=True)
        print(f"  ✓ {paths[0].name}")

        fig = plot_radar_absolute(
            radar, title="Split Proximal/Distal", theta_convention="math", summaries=summaries
        )
        paths = save_figure(fig, outdir / "07_radar_absolute", close=True)
        print(f"  ✓ {paths[0].name}")

    metadata = {
        "pattern": "rim_wedge",
        "config": {
            "delta_deg": config.delta_deg,
            "B": config.B,
            "empty_fg_policy": config.empty_fg_policy,
        },
        "metrics": {
            "Spatial_Score": float(summaries.anisotropy),
            "Directionality": float(summaries.r_mean),
            "coverage_geom": float(summaries.coverage_geom),
        },
        "normalization": norm_stats,
    }
    save_metadata(outdir / "07_radar_absolute", metadata)


def test_workflow_figures(outdir: Path, coords: np.ndarray, expr: np.ndarray, config: BioRSPConfig):
    """Test end-to-end workflow figures."""
    print("\n3. Testing workflow figures...")

    v = compute_vantage(coords, method="geometric_median", seed=42)
    r, theta = polar_coordinates(coords, v)

    thresh = 1.0 if np.allclose(expr, np.round(expr)) else 1e-6
    coverage_expr = float(np.mean(expr >= thresh))

    fg_mask, _ = define_foreground(expr, mode="quantile", q=0.9)
    foreground_fraction = float(np.mean(fg_mask))

    theta_grid = np.linspace(-np.pi, np.pi, config.B, endpoint=False)

    make_end_to_end_figure(
        z=coords,
        y=fg_mask,
        v=v,
        theta_grid=theta_grid,
        delta_deg=config.delta_deg,
        outpath=str(outdir / "08_workflow_standard"),
        feature_name="Test Gene",
        seed=42,
        coverage_expr=coverage_expr,
        expr_threshold=thresh,
        foreground_fraction=foreground_fraction,
        debug=False,
    )
    print("  ✓ 08_workflow_standard.pdf")

    make_end_to_end_figure(
        z=coords,
        y=fg_mask,
        v=v,
        theta_grid=theta_grid,
        delta_deg=config.delta_deg,
        outpath=str(outdir / "09_workflow_debug"),
        feature_name="Test Gene",
        seed=42,
        coverage_expr=coverage_expr,
        expr_threshold=thresh,
        foreground_fraction=foreground_fraction,
        debug=True,
    )
    print("  ✓ 09_workflow_debug.pdf")

    metadata = {
        "pattern": "rim_wedge",
        "Coverage": coverage_expr,
        "expr_threshold": thresh,
        "foreground_fraction": foreground_fraction,
        "config": {
            "delta_deg": config.delta_deg,
            "B": config.B,
        },
        "note": "coverage_expr != foreground_fraction (biological vs internal threshold)",
    }
    save_metadata(outdir / "09_workflow_debug", metadata)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="BioRSP plotting smoke test")
    parser.add_argument(
        "--outdir",
        type=str,
        default="outputs/smoke_plots",
        help="Output directory for test plots",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("BioRSP Plotting Smoke Test")
    print("=" * 60)
    print(f"Output: {outdir}")

    print("\nGenerating synthetic data...")
    coords, expr = generate_synthetic_data(n=1000, pattern="rim_wedge", seed=args.seed)
    print(f"  N = {len(coords)} cells")
    print(f"  Expression range: [{expr.min():.1f}, {expr.max():.1f}]")

    config = BioRSPConfig(
        delta_deg=30,
        B=24,
        empty_fg_policy="zero",
        seed=args.seed,
    )

    test_embedding_plots(outdir, coords, expr)
    test_radar_plots(outdir, coords, expr, config)
    test_workflow_figures(outdir, coords, expr, config)

    print("\n" + "=" * 60)
    print("✅ All smoke tests PASSED!")
    print("=" * 60)
    print(f"\nOutputs saved to: {outdir.absolute()}")

    plot_files = list(outdir.glob("*.pdf")) + list(outdir.glob("*.png"))
    json_files = list(outdir.glob("*.json"))
    print("\nGenerated files:")
    print(f"  {len(plot_files)} plot files")
    print(f"  {len(json_files)} metadata files")


if __name__ == "__main__":
    main()
