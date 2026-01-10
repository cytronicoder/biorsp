#!/usr/bin/env python3
"""
Cross-Embedding Stability Benchmark.

Tests whether BioRSP scores are stable across different 2D embeddings
(UMAP with different seeds/parameters).

Usage:
    python run_stability.py --mode quick --outdir outputs/stability --seed 42

Outputs:
    - fig_stability_embeddings.png
    - stability_metrics.json
    - report.md
"""

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from biorsp import BioRSPConfig

MODE_CONFIGS = {
    "quick": {
        "n_cells": 1000,
        "n_genes": 30,
        "n_embeddings": 3,
    },
    "validation": {
        "n_cells": 2000,
        "n_genes": 50,
        "n_embeddings": 5,
    },
    "publication": {
        "n_cells": 3000,
        "n_genes": 80,
        "n_embeddings": 7,
    },
}


def generate_umap_embeddings(X_high_dim, n_embeddings, rng):
    """Generate multiple UMAP embeddings with different seeds."""
    try:
        import umap
    except ImportError:
        print("Warning: umap-learn not installed. Using PCA + noise instead.")
        return generate_pca_embeddings(X_high_dim, n_embeddings, rng)

    embeddings = []
    for i in range(n_embeddings):
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=15 + i * 5,  # Vary neighborhood
            min_dist=0.1 + i * 0.05,
            random_state=42 + i * 100,
        )
        emb = reducer.fit_transform(X_high_dim)
        embeddings.append(emb)

    return embeddings


def generate_pca_embeddings(X_high_dim, n_embeddings, rng):
    """Fallback: PCA with added noise for variation."""
    from scipy.linalg import svd

    # Center and PCA
    X_centered = X_high_dim - X_high_dim.mean(axis=0)
    U, s, Vt = svd(X_centered, full_matrices=False)
    coords_base = U[:, :2] * s[:2]

    embeddings = []
    for i in range(n_embeddings):
        noise_scale = 0.05 * (i + 1)
        noise = rng.normal(0, noise_scale, coords_base.shape)
        emb = coords_base + noise * np.std(coords_base)
        embeddings.append(emb)

    return embeddings


def run_stability(args):
    """Run cross-embedding stability benchmark."""
    from simlib import (
        datasets,
        expression,
        io,
        metrics,
        rng as rng_module,
        scoring,
        shapes,
    )

    mode_cfg = MODE_CONFIGS[args.mode]

    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Cross-Embedding Stability Benchmark")
    print(f"Mode: {args.mode}")
    print(f"Embeddings: {mode_cfg['n_embeddings']}")
    print("=" * 60)

    start_time = time.time()
    gen = rng_module.make_rng(args.seed, "stability")

    config = BioRSPConfig(
        B=72,
        delta_deg=60.0,
        n_permutations=0,  # No permutations for speed
        qc_mode="principled",
    )

    # Generate base coordinates and gene panel
    print("\n[1/4] Generating base dataset...")

    # Base 2D coordinates (ground truth spatial structure)
    coords_base, _ = shapes.generate_coords("disk", mode_cfg["n_cells"], gen)
    libsize = expression.simulate_library_size(
        mode_cfg["n_cells"], gen, model="lognormal", params={"mean": 2000, "sigma": 0.5}
    )

    # Generate genes with factorial design (half structured, half iid)
    n_per_arch = mode_cfg["n_genes"] // 4
    X, var_names, truth_df = datasets.make_factorial_panel(
        coords=coords_base,
        libsize=libsize,
        rng=gen,
        n_per_archetype=n_per_arch,
    )

    # Generate multiple embeddings
    print("[2/4] Generating embeddings...")

    n_latent_dims = 20
    X_latent = np.zeros((mode_cfg["n_cells"], n_latent_dims))

    # First 2 dims are the true spatial coords (scaled)
    X_latent[:, 0] = coords_base[:, 0]
    X_latent[:, 1] = coords_base[:, 1]

    for d in range(2, n_latent_dims):
        if d < 5:
            # Correlated with spatial
            X_latent[:, d] = X_latent[:, d % 2] + gen.normal(0, 0.3, mode_cfg["n_cells"])
        else:
            # Pure noise
            X_latent[:, d] = gen.normal(0, 1, mode_cfg["n_cells"])

    embeddings = generate_umap_embeddings(X_latent, mode_cfg["n_embeddings"], gen)
    print(f"   Generated {len(embeddings)} embeddings")

    # Score genes on each embedding
    print("[3/4] Scoring genes on each embedding...")

    scores_list = []
    labels_list = []

    for emb_idx, coords in enumerate(embeddings):
        print(f"   Embedding {emb_idx + 1}/{len(embeddings)}...")

        adata = datasets.package_as_anndata(coords, X, var_names, embedding_key="X_sim")

        scores_df = scoring.score_dataset(
            adata, genes=var_names, config=config, embedding_key="X_sim"
        )

        s_scores = scores_df.set_index("gene").loc[var_names, "spatial_score"].values
        scores_list.append(s_scores)

        # Classify
        c_scores = scores_df.set_index("gene").loc[var_names, "coverage_expr"].values
        labels = metrics.classify_by_quadrant(c_scores, s_scores)
        labels_list.append(labels)

    print("[4/4] Computing stability metrics...")

    stability = metrics.embedding_stability_metrics(
        scores_list=scores_list,
        labels_list=labels_list,
    )

    print("\nResults:")
    print(f"  Score correlation (mean): {stability['score_correlation']:.3f}")
    print(f"  Score correlation (min): {stability['score_correlation_min']:.3f}")
    print(f"  Label agreement: {stability.get('label_agreement', np.nan):.3f}")
    print(f"  Score std (mean): {stability['score_std_mean']:.3f}")

    # Generate figures
    print("\nGenerating figures...")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    n_emb = len(scores_list)
    corr_matrix = np.zeros((n_emb, n_emb))
    for i in range(n_emb):
        for j in range(n_emb):
            mask = ~(np.isnan(scores_list[i]) | np.isnan(scores_list[j]))
            if mask.sum() > 3:
                corr_matrix[i, j] = np.corrcoef(scores_list[i][mask], scores_list[j][mask])[0, 1]

    im = axes[0].imshow(corr_matrix, cmap="RdYlGn", vmin=0.5, vmax=1.0)
    axes[0].set_title("S Score Correlation\nAcross Embeddings", fontweight="bold")
    axes[0].set_xlabel("Embedding")
    axes[0].set_ylabel("Embedding")
    plt.colorbar(im, ax=axes[0], shrink=0.8)

    mask = ~(np.isnan(scores_list[0]) | np.isnan(scores_list[1]))
    axes[1].scatter(
        scores_list[0][mask],
        scores_list[1][mask],
        alpha=0.6,
        s=30,
        c=truth_df["organization_regime"]
        .map({"iid": "#9E9E9E", "structured": "#FF5722"})
        .values[mask],
    )
    axes[1].plot([0, 1], [0, 1], "k--", alpha=0.5)
    axes[1].set_xlabel("S (Embedding 1)")
    axes[1].set_ylabel("S (Embedding 2)")
    axes[1].set_title("Score Agreement\n(Embeddings 1 vs 2)", fontweight="bold")
    r = corr_matrix[0, 1]
    axes[1].text(0.05, 0.95, f"r = {r:.3f}", transform=axes[1].transAxes, fontsize=12, va="top")

    metric_names = ["Score\nCorrelation", "Label\nAgreement"]
    metric_vals = [stability["score_correlation"], stability.get("label_agreement", 0)]
    colors = ["#2196F3", "#4CAF50"]

    bars = axes[2].bar(metric_names, metric_vals, color=colors, alpha=0.8, edgecolor="black")
    for bar, val in zip(bars, metric_vals):
        axes[2].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    axes[2].set_ylim(0, 1.1)
    axes[2].axhline(0.9, color="green", linestyle="--", alpha=0.5, label="Target: 0.9")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Stability Summary", fontweight="bold")
    axes[2].legend(loc="lower right")

    plt.suptitle("Cross-Embedding Stability", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    io.save_figure(fig, output_dir, "fig_stability_embeddings.png")
    plt.close(fig)

    # Save metrics
    with open(output_dir / "stability_metrics.json", "w") as f:
        json.dump(
            {
                k: float(v) if isinstance(v, (np.floating, float)) else v
                for k, v in stability.items()
            },
            f,
            indent=2,
        )

    # Report
    elapsed = time.time() - start_time

    stability_pass = stability["score_correlation"] >= 0.8

    report = f"""# Cross-Embedding Stability Report

Mode: {args.mode}
Runtime: {elapsed:.1f}s
N embeddings: {mode_cfg['n_embeddings']}
N genes: {len(var_names)}

## Summary

BioRSP scores should be stable across different 2D embeddings of the same data.

## Results

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Score Correlation (mean) | {stability['score_correlation']:.3f} | ≥ 0.80 | {'✓ PASS' if stability_pass else '✗ FAIL'} |
| Score Correlation (min) | {stability['score_correlation_min']:.3f} | ≥ 0.70 | - |
| Label Agreement | {stability.get('label_agreement', np.nan):.3f} | ≥ 0.80 | - |
| Score Std (per gene) | {stability['score_std_mean']:.3f} | ≤ 0.10 | - |

## Interpretation

- **Score Correlation**: Pearson correlation of S scores between embeddings.
  High correlation (>0.9) indicates that spatial scores are robust to embedding variation.

- **Label Agreement**: Fraction of genes classified into the same archetype across embeddings.
  High agreement indicates stable classification.

- **Score Std**: Standard deviation of S across embeddings for each gene.
  Low values indicate consistent scores.

Different UMAP seeds/parameters create slightly different 2D projections of the same
high-dimensional data. BioRSP should identify the same spatial patterns regardless
of these minor differences.
"""

    with open(output_dir / "report.md", "w") as f:
        f.write(report)

    print(f"\n✓ Stability benchmark complete! Outputs in {output_dir}")

    return stability


def main():
    parser = argparse.ArgumentParser(description="Run cross-embedding stability benchmark")
    parser.add_argument(
        "--mode", type=str, choices=["quick", "validation", "publication"], default="quick"
    )
    parser.add_argument("--outdir", type=str, default=str(ROOT / "outputs" / "stability"))
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    run_stability(args)


if __name__ == "__main__":
    main()
