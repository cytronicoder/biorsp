"""
Validated plotting utilities for disease-stratified BioRSP analysis.

This module provides plotting functions with scientific consistency validation:
- Archetype scatter plots with quadrant-label validation
- Gene exemplar plots showing both coverage and foreground masks
- Debug plots for intermediate processing steps
"""

from __future__ import annotations

import logging
from pathlib import Path

import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from biorsp import BioRSPConfig, compute_rsp_radar, compute_vantage, polar_coordinates
from biorsp.core.geometry import get_sector_indices
from biorsp.preprocess.foreground import define_foreground
from biorsp.preprocess.normalization import normalize_radii

logger = logging.getLogger(__name__)

ARCHETYPE_COLORS = {
    "Ubiquitous": "#4DBEEE",
    "Gradient": "#77AC30",
    "Patchy": "#D95319",
    "Basal": "#A2142F",
}


def validate_archetype_quadrants(
    df: pd.DataFrame,
    cutoffs: dict[str, float],
    strict: bool = False,
    outdir: Path | None = None,
) -> list[dict]:
    """
    Validate that archetype labels match their expected quadrants.

    Parameters
    ----------
    df : pd.DataFrame
        Classified gene table
    cutoffs : Dict[str, float]
        Classification cutoffs (c_cut, s_cut, s_cut_method, fdr_cut)
    strict : bool
        If True, raise on mismatches
    outdir : Optional[Path]
        If provided, save mismatches to CSV

    Returns
    -------
    List[Dict]
        List of mismatch records

    Raises
    ------
    ValueError
        If strict=True and mismatches found
    """
    c_cut = cutoffs["c_cut"]
    s_cut = cutoffs["s_cut"]
    s_method = cutoffs.get("s_cut_method", "unknown")
    fdr_cut = cutoffs.get("fdr_cut", 0.05)

    mismatches = []
    for _idx, row in df.iterrows():
        archetype = row["Archetype"]
        c = row["Coverage"]
        s = row["Spatial_Bias_Score"]

        high_c = c >= c_cut
        high_s = row.get("q_value", 1.0) < fdr_cut and s > 0 if s_method == "fdr" else s >= s_cut

        if high_c and high_s:
            expected = "Gradient"
        elif high_c and not high_s:
            expected = "Ubiquitous"
        elif not high_c and high_s:
            expected = "Patchy"
        else:
            expected = "Basal"

        if archetype != expected:
            mismatches.append(
                {
                    "gene": row["gene"],
                    "gene_symbol": row.get("gene_symbol", row["gene"]),
                    "Archetype": archetype,
                    "expected": expected,
                    "Coverage": c,
                    "Spatial_Bias_Score": s,
                    "q_value": row.get("q_value", np.nan),
                    "p_value": row.get("p_value", np.nan),
                }
            )

    if mismatches:
        msg = f"Found {len(mismatches)}/{len(df)} archetype-quadrant mismatches"
        if outdir:
            mismatch_file = outdir / "quadrant_mismatch.csv"
            pd.DataFrame(mismatches).to_csv(mismatch_file, index=False)
            msg += f". Saved to {mismatch_file}"

        if strict:
            raise ValueError(msg)
        else:
            logger.warning(msg)

    return mismatches


def plot_cs_scatter(
    df: pd.DataFrame,
    cutoffs: dict[str, float],
    outdir: Path,
    strict: bool = False,
    max_points: int = 5000,
    seed: int = 42,
):
    """
    Create C-S scatter plot with archetype validation.

    Validates that every gene's archetype matches its quadrant position
    using the exact same cutoffs and logic as classification.
    """
    try:
        c_cut = cutoffs["c_cut"]
        s_cut = cutoffs["s_cut"]
        s_method = cutoffs.get("s_cut_method", "unknown")

        mismatches = validate_archetype_quadrants(df, cutoffs, strict=strict, outdir=outdir)

        fig, ax = plt.subplots(figsize=(10, 10))

        if len(df) > max_points:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(df), size=max_points, replace=False)
            plot_df = df.iloc[idx].copy()
        else:
            plot_df = df.copy()

        for archetype, color in ARCHETYPE_COLORS.items():
            mask = plot_df["Archetype"] == archetype
            if not mask.any():
                continue
            ax.scatter(
                plot_df.loc[mask, "Coverage"],
                plot_df.loc[mask, "Spatial_Bias_Score"],
                c=color,
                s=15,
                alpha=0.6,
                label=f"{archetype} (n={mask.sum()})",
                rasterized=True,
            )

        ax.axvline(c_cut, color="gray", linestyle="--", linewidth=1.5, alpha=0.7, zorder=1000)
        ax.axhline(s_cut, color="gray", linestyle="--", linewidth=1.5, alpha=0.7, zorder=1000)

        ax.set_xlabel("Coverage Score $C$ (expr $\\geq t_g$)", fontsize=16)
        ax.set_ylabel("Spatial Bias Score $S$", fontsize=16)

        title_parts = [f"Gene Archetypes (n={len(df):,} genes)"]
        title_parts.append(
            f"$c_{{\\mathrm{{cut}}}}$={c_cut:.3f}, $s_{{\\mathrm{{cut}}}}$={s_cut:.3f} ({s_method})"
        )
        ax.set_title("\n".join(title_parts), fontsize=16)

        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=12, frameon=True)

        x_max = min(1.02, plot_df["Coverage"].max() * 1.1)
        y_max = plot_df["Spatial_Bias_Score"].quantile(0.99) * 1.2
        ax.set_xlim(-0.02, x_max)
        ax.set_ylim(-0.05, y_max)

        plt.tight_layout()

        figures_dir = outdir / "figures"
        figures_dir.mkdir(exist_ok=True)
        for ext in ["png", "pdf"]:
            outfile = figures_dir / f"fig_cs_scatter.{ext}"
            plt.savefig(outfile, dpi=300 if ext == "png" else None, bbox_inches="tight")
        plt.close(fig)

        status = f" ({len(mismatches)} mismatches)" if mismatches else " (validated)"
        logger.info(f"  Saved C-S scatter to {figures_dir / 'fig_cs_scatter.png'}{status}")

    except Exception as e:
        if strict:
            raise
        else:
            logger.error(f"Failed to create C-S scatter plot: {e}")
            _log_plot_failure(outdir, "cs_scatter", None, e)


def plot_cs_marginals(
    df: pd.DataFrame, cutoffs: dict[str, float], outdir: Path, strict: bool = False
):
    """Create marginal distribution plots for C and S scores."""
    try:
        c_cut = cutoffs["c_cut"]
        s_cut = cutoffs["s_cut"]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].hist(df["Coverage"], bins=50, color="steelblue", alpha=0.7, edgecolor="black")
        axes[0].axvline(
            c_cut,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"$c_{{\\mathrm{{cut}}}}$={c_cut:.3f}",
        )
        axes[0].set_xlabel("Coverage Score $C$", fontsize=14)
        axes[0].set_ylabel("Count", fontsize=14)
        axes[0].set_title("Coverage Distribution", fontsize=16)
        axes[0].legend(fontsize=12)
        axes[0].grid(axis="y", alpha=0.3)

        axes[1].hist(df["Spatial_Bias_Score"], bins=50, color="coral", alpha=0.7, edgecolor="black")
        axes[1].axvline(
            s_cut,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"$s_{{\\mathrm{{cut}}}}$={s_cut:.3f}",
        )
        axes[1].set_xlabel("Spatial Bias Score $S$", fontsize=14)
        axes[1].set_ylabel("Count", fontsize=14)
        axes[1].set_title("Spatial Score Distribution", fontsize=16)
        axes[1].legend(fontsize=12)
        axes[1].grid(axis="y", alpha=0.3)

        plt.tight_layout()
        figures_dir = outdir / "figures"
        figures_dir.mkdir(exist_ok=True)
        for ext in ["png", "pdf"]:
            outfile = figures_dir / f"fig_cs_marginals.{ext}"
            plt.savefig(outfile, dpi=300 if ext == "png" else None, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"  Saved marginal plots to {figures_dir / 'fig_cs_marginals.png'}")

    except Exception as e:
        if strict:
            raise
        else:
            logger.error(f"Failed to create marginal plots: {e}")
            _log_plot_failure(outdir, "cs_marginals", None, e)


def plot_gene_exemplar(
    adata: anndata.AnnData,
    gene: str,
    gene_row: pd.Series,
    embedding_key: str,
    config: BioRSPConfig,
    outdir: Path,
    var_to_symbol: dict[str, str],
    coverage_threshold: float,
    strict: bool = False,
):
    """
    Plot exemplar for a single gene showing:
    1. Embedding with coverage-positive cells (colored) and internal FG (outlined)
    2. Radar plot with sector support annotations
    3. Gene metadata (C, S, coverage_geom, q_value)

    Parameters
    ----------
    adata : AnnData
        Subset data
    gene : str
        Gene identifier (var_name)
    gene_row : pd.Series
        Row from classified gene table
    embedding_key : str
        Embedding key
    config : BioRSPConfig
        Configuration
    outdir : Path
        Output directory
    var_to_symbol : Dict[str, str]
        Gene symbol mapping
    coverage_threshold : float
        Expression threshold for coverage
    strict : bool
        Raise on errors
    """
    try:
        from biorsp import plot_radar

        gene_symbol = var_to_symbol.get(gene, gene)
        archetype = gene_row["Archetype"]

        idx = adata.var_names.get_loc(gene)
        x = adata.X[:, idx].toarray().flatten() if hasattr(adata.X, "toarray") else adata.X[:, idx]

        coverage_mask = x >= coverage_threshold
        n_coverage = coverage_mask.sum()

        y_fg, fg_info = define_foreground(
            x,
            mode=config.foreground_mode,
            q=config.foreground_quantile,
            abs_threshold=config.foreground_threshold,
            min_fg=config.min_fg_total,
            rng=np.random.default_rng(config.seed),
        )

        coords = adata.obsm[embedding_key]

        fig = plt.figure(figsize=(18, 5))
        ax1 = plt.subplot(1, 3, 1)
        ax2 = plt.subplot(1, 3, 2)
        ax3 = plt.subplot(1, 3, 3)

        ax1.scatter(coords[:, 0], coords[:, 1], c="lightgray", s=1, alpha=0.3, rasterized=True)

        if n_coverage > 0:
            color = ARCHETYPE_COLORS.get(archetype, "purple")
            ax1.scatter(
                coords[coverage_mask, 0],
                coords[coverage_mask, 1],
                c=color,
                s=10,
                alpha=0.6,
                label=f"Coverage+ (n={n_coverage})",
                rasterized=True,
            )

        if y_fg is not None and np.any(y_fg):
            fg_mask = y_fg if y_fg.dtype == bool else y_fg > 0.5
            ax1.scatter(
                coords[fg_mask, 0],
                coords[fg_mask, 1],
                facecolors="none",
                edgecolors="black",
                s=15,
                linewidths=0.8,
                alpha=0.8,
                label=f"Internal FG (n={fg_mask.sum()})",
                rasterized=True,
            )

        ax1.set_xlabel("UMAP 1", fontsize=12)
        ax1.set_ylabel("UMAP 2", fontsize=12)
        ax1.set_title(f"{gene_symbol} Expression\n{archetype}", fontsize=14, fontweight="bold")
        ax1.legend(loc="best", fontsize=10)
        ax1.set_aspect("equal")

        if y_fg is not None:
            center = compute_vantage(
                coords,
                method=config.vantage,
                knn_k=config.center_knn_k,
                density_percentile=config.center_density_percentile,
                tol=config.geom_median_tol,
                max_iter=config.geom_median_max_iter,
                seed=config.seed,
            )
            r, theta = polar_coordinates(coords, center)
            r_norm, _ = normalize_radii(r)
            sector_indices = get_sector_indices(theta, config.B, config.delta_deg)

            radar = compute_rsp_radar(
                r_norm, theta, y_fg, config=config, sector_indices=sector_indices
            )

            plot_radar(
                radar,
                ax=ax2,
                center_offset=0.2,
                show_bg_support=True,
                show_zero_fill=True,
                title=f"RSP Radar\n(B={config.B}, δ={config.delta_deg}°)",
            )
        else:
            ax2.text(
                0.5, 0.5, "Insufficient FG", ha="center", va="center", fontsize=14, color="red"
            )
            ax2.axis("off")

        ax3.axis("off")
        metadata_text = [
            f"Gene: {gene_symbol}",
            f"Archetype: {archetype}",
            "",
            f"Coverage $C$: {gene_row['Coverage']:.3f}",
            f"Spatial $S$: {gene_row['Spatial_Score']:.3f}",
            f"Coverage Geom: {gene_row.get('coverage_geom', 0):.3f}",
            f"$p$-value: {gene_row.get('p_value', np.nan):.2e}",
            f"$q$-value: {gene_row.get('q_value', np.nan):.2e}",
            "",
            f"Coverage threshold: {coverage_threshold:.3f}",
            f"FG mode: {fg_info.get('mode', 'N/A') if y_fg else 'N/A'}",
            f"FG cells: {fg_info.get('n_fg', 0) if y_fg else 0}",
        ]
        ax3.text(0.1, 0.9, "\n".join(metadata_text), fontsize=11, va="top", family="monospace")

        plt.tight_layout()

        exemplar_dir = outdir / "exemplars"
        exemplar_dir.mkdir(exist_ok=True)

        safe_name = gene_symbol.replace("/", "_").replace(":", "_")
        outfile = exemplar_dir / f"exemplar_{safe_name}.png"
        plt.savefig(outfile, dpi=200, bbox_inches="tight")
        plt.close(fig)

    except Exception as e:
        if strict:
            raise
        else:
            logger.warning(f"Failed to create exemplar for {gene}: {e}")
            _log_plot_failure(outdir, "gene_exemplar", gene, e)


def plot_debug_pointcloud(
    coords: np.ndarray,
    x: np.ndarray,
    coverage_mask: np.ndarray,
    fg_mask: np.ndarray | None,
    gene: str,
    outdir: Path,
):
    """Debug plot showing coverage vs internal FG masks."""
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(coords[:, 0], coords[:, 1], c="lightgray", s=1, alpha=0.2, label="Background")
    ax.scatter(
        coords[coverage_mask, 0],
        coords[coverage_mask, 1],
        c="blue",
        s=5,
        alpha=0.5,
        label=f"Coverage+ (n={coverage_mask.sum()})",
    )

    if fg_mask is not None and np.any(fg_mask):
        ax.scatter(
            coords[fg_mask, 0],
            coords[fg_mask, 1],
            facecolors="none",
            edgecolors="red",
            s=10,
            linewidths=1,
            label=f"Internal FG (n={fg_mask.sum()})",
        )

    ax.set_title(f"DEBUG: {gene} Masks", fontsize=14, fontweight="bold")
    ax.legend()
    ax.set_aspect("equal")

    debug_dir = outdir / "debug"
    debug_dir.mkdir(exist_ok=True)
    plt.savefig(debug_dir / f"debug_pointcloud_{gene}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _log_plot_failure(outdir: Path, stage: str, gene: str | None, exception: Exception):
    """Log plotting failure to CSV."""
    failure_file = outdir / "plot_failures.csv"
    pd.DataFrame(
        [
            {
                "stage": stage,
                "gene": gene if gene else "",
                "exception": str(exception),
            }
        ]
    ).to_csv(failure_file, mode="a", header=not failure_file.exists(), index=False)
