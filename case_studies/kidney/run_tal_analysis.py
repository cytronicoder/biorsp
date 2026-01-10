#!/usr/bin/env python3
"""
Analyze how genes are spatially organized in Thick Ascending Limb (TAL) cells
from the human kidney reference dataset.

Key Outputs:
- Coverage Score (C_g): What fraction of TAL cells express this gene?
- Spatial Score (S_g): Does expression cluster in specific regions, or spread uniformly?
- Archetype: Classification into spatial patterns (localized, housekeeping, niche, sparse)
- Gene-Gene Relationships: Which genes show similar spatial patterns?

Workflow:
1. Load your reference dataset
2. Extract TAL cells (with optional subsampling for quick testing)
3. Select genes to analyze (canonical markers + discovery candidates)
4. Score each gene for coverage and spatial organization
5. Classify genes into spatial archetypes
6. Optionally compute pairwise gene relationships
7. Generate publication-ready figures and data tables

Quick Examples:
    # Fast test run (2 minutes)
    python run_tal_analysis.py \\
      --ref_data data/kpmp.h5ad \\
      --outdir results/tal_pilot \\
      --controls "SLC12A1,UMOD,EGF" \\
      --max_genes 10 \\
      --n_permutations 100

    # Full analysis (30-60 minutes)
    python run_tal_analysis.py \\
      --ref_data data/kpmp.h5ad \\
      --outdir results/tal_full \\
      --controls "SLC12A1,UMOD,EGF" \\
      --max_genes 500 \\
      --n_permutations 1000 \\
      --do_genegene \\
      --n_workers 4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import warnings
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse
from tqdm import tqdm

# Prevent BLAS thread oversubscription when using multiple workers
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from biorsp import (
    BioRSPConfig,
    classify_genes,
    compute_rsp_radar,
    plot_radar,
    polar_coordinates,
    score_gene_pairs,
    score_genes,
)
from biorsp.preprocess.geometry import compute_vantage

warnings.filterwarnings("ignore", message=".*dtype argument is deprecated.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="legacy_api_wrap")

try:
    import anndata
    import scanpy as sc
except ImportError:
    anndata = None
    sc = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# CLI Argument Parsing
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the TAL case study."""
    parser = argparse.ArgumentParser(
        description="BioRSP TAL Case Study: Spatial gene scoring in human kidney",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Pilot run (fast, for testing)
  python run_tal_analysis.py --ref_data data/kpmp.h5ad --outdir results/pilot \\
      --controls "SLC12A1,UMOD,EGF" --max_genes 10 --n_permutations 100

  # Full run with gene-gene analysis
  python run_tal_analysis.py --ref_data data/kpmp.h5ad --outdir results/full \\
      --controls "SLC12A1,UMOD,EGF" --max_genes 500 --n_permutations 1000 \\
      --do_genegene --n_workers 4
""",
    )

    # Input/Output
    io_group = parser.add_argument_group("Input/Output")
    io_group.add_argument(
        "--ref_data",
        type=str,
        required=True,
        help="Path to reference data (.h5ad preferred)",
    )
    io_group.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="Directory to save all results",
    )

    # Cell selection
    cell_group = parser.add_argument_group("Cell Selection")
    cell_group.add_argument(
        "--celltype_key",
        type=str,
        default="subclass.l1",
        help="Metadata column for cell type annotations (default: subclass.l1)",
    )
    cell_group.add_argument(
        "--tal_labels",
        type=str,
        nargs="+",
        default=["TAL"],
        help="Label(s) identifying TAL cells (default: TAL)",
    )
    cell_group.add_argument(
        "--donor_key",
        type=str,
        default="donor_id",
        help="Metadata column for donor identifiers (default: donor_id)",
    )
    cell_group.add_argument(
        "--subsample",
        type=int,
        default=None,
        help="Subsample to N cells for faster testing (default: no subsampling)",
    )

    # Gene selection
    gene_group = parser.add_argument_group("Gene Selection")
    gene_group.add_argument(
        "--controls",
        type=str,
        help="Comma-separated canonical TAL markers (e.g., 'SLC12A1,UMOD,EGF')",
    )
    gene_group.add_argument(
        "--min_pct",
        type=float,
        default=0.01,
        help="Minimum expression prevalence for discovery genes (default: 0.01)",
    )
    gene_group.add_argument(
        "--max_genes",
        type=int,
        default=None,
        help="Maximum discovery genes to analyze (default: all passing filters)",
    )
    gene_group.add_argument(
        "--exclude_patterns",
        type=str,
        default="^MT-|^mt-|^RPS|^RPL",
        help="Regex pattern for genes to exclude (default: MT/ribosomal)",
    )

    # Scoring parameters
    scoring_group = parser.add_argument_group("Scoring Parameters")
    scoring_group.add_argument(
        "--embedding_key",
        type=str,
        default=None,
        help="Key in adata.obsm for embedding (default: auto-detect X_umap)",
    )
    scoring_group.add_argument(
        "--B",
        type=int,
        default=72,
        help="Number of angular sectors (default: 72 = 5° resolution)",
    )
    scoring_group.add_argument(
        "--delta_deg",
        type=float,
        default=60.0,
        help="Sector width in degrees (default: 60°; use 180 for half-plane contrast)",
    )
    scoring_group.add_argument(
        "--foreground_quantile",
        type=float,
        default=0.90,
        help="Quantile for internal foreground selection (default: 0.90)",
    )
    scoring_group.add_argument(
        "--expr_threshold_mode",
        type=str,
        choices=["detect", "fixed", "nonzero_quantile"],
        default="detect",
        help="How to determine coverage threshold (default: detect)",
    )
    scoring_group.add_argument(
        "--expr_threshold_value",
        type=float,
        default=None,
        help="Fixed coverage threshold (only used if mode=fixed)",
    )
    scoring_group.add_argument(
        "--empty_fg_policy",
        type=str,
        choices=["nan", "zero"],
        default="zero",
        help="Policy for empty-foreground sectors (default: zero = no signal)",
    )
    scoring_group.add_argument(
        "--n_permutations",
        type=int,
        default=200,
        help="Number of permutations for p-value calculation (default: 200)",
    )

    # Classification parameters
    class_group = parser.add_argument_group("Archetype Classification")
    class_group.add_argument(
        "--c_cut",
        type=float,
        default=0.10,
        help="Coverage cutoff for archetype classification (default: 0.10)",
    )
    class_group.add_argument(
        "--s_cut",
        type=float,
        default=None,
        help="Spatial score cutoff (default: auto from FDR or empirical null)",
    )
    class_group.add_argument(
        "--fdr_cut",
        type=float,
        default=0.05,
        help="FDR threshold for significance (default: 0.05)",
    )

    # Gene-gene analysis
    pair_group = parser.add_argument_group("Gene-Gene Analysis (Optional)")
    pair_group.add_argument(
        "--do_genegene",
        action="store_true",
        help="Compute pairwise gene relationships",
    )
    pair_group.add_argument(
        "--top_for_pairs",
        type=int,
        default=50,
        help="Number of top genes for pairwise analysis (default: 50)",
    )

    # Plotting
    plot_group = parser.add_argument_group("Plotting")
    plot_group.add_argument(
        "--top_k_plots",
        type=int,
        default=12,
        help="Number of top genes to generate detailed plots for (default: 12)",
    )
    plot_group.add_argument(
        "--plot_mode",
        type=str,
        choices=["signed", "absolute"],
        default="signed",
        help="Radar plot mode: 'signed' (proximal/distal color) or 'absolute' (split panels)",
    )

    # Runtime
    runtime_group = parser.add_argument_group("Runtime")
    runtime_group.add_argument(
        "--n_workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1 = serial)",
    )
    runtime_group.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    runtime_group.add_argument(
        "--log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging verbosity (default: INFO)",
    )

    return parser.parse_args()


# =============================================================================
# Data Loading and Utilities
# =============================================================================


def compute_file_checksum(path: str, algorithm: str = "sha256") -> str:
    """Compute checksum of a file for provenance tracking."""
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_reference(path: str) -> anndata.AnnData:
    """Load reference data from H5AD file."""
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Reference file not found: {path}")

    if path_obj.suffix.lower() != ".h5ad":
        raise ValueError(
            f"Only .h5ad files are supported. Got: {path_obj.suffix}\n"
            "Convert using: scripts/convert_to_h5ad.R or scripts/build_h5ad_from_export.py"
        )

    if anndata is None:
        raise ImportError("anndata is required: pip install anndata scanpy")

    logger.info(f"Loading H5AD from {path}...")
    adata = anndata.read_h5ad(path)
    logger.info(f"Loaded: {adata.n_obs} cells × {adata.n_vars} genes")
    return adata


def detect_embedding_key(adata: anndata.AnnData, preferred_key: str | None = None) -> str:
    """Auto-detect embedding key from adata.obsm."""
    if preferred_key is not None:
        if preferred_key in adata.obsm:
            return preferred_key
        raise ValueError(f"Embedding key '{preferred_key}' not found in adata.obsm")
    umap_candidates = ["X_umap", "X_UMAP", "umap"]
    for key in umap_candidates:
        if key in adata.obsm:
            logger.info(f"Auto-detected embedding: {key}")
            return key
    for key in adata.obsm:
        if "umap" in key.lower():
            logger.info(f"Auto-detected embedding: {key}")
            return key
    for key in adata.obsm:
        emb = adata.obsm[key]
        if hasattr(emb, "shape") and len(emb.shape) == 2 and emb.shape[1] >= 2:
            logger.warning(f"Using fallback embedding: {key}")
            return key

    raise ValueError(
        f"No suitable embedding found. Available keys: {list(adata.obsm.keys())}\n"
        "Hint: Provide --embedding_key explicitly or ensure X_umap exists."
    )


def build_symbol_mappings(adata: anndata.AnnData) -> tuple[dict[str, str], dict[str, str]]:
    """Build var_name <-> gene_symbol mappings."""
    var_to_symbol = {}
    symbol_cols = ["feature_name", "gene_symbol", "gene_name", "symbol"]
    for col in symbol_cols:
        if col in adata.var.columns:
            var_to_symbol = adata.var[col].to_dict()
            logger.info(f"Using '{col}' column for gene symbols")
            break

    if not var_to_symbol:
        var_to_symbol = {name: name for name in adata.var_names}
        logger.info("Using var_names as gene symbols (no symbol column found)")

    symbol_to_var = {v: k for k, v in var_to_symbol.items()}
    return var_to_symbol, symbol_to_var


# =============================================================================
# Gene Selection
# =============================================================================


def select_genes(
    adata: anndata.AnnData,
    controls_str: str | None,
    symbol_to_var: dict[str, str],
    var_to_symbol: dict[str, str],
    min_pct: float,
    max_genes: int | None,
    exclude_pattern: str,
    seed: int,
) -> tuple[list[str], list[str], dict]:
    """Select genes for analysis: controls + discovery set.

    Returns
    -------
    genes_to_analyze : List[str]
        All genes (var_names) to score
    control_vars : List[str]
        Control gene var_names (subset of genes_to_analyze)
    selection_info : Dict
        Metadata about selection for provenance
    """
    all_genes = adata.var_names.tolist()
    n_cells = adata.n_obs

    selection_info = {
        "n_genes_total": len(all_genes),
        "min_pct": min_pct,
        "max_genes": max_genes,
        "exclude_pattern": exclude_pattern,
    }

    # Parse controls
    control_vars = []
    control_symbols = []
    missing_controls = []
    if controls_str:
        for c in controls_str.split(","):
            c = c.strip()
            if c in symbol_to_var:
                v = symbol_to_var[c]
                if v in all_genes:
                    control_vars.append(v)
                    control_symbols.append(c)
                else:
                    missing_controls.append(c)
            elif c in all_genes:
                control_vars.append(c)
                control_symbols.append(var_to_symbol.get(c, c))
            else:
                missing_controls.append(c)

        if missing_controls:
            logger.warning(f"Control genes not found (skipped): {missing_controls}")

        selection_info["controls_requested"] = controls_str.split(",")
        selection_info["controls_found"] = control_symbols
        selection_info["controls_missing"] = missing_controls
        logger.info(f"Controls: {len(control_vars)} found, {len(missing_controls)} missing")
    else:
        selection_info["controls_requested"] = []
        selection_info["controls_found"] = []

    logger.info("Computing gene prevalence...")
    X = adata.X
    if scipy.sparse.issparse(X):
        nonzero_frac = np.asarray((X > 0).sum(axis=0)).flatten() / n_cells
    else:
        nonzero_frac = np.asarray((X > 0).sum(axis=0)).flatten() / n_cells
    prevalence_mask = nonzero_frac >= min_pct
    exclude_re = re.compile(exclude_pattern) if exclude_pattern else None
    if exclude_re:
        exclude_by_varname = np.array([bool(exclude_re.match(g)) for g in all_genes])
        exclude_by_symbol = np.array(
            [bool(exclude_re.match(var_to_symbol.get(g, g))) for g in all_genes]
        )
        exclude_mask = exclude_by_varname | exclude_by_symbol
        prevalence_mask = prevalence_mask & ~exclude_mask
        n_excluded = int(exclude_mask.sum())
        logger.info(
            f"Excluded {n_excluded} genes matching '{exclude_pattern}' (var_name or symbol)"
        )
    discovery_genes = [g for g, m in zip(all_genes, prevalence_mask) if m and g not in control_vars]
    logger.info(f"Discovery candidates: {len(discovery_genes)} genes with prevalence >= {min_pct}")
    if max_genes and len(discovery_genes) > max_genes:
        disc_prev = {g: nonzero_frac[all_genes.index(g)] for g in discovery_genes}
        discovery_genes = sorted(discovery_genes, key=lambda g: -disc_prev[g])[:max_genes]
        logger.info(f"Subsampled to top {max_genes} discovery genes by prevalence")

    genes_to_analyze = control_vars + discovery_genes
    selection_info["n_controls"] = len(control_vars)
    selection_info["n_discovery"] = len(discovery_genes)
    selection_info["n_total"] = len(genes_to_analyze)

    logger.info(
        f"Gene selection complete: {len(control_vars)} controls + {len(discovery_genes)} discovery = {len(genes_to_analyze)} total"
    )

    return genes_to_analyze, control_vars, selection_info


# =============================================================================
# Plotting Functions (v3 compatible)
# =============================================================================


def plot_gene_v3(
    gene: str,
    adata: anndata.AnnData,
    gene_result: pd.Series,
    embedding_key: str,
    config: BioRSPConfig,
    var_to_symbol: dict[str, str],
    outdir: Path,
    plot_mode: str = "signed",
):
    """Create a two-panel visualization for a single gene.

    Left panel: Cell embedding colored by expression level, with a boundary
                showing which cells actually express the gene (above threshold)
    Right panel: Radar plot showing radial distribution of expression
    Annotation box: Key metrics (coverage, spatial score, significance)
    """
    symbol = var_to_symbol.get(gene, gene)

    # Get expression data
    X = adata[:, gene].X
    expr = X.toarray().flatten() if scipy.sparse.issparse(X) else np.asarray(X).flatten()

    # Get embedding
    coords = adata.obsm[embedding_key][:, :2]

    # Coverage mask (biological threshold)
    t_g = gene_result.get("expr_threshold_value", 1.0)
    coverage_mask = expr >= t_g

    # Internal foreground (quantile-based for radar)
    q_thresh = np.quantile(expr, config.foreground_quantile)
    fg_mask = expr >= q_thresh if q_thresh > 0 else expr > 0

    # Compute vantage and polar coords using package functions
    center = compute_vantage(coords, method=config.vantage, seed=config.seed)
    r, theta = polar_coordinates(coords, center)

    # Compute radar for plotting
    radar = compute_rsp_radar(r, theta, fg_mask.astype(float), config=config)

    # Create figure
    if plot_mode == "absolute":
        fig = plt.figure(figsize=(18, 6))
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1])
        ax_emb = fig.add_subplot(gs[0, 0])
        ax_radar1 = fig.add_subplot(gs[0, 1], projection="polar")
        ax_radar2 = fig.add_subplot(gs[0, 2], projection="polar")
    else:
        fig = plt.figure(figsize=(14, 6))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1])
        ax_emb = fig.add_subplot(gs[0, 0])
        ax_radar = fig.add_subplot(gs[0, 1], projection="polar")

    # --- Embedding panel ---
    # Plot all cells colored by expression
    sc = ax_emb.scatter(
        coords[:, 0],
        coords[:, 1],
        c=expr,
        s=5,
        alpha=0.7,
        cmap="viridis",
        edgecolors="none",
    )
    plt.colorbar(sc, ax=ax_emb, label="Expression", shrink=0.6)

    # Outline coverage-positive cells
    if np.any(coverage_mask):
        ax_emb.scatter(
            coords[coverage_mask, 0],
            coords[coverage_mask, 1],
            s=15,
            facecolors="none",
            edgecolors="red",
            linewidths=0.5,
            alpha=0.6,
            label=f"Coverage (x ≥ {t_g:.2g})",
        )

    # Mark vantage point
    ax_emb.scatter(center[0], center[1], c="black", marker="X", s=150, zorder=10, label="Vantage")

    ax_emb.set_title(f"{symbol}\n(TAL Cells)", fontsize=14)
    ax_emb.axis("off")
    ax_emb.legend(loc="lower left", fontsize=9)

    # --- Radar panel(s) ---
    if plot_mode == "absolute":
        plot_radar(radar, ax=ax_radar1, title="Proximal ($R > 0$)", mode="proximal")
        plot_radar(radar, ax=ax_radar2, title="Distal ($R < 0$)", mode="distal")
    else:
        plot_radar(radar, ax=ax_radar, title=f"{symbol} RSP", mode="signed")

    # --- Annotation box with key metrics ---
    c_g = gene_result.get("coverage_expr", np.nan)
    s_g = gene_result.get("spatial_score", np.nan)
    sign = gene_result.get("spatial_sign", 0)
    r_mean = gene_result.get("r_mean_bg", np.nan)
    cov_bg = gene_result.get("coverage_bg", np.nan)
    p_val = gene_result.get("p_value", np.nan)
    q_val = gene_result.get("q_value", np.nan)

    sign_str = "+" if sign > 0 else ("-" if sign < 0 else "0")

    def _fmt(x, fmt="{:.3f}"):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "—"
        return fmt.format(x)

    info_lines = [
        f"$C_g$ (coverage): {_fmt(c_g)}",
        f"$S_g$ (spatial): {_fmt(s_g)}",
        f"Sign: {sign_str}",
        f"$\\bar{{R}}_{{bg}}$: {_fmt(r_mean)}",
        f"Coverage$_{{bg}}$: {_fmt(cov_bg)}",
        f"p-value: {_fmt(p_val, '{:.2e}')}",
        f"q-value: {_fmt(q_val, '{:.2e}')}",
    ]

    fig.text(
        0.98,
        0.95,
        "\n".join(info_lines),
        transform=fig.transFigure,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
        family="monospace",
    )

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    outdir.mkdir(parents=True, exist_ok=True)
    plt.savefig(outdir / f"plot_{symbol}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_archetype_scatter(
    df: pd.DataFrame,
    control_symbols: list[str],
    c_cut: float,
    s_cut: float | None,
    outpath: Path,
):
    """Create C_g vs S_g scatter plot with archetype quadrants.

    This is the primary "story figure" for biologists.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Get cutoffs
    if s_cut is None:
        s_cut = df.attrs.get("s_cut", 0.05)  # Default from classify_genes

    # Plot all genes
    if "archetype" in df.columns:
        archetype_colors = {
            "localized_program": "#e41a1c",
            "housekeeping_uniform": "#377eb8",
            "niche_biomarker": "#4daf4a",
            "sparse_presence": "#999999",
        }
        for arch, color in archetype_colors.items():
            mask = df["archetype"] == arch
            ax.scatter(
                df.loc[mask, "coverage_expr"],
                df.loc[mask, "spatial_score"],
                c=color,
                label=arch.replace("_", " ").title(),
                s=30,
                alpha=0.7,
                edgecolors="none",
            )
    else:
        ax.scatter(
            df["coverage_expr"],
            df["spatial_score"],
            c="steelblue",
            s=30,
            alpha=0.7,
            edgecolors="none",
        )

    # Draw quadrant lines
    ax.axvline(c_cut, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.axhline(s_cut, color="gray", linestyle="--", alpha=0.5, linewidth=1)

    # Label control genes
    symbol_col = "gene_symbol" if "gene_symbol" in df.columns else "gene"
    for _, row in df.iterrows():
        sym = row.get(symbol_col, row["gene"])
        if sym in control_symbols:
            ax.annotate(
                sym,
                (row["coverage_expr"], row["spatial_score"]),
                fontsize=9,
                fontweight="bold",
                xytext=(5, 5),
                textcoords="offset points",
                color="red",
            )

    # Quadrant labels
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.text(
        xlim[0] + 0.02 * (xlim[1] - xlim[0]),
        ylim[1] - 0.05 * (ylim[1] - ylim[0]),
        "Niche Biomarker\n(low C, high S)",
        fontsize=9,
        fontstyle="italic",
        alpha=0.7,
        ha="left",
        va="top",
    )
    ax.text(
        xlim[1] - 0.02 * (xlim[1] - xlim[0]),
        ylim[1] - 0.05 * (ylim[1] - ylim[0]),
        "Localized Program\n(high C, high S)",
        fontsize=9,
        fontstyle="italic",
        alpha=0.7,
        ha="right",
        va="top",
    )
    ax.text(
        xlim[0] + 0.02 * (xlim[1] - xlim[0]),
        ylim[0] + 0.05 * (ylim[1] - ylim[0]),
        "Sparse Presence\n(low C, low S)",
        fontsize=9,
        fontstyle="italic",
        alpha=0.7,
        ha="left",
        va="bottom",
    )
    ax.text(
        xlim[1] - 0.02 * (xlim[1] - xlim[0]),
        ylim[0] + 0.05 * (ylim[1] - ylim[0]),
        "Housekeeping/Uniform\n(high C, low S)",
        fontsize=9,
        fontstyle="italic",
        alpha=0.7,
        ha="right",
        va="bottom",
    )

    ax.set_xlabel("Coverage ($C_g$): Fraction of cells expressing", fontsize=12)
    ax.set_ylabel("Spatial Score ($S_g$): Radial organization magnitude", fontsize=12)
    ax.set_title("Gene Archetype Landscape\n(TAL Cells)", fontsize=14)

    if "archetype" in df.columns:
        ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved archetype scatter to {outpath}")


def plot_gene_pairs_heatmap(
    pairs_df: pd.DataFrame,
    var_to_symbol: dict[str, str],
    outpath: Path,
    top_n: int = 30,
):
    """Create a heatmap of gene-gene copattern scores."""
    if pairs_df.empty:
        logger.warning("No gene pairs to plot")
        return

    # Get top pairs by absolute copattern score
    pairs_df = pairs_df.copy()
    pairs_df["abs_copattern"] = pairs_df["copattern_score"].abs()
    top_pairs = pairs_df.nlargest(top_n, "abs_copattern")

    # Get unique genes
    genes_a = top_pairs["gene_a"].unique()
    genes_b = top_pairs["gene_b"].unique()
    all_genes = sorted(set(genes_a) | set(genes_b))

    if len(all_genes) < 2:
        logger.warning("Not enough genes for heatmap")
        return

    # Build matrix
    n = len(all_genes)
    gene_idx = {g: i for i, g in enumerate(all_genes)}
    matrix = np.zeros((n, n))
    np.fill_diagonal(matrix, 1.0)

    for _, row in top_pairs.iterrows():
        i = gene_idx.get(row["gene_a"])
        j = gene_idx.get(row["gene_b"])
        if i is not None and j is not None:
            matrix[i, j] = row["copattern_score"]
            matrix[j, i] = row["copattern_score"]

    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    symbols = [var_to_symbol.get(g, g) for g in all_genes]

    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(symbols, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(symbols, fontsize=8)
    ax.set_title("Gene-Gene Copattern Scores\n(Top pairs)", fontsize=14)

    plt.colorbar(im, ax=ax, label="Copattern Score", shrink=0.8)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved gene pairs heatmap to {outpath}")


# =============================================================================
# Main Pipeline
# =============================================================================


def main():
    args = parse_args()

    # Setup logging
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper(), logging.INFO))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("BioRSP Case Study: TAL Cells (v3 API)")
    logger.info("=" * 60)

    # Initialize run metadata
    run_meta = {
        "timestamp": datetime.now().isoformat(),
        "args": vars(args),
        "biorsp_version": "3.0",  # Placeholder - could import from package
    }

    # -------------------------------------------------------------------------
    # Stage 1: Load Data
    # -------------------------------------------------------------------------
    logger.info("[Stage 1] Loading reference dataset...")
    adata = load_reference(args.ref_data)

    # Compute checksum for provenance
    try:
        run_meta["input_checksum"] = {
            "file": args.ref_data,
            "sha256": compute_file_checksum(args.ref_data),
        }
    except Exception as e:
        logger.warning(f"Could not compute checksum: {e}")
        run_meta["input_checksum"] = None

    run_meta["n_cells_total"] = int(adata.n_obs)
    run_meta["n_genes_total"] = int(adata.n_vars)

    # Build symbol mappings
    var_to_symbol, symbol_to_var = build_symbol_mappings(adata)

    # -------------------------------------------------------------------------
    # Stage 2: Subset to TAL cells
    # -------------------------------------------------------------------------
    logger.info("[Stage 2] Subsetting to TAL cells...")

    if args.celltype_key not in adata.obs.columns:
        available = list(adata.obs.columns)
        raise ValueError(
            f"Cell type column '{args.celltype_key}' not found.\n" f"Available columns: {available}"
        )

    tal_mask = adata.obs[args.celltype_key].isin(args.tal_labels)
    n_tal = tal_mask.sum()
    logger.info(f"Found {n_tal} TAL cells (labels: {args.tal_labels})")

    if n_tal == 0:
        unique_labels = adata.obs[args.celltype_key].unique().tolist()
        raise ValueError(
            f"No cells match labels {args.tal_labels}.\n"
            f"Available labels in '{args.celltype_key}': {unique_labels}"
        )

    adata_tal = adata[tal_mask].copy()

    # Subsample if requested
    if args.subsample and args.subsample < adata_tal.n_obs:
        logger.info(f"Subsampling to {args.subsample} cells (seed={args.seed})")
        np.random.seed(args.seed)
        idx = np.random.choice(adata_tal.n_obs, args.subsample, replace=False)
        adata_tal = adata_tal[idx].copy()

    # Record donor distribution
    if args.donor_key in adata_tal.obs.columns:
        donor_counts = adata_tal.obs[args.donor_key].value_counts().to_dict()
        run_meta["donor_distribution"] = donor_counts
        logger.info(f"Donor distribution: {len(donor_counts)} donors")
    else:
        run_meta["donor_distribution"] = None
        logger.warning(f"Donor key '{args.donor_key}' not found in metadata")

    run_meta["n_tal_cells"] = int(adata_tal.n_obs)
    logger.info(f"Analysis subset: {adata_tal.n_obs} TAL cells")

    if adata_tal.n_obs < 100:
        logger.warning("Very few cells (<100). Results may be unstable.")

    # -------------------------------------------------------------------------
    # Stage 3: Detect embedding
    # -------------------------------------------------------------------------
    logger.info("[Stage 3] Detecting embedding...")
    embedding_key = detect_embedding_key(adata_tal, args.embedding_key)
    run_meta["embedding_key"] = embedding_key

    # -------------------------------------------------------------------------
    # Stage 4: Gene selection
    # -------------------------------------------------------------------------
    logger.info("[Stage 4] Selecting genes for analysis...")
    genes_to_analyze, control_vars, selection_info = select_genes(
        adata_tal,
        args.controls,
        symbol_to_var,
        var_to_symbol,
        args.min_pct,
        args.max_genes,
        args.exclude_patterns,
        args.seed,
    )
    run_meta["gene_selection"] = selection_info
    control_symbols = [var_to_symbol.get(v, v) for v in control_vars]

    if len(genes_to_analyze) == 0:
        raise ValueError("No genes selected for analysis. Check --min_pct and --exclude_patterns.")

    # -------------------------------------------------------------------------
    # Stage 5: Run BioRSP scoring (v3 API)
    # -------------------------------------------------------------------------
    logger.info("[Stage 5] Running BioRSP scoring...")

    # Determine stratify key (convert categorical to string to avoid numpy dtype issues)
    stratify_key = None
    if args.donor_key in adata_tal.obs.columns:
        donor_col = adata_tal.obs[args.donor_key]
        # Convert categorical to string for stratification
        if hasattr(donor_col, "cat"):
            adata_tal.obs[args.donor_key] = donor_col.astype(str)
        stratify_key = args.donor_key

    # Build config
    config = BioRSPConfig(
        B=args.B,
        delta_deg=args.delta_deg,
        foreground_quantile=args.foreground_quantile,
        expr_threshold_mode=args.expr_threshold_mode,
        expr_threshold_value=args.expr_threshold_value,
        empty_fg_policy=args.empty_fg_policy,
        n_permutations=args.n_permutations,
        seed=args.seed,
        stratify_key=stratify_key,
    )

    run_meta["config"] = asdict(config)

    logger.info(f"Config: B={config.B}, delta={config.delta_deg}°, q={config.foreground_quantile}")
    logger.info(f"  threshold_mode={config.expr_threshold_mode}, empty_fg={config.empty_fg_policy}")
    logger.info(f"  n_permutations={config.n_permutations}, seed={config.seed}")

    # Call score_genes (v3 API)
    df_results = score_genes(
        adata_tal,
        genes=genes_to_analyze,
        embedding_key=embedding_key,
        config=config,
    )

    # Add gene symbols
    df_results["gene_symbol"] = df_results["gene"].map(var_to_symbol)
    df_results["is_control"] = df_results["gene"].isin(control_vars)

    logger.info(f"Scored {len(df_results)} genes")
    run_meta["n_genes_scored"] = len(df_results)

    # -------------------------------------------------------------------------
    # Stage 6: Classify genes into archetypes
    # -------------------------------------------------------------------------
    logger.info("[Stage 6] Classifying genes into archetypes...")
    df_classified = classify_genes(
        df_results,
        c_cut=args.c_cut,
        s_cut=args.s_cut,
        fdr_cut=args.fdr_cut,
    )

    # Record classification cutoffs
    run_meta["classification"] = {
        "c_cut": df_classified.attrs.get("c_cut", args.c_cut),
        "s_cut": df_classified.attrs.get("s_cut"),
        "s_cut_method": df_classified.attrs.get("s_cut_method", "manual"),
        "fdr_cut": args.fdr_cut,
    }

    archetype_counts = df_classified["archetype"].value_counts().to_dict()
    logger.info(f"Archetype distribution: {archetype_counts}")
    run_meta["archetype_counts"] = archetype_counts

    # -------------------------------------------------------------------------
    # Stage 7: Save results
    # -------------------------------------------------------------------------
    logger.info("[Stage 7] Saving results...")

    # Sort by spatial_score (primary), coverage_expr (secondary)
    df_sorted = df_classified.sort_values(
        ["spatial_score", "coverage_expr"], ascending=[False, False]
    )

    # Main results CSV
    csv_path = outdir / "tal_gene_results.csv"
    df_sorted.to_csv(csv_path, index=False)
    logger.info(f"Saved: {csv_path}")

    # Top genes list (significant only if q-values exist)
    txt_path = outdir / "tal_top_genes.txt"
    if "q_value" in df_sorted.columns and not df_sorted["q_value"].isna().all():
        sig_df = df_sorted[df_sorted["q_value"] < args.fdr_cut]
        top_genes_df = sig_df.head(50)
    else:
        top_genes_df = df_sorted.head(50)

    with open(txt_path, "w") as f:
        f.write("# Top genes by spatial_score (descending)\n")
        f.write("# gene_symbol\tcoverage_expr\tspatial_score\tarchetype\n")
        for _, row in top_genes_df.iterrows():
            f.write(
                f"{row.get('gene_symbol', row['gene'])}\t"
                f"{row['coverage_expr']:.4f}\t"
                f"{row['spatial_score']:.4f}\t"
                f"{row.get('archetype', 'N/A')}\n"
            )
    logger.info(f"Saved: {txt_path}")

    # -------------------------------------------------------------------------
    # Stage 8: Gene-gene analysis (optional)
    # -------------------------------------------------------------------------
    if args.do_genegene:
        logger.info("[Stage 8] Running gene-gene relationship analysis...")

        # Select top genes for pairwise analysis
        if "q_value" in df_sorted.columns and not df_sorted["q_value"].isna().all():
            pair_candidates = df_sorted[df_sorted["q_value"] < args.fdr_cut]["gene"].tolist()
        else:
            pair_candidates = df_sorted.head(args.top_for_pairs * 2)["gene"].tolist()

        pair_candidates = pair_candidates[: args.top_for_pairs]
        logger.info(f"Computing pairwise relationships for {len(pair_candidates)} genes...")

        if len(pair_candidates) >= 2:
            df_pairs = score_gene_pairs(
                adata_tal,
                genes=pair_candidates,
                embedding_key=embedding_key,
                config=config,
            )

            pairs_path = outdir / "tal_gene_pairs.csv"
            df_pairs.to_csv(pairs_path, index=False)
            logger.info(f"Saved: {pairs_path} ({len(df_pairs)} pairs)")
            run_meta["n_gene_pairs"] = len(df_pairs)

            # Plot heatmap
            heatmap_path = outdir / "tal_gene_pairs_heatmap.png"
            plot_gene_pairs_heatmap(df_pairs, var_to_symbol, heatmap_path)
        else:
            logger.warning("Not enough genes for pairwise analysis")
            run_meta["n_gene_pairs"] = 0
    else:
        run_meta["n_gene_pairs"] = None

    # -------------------------------------------------------------------------
    # Stage 9: Generate plots
    # -------------------------------------------------------------------------
    logger.info("[Stage 9] Generating plots...")

    plots_dir = outdir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Archetype scatter (primary story figure)
    scatter_path = outdir / "tal_archetypes_scatter.png"
    plot_archetype_scatter(
        df_sorted,
        control_symbols,
        c_cut=run_meta["classification"]["c_cut"],
        s_cut=run_meta["classification"]["s_cut"],
        outpath=scatter_path,
    )

    # Per-gene plots for top genes + controls
    top_for_plots = df_sorted.head(args.top_k_plots)["gene"].tolist()
    plot_genes = list(set(top_for_plots + control_vars))

    logger.info(f"Generating per-gene plots for {len(plot_genes)} genes...")
    for gene in tqdm(plot_genes, desc="Plotting genes"):
        gene_row = df_sorted[df_sorted["gene"] == gene]
        if gene_row.empty:
            continue

        try:
            plot_gene_v3(
                gene,
                adata_tal,
                gene_row.iloc[0],
                embedding_key,
                config,
                var_to_symbol,
                plots_dir,
                plot_mode=args.plot_mode,
            )
        except Exception as e:
            logger.warning(f"Failed to plot {gene}: {e}")

    # -------------------------------------------------------------------------
    # Stage 10: Finalize metadata
    # -------------------------------------------------------------------------
    run_meta["end_time"] = datetime.now().isoformat()
    start_dt = datetime.fromisoformat(run_meta["timestamp"])
    end_dt = datetime.fromisoformat(run_meta["end_time"])
    run_meta["duration_seconds"] = (end_dt - start_dt).total_seconds()

    run_meta["output_files"] = {
        "results_csv": str(csv_path),
        "top_genes_txt": str(txt_path),
        "archetype_scatter": str(scatter_path),
        "plots_dir": str(plots_dir),
    }
    if args.do_genegene and run_meta.get("n_gene_pairs", 0) > 0:
        run_meta["output_files"]["gene_pairs_csv"] = str(outdir / "tal_gene_pairs.csv")
        run_meta["output_files"]["gene_pairs_heatmap"] = str(outdir / "tal_gene_pairs_heatmap.png")

    meta_path = outdir / "run_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2, default=str)
    logger.info(f"Saved: {meta_path}")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Analysis complete!")
    logger.info(f"  Duration: {run_meta['duration_seconds']:.1f} seconds")
    logger.info(f"  Genes scored: {run_meta['n_genes_scored']}")
    logger.info(f"  Results: {outdir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
