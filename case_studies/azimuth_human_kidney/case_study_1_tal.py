#!/usr/bin/env python3
"""
Case Study 1: Azimuth Human Kidney Reference (TAL Cells)
--------------------------------------------------------
This script executes the BioRSP analysis for the Thick Ascending Limb (TAL)
cell population from the Azimuth Human Kidney reference.

It performs the following steps:
1. Loads the reference dataset (H5AD preferred).
2. Subsets cells to the TAL population and performs optional subsampling.
3. Selects a panel of genes (controls + discovery).
4. Runs BioRSP (Radial Structure Procedure) on each gene.
5. Computes depth-aware permutation statistics.
6. Performs stability diagnostics (donor-aware reruns).
7. Generates ranked summary CSVs and visualization plots.

Usage:
    python case_study_1_tal.py \
      --ref_data path/to/ref.h5ad \
      --outdir results/tal_case_study \
      --controls "SLC12A1,UMOD,EGF" \
      --donor_key donor_id \
      --n_workers 4
"""

import argparse
import faulthandler
import json
import logging
import signal
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse
from tqdm import tqdm

# Suppress known FutureWarning from legacy_api_wrap about dtype argument deprecation
warnings.filterwarnings("ignore", message=".*dtype argument is deprecated.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="legacy_api_wrap")

# BioRSP imports
try:
    from biorsp.config import BioRSPConfig
    from biorsp.geometry import geometric_median, polar_coordinates
    from biorsp.inference import compute_p_value
    from biorsp.plotting import plot_radar, plot_radar_absolute
    from biorsp.radar import compute_rsp_radar
    from biorsp.summaries import compute_scalar_summaries
except ImportError:
    # Fallback for running from examples folder without install
    sys.path.append(str(Path(__file__).parent.parent))
    from biorsp.config import BioRSPConfig
    from biorsp.geometry import geometric_median, polar_coordinates
    from biorsp.inference import compute_p_value
    from biorsp.plotting import plot_radar, plot_radar_absolute
    from biorsp.radar import compute_rsp_radar
    from biorsp.summaries import compute_scalar_summaries

# Optional imports
try:
    import anndata
    import scanpy as sc
except ImportError:
    anndata = None
    sc = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="BioRSP Case Study 1: TAL Cells in Azimuth Kidney Reference"
    )

    # Input/Output
    parser.add_argument(
        "--ref_data",
        type=str,
        required=True,
        help="Path to reference data (.h5ad preferred, or .Rds)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="Directory to save results",
    )

    # Cell Type Selection
    parser.add_argument(
        "--celltype_key",
        type=str,
        default="subclass.l1",
        help="Metadata column containing cell type annotations",
    )
    parser.add_argument(
        "--tal_labels",
        type=str,
        nargs="+",
        default=["TAL"],
        help="Label(s) identifying TAL cells in the annotation column",
    )
    parser.add_argument(
        "--donor_key",
        type=str,
        default="donor_id",
        help="Metadata column containing donor identifiers",
    )

    # Gene Selection
    parser.add_argument(
        "--controls",
        type=str,
        help="Comma-separated list of canonical TAL markers (e.g., 'SLC12A1,UMOD,EGF')",
    )
    parser.add_argument(
        "--min_pct",
        type=float,
        default=0.01,
        help="Minimum fraction of cells expressing a gene to be included in discovery panel",
    )
    parser.add_argument(
        "--max_genes",
        type=int,
        default=None,
        help="Maximum number of discovery genes to analyze (for speed)",
    )
    parser.add_argument(
        "--top_k_plots",
        type=int,
        default=12,
        help="Number of top genes to generate detailed plots for",
    )

    # BioRSP Parameters
    parser.add_argument(
        "--n_perm",
        type=int,
        default=200,
        help="Number of permutations for p-value calculation",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=1,
        help="Number of parallel workers (1 = serial)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    parser.add_argument(
        "--annoy_index",
        type=str,
        default=None,
        help="Optional path to an idx.annoy file for nearest-neighbor queries",
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=None,
        help="Optional number of cells to subsample for robustness testing",
    )
    parser.add_argument(
        "--min_cells_per_donor",
        type=int,
        default=100,
        help="Minimum cells required per donor for donor-aware analysis",
    )
    parser.add_argument(
        "--min_adequacy_fraction",
        type=float,
        default=0.9,
        help="Minimum adequacy fraction for a gene to be considered adequate",
    )
    parser.add_argument(
        "--plot_mode",
        type=str,
        choices=["relative", "absolute", "combined"],
        default="absolute",
        help="Radar plotting mode: 'relative' (signed), 'absolute' (split enrichment/depletion), or 'combined' (both on one axis)",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level for the run",
    )

    parser.add_argument(
        "--per_gene_timeout",
        type=float,
        default=None,
        help="Time limit (seconds) per gene before marking as timed out (None = no timeout)",
    )

    parser.add_argument(
        "--executor",
        type=str,
        choices=["auto", "thread", "process"],
        default="auto",
        help="Executor type for parallel runs; 'auto' chooses 'process' when n_workers > 1",
    )

    return parser.parse_args()


def load_reference(path: str) -> anndata.AnnData:
    """
    Load reference data from h5ad or Rds.
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Reference file not found: {path}")

    if path_obj.suffix.lower() == ".h5ad":
        if anndata is None:
            raise ImportError("anndata is required to load .h5ad files.")
        logger.info(f"Loading H5AD from {path}... (this may take a few minutes for large files)")
        adata = anndata.read_h5ad(path)
        logger.info(f"Successfully loaded {path}: {adata.shape}")
        return adata

    elif path_obj.suffix.lower() == ".rds":
        raise ValueError(
            f"Direct loading of .Rds files is not robustly supported in this script. "
            f"Please run the provided R conversion script first:\n"
            f"  Rscript examples/convert_ref_rds_to_h5ad.R {path} {path_obj.with_suffix('.h5ad')}\n"
            f"Then run this script with the .h5ad file."
        )
    else:
        raise ValueError(f"Unsupported file extension: {path_obj.suffix}")


def get_embedding(adata: anndata.AnnData) -> np.ndarray:
    """
    Extract 2D embedding from AnnData.

    This function is intentionally flexible: it looks for common obsm keys
    that include 'umap' (case-insensitive), falls back to explicit obs
    columns, and finally uses any 2-col obsm entry as a last resort.
    """
    # 1) obsm keys containing 'umap'
    for key in list(adata.obsm.keys()):
        if "umap" in key.lower():
            emb = adata.obsm[key]
            if getattr(emb, "ndim", 2) == 2 and emb.shape[1] >= 2:
                logger.info(f"Found embedding in obsm['{key}']")
                return emb[:, :2]

    # 2) commonly used obs columns
    if "UMAP_1" in adata.obs.columns and "UMAP_2" in adata.obs.columns:
        logger.info("Found embedding in obs columns UMAP_1, UMAP_2")
        return adata.obs[["UMAP_1", "UMAP_2"]].values

    # 3) fallback: pick first 2 columns of any 2D obsm entry
    for key in list(adata.obsm.keys()):
        emb = adata.obsm[key]
        if getattr(emb, "ndim", 2) == 2 and emb.shape[1] >= 2:
            logger.info(f"Using obsm['{key}'] as embedding (fallback)")
            return emb[:, :2]

    raise ValueError(
        f"Could not find a 2D embedding in AnnData; available obsm keys: {list(adata.obsm.keys())}"
    )


def get_umis(adata: anndata.AnnData) -> np.ndarray:
    """
    Extract or compute UMI counts per cell.

    Preference: look for common metadata columns, else sum adata.X, else try adata.raw.
    """
    candidates = ["nCount_RNA", "n_counts", "total_counts", "library_size"]
    for col in candidates:
        if col in adata.obs.columns:
            logger.info(f"Using UMI counts from obs['{col}']")
            return np.asarray(adata.obs[col].values).astype(float)

    # Compute from X if available
    logger.info("Computing UMI counts from expression matrix sum...")
    try:
        if scipy.sparse.issparse(adata.X):
            vals = np.asarray(adata.X.sum(axis=1)).reshape(-1)
        else:
            vals = np.asarray(adata.X.sum(axis=1)).reshape(-1)
        return vals.astype(float)
    except Exception:
        if getattr(adata, "raw", None) is not None and getattr(adata.raw, "X", None) is not None:
            logger.info("Computing UMIs from adata.raw.X")
            if scipy.sparse.issparse(adata.raw.X):
                return np.asarray(adata.raw.X.sum(axis=1)).reshape(-1).astype(float)
            else:
                return np.asarray(adata.raw.X.sum(axis=1)).reshape(-1).astype(float)

    raise ValueError(
        "Could not determine UMI counts from AnnData (no suitable metadata or counts)."
    )


def load_annoy_index(path: str, dim: int):
    """Load an Annoy index from disk and return the index object.

    Requires the 'annoy' package (pip install annoy).
    """
    try:
        from annoy import AnnoyIndex
    except ImportError as exc:
        raise ImportError("Install 'annoy' (pip install annoy) to load idx.annoy files") from exc
    t = AnnoyIndex(dim, metric="angular")
    t.load(path)
    return t


def analyze_single_gene(
    gene_name: str,
    expression_vector: np.ndarray,
    embedding: np.ndarray,
    umis: np.ndarray,
    config: BioRSPConfig,
    n_perm: int,
    seed: int,
) -> Dict:
    """
    Run BioRSP pipeline for a single gene.
    """
    try:
        logger.debug(f"Starting analysis for gene: {gene_name}")
        # 1. Define Foreground (Top 10% quantile rule)
        # Handle sparsity if passed as sparse matrix/array
        if scipy.sparse.issparse(expression_vector):
            x = expression_vector.toarray().flatten()
        else:
            x = np.array(expression_vector).flatten()

        n_cells = len(x)
        if n_cells == 0:
            return {
                "gene": gene_name,
                "A_g": np.nan,
                "P_g": np.nan,
                "theta_g": np.nan,
                "p_strat": np.nan,
                "n_fg": 0,
                "adequacy_fraction": 0.0,
                "is_adequate": False,
                "error": "No cells",
            }

        # Quantile threshold
        threshold = np.quantile(x, 0.90)
        # If many zero-expressed cells, threshold might be 0.
        # We enforce strictly greater than threshold if threshold > 0,
        # or just > 0 if threshold is 0 to avoid empty foreground?
        # The prompt says "top 10%". If >90% are zero, top 10% are the highest zeros?
        # Usually implies > 0. Let's use >= threshold but ensure we don't take all zeros.
        # Better rule: mask = x >= threshold. If threshold == 0, take x > 0.

        if threshold == 0:
            fg_mask = x > 0
        else:
            fg_mask = x >= threshold

        # If we still have too many (e.g. ties), or too few?
        # Let's stick to the simple quantile rule as requested.

        n_fg = int(np.sum(fg_mask))
        min_fg_total = getattr(config, "min_fg_total", max(5, int(0.01 * n_cells)))
        if n_fg < min_fg_total:
            return {
                "gene": gene_name,
                "A_g": np.nan,
                "P_g": np.nan,
                "theta_g": np.nan,
                "p_strat": np.nan,
                "n_fg": int(n_fg),
                "adequacy_fraction": 0.0,
                "is_adequate": False,
                "error": f"Insufficient foreground (n_fg={n_fg} < min_fg_total={min_fg_total})",
            }

        # 2. Compute Radar
        center = np.median(embedding, axis=0)
        r, theta = polar_coordinates(embedding, center)

        radar_res = compute_rsp_radar(
            r,
            theta,
            fg_mask,
            B=config.n_angles,
            delta_deg=config.sector_width_deg,
            min_fg_sector=config.min_fg_sector,
            min_bg_sector=config.min_bg_sector,
        )

        # Ensure rsp is a 1D numpy array
        radar_res.rsp = np.asarray(radar_res.rsp).flatten()

        # 3. Compute Summaries
        summaries = compute_scalar_summaries(radar_res)
        adequacy_fraction = float(np.mean(~np.isnan(radar_res.rsp)))
        is_adequate = adequacy_fraction >= config.min_adequacy_fraction

        # 4. Compute P-value (Depth-aware)
        p_val = np.nan
        if is_adequate:
            try:
                logger.debug(f"Computing p-value (n_perm={n_perm}) for gene {gene_name}")
                p_val, _, _ = compute_p_value(
                    r,
                    theta,
                    fg_mask,
                    B=config.n_angles,
                    delta_deg=config.sector_width_deg,
                    n_perm=n_perm,
                    umi_counts=umis,
                    umi_bins=config.umi_bins,
                    seed=seed,
                    min_fg_sector=config.min_fg_sector,
                    min_bg_sector=config.min_bg_sector,
                )
                logger.debug(f"Completed p-value for gene {gene_name}: p={p_val}")
            except Exception as e:
                logger.debug(f"Permutation p-value failed for gene {gene_name}: {e}")
                p_val = np.nan

        # Extract scalar values
        def _to_scalar(val):
            if np.isscalar(val):
                return float(val)
            elif hasattr(val, "__len__") and len(val) > 0:
                return float(val[0])
            else:
                return np.nan

        return {
            "gene": gene_name,
            "A_g": (
                _to_scalar(summaries.rms_anisotropy)
                if hasattr(summaries, "rms_anisotropy")
                else np.nan
            ),
            "P_g": (
                _to_scalar(summaries.peak_distal) if hasattr(summaries, "peak_distal") else np.nan
            ),
            "theta_g": (
                _to_scalar(summaries.peak_distal_angle)
                if hasattr(summaries, "peak_distal_angle")
                else np.nan
            ),
            "p_strat": float(p_val) if not np.isnan(p_val) else np.nan,
            "n_fg": int(n_fg),
            "adequacy_fraction": float(adequacy_fraction),
            "is_adequate": bool(is_adequate),
            "error": None,
        }

    except Exception as e:
        return {
            "gene": gene_name,
            "A_g": np.nan,
            "P_g": np.nan,
            "theta_g": np.nan,
            "p_strat": np.nan,
            "n_fg": 0,
            "adequacy_fraction": 0.0,
            "is_adequate": False,
            "error": str(e),
        }


def _process_gene_for_executor(
    gene_name: str,
    expression_vector: np.ndarray,
    embedding: np.ndarray,
    umis: np.ndarray,
    config: BioRSPConfig,
    n_perm: int,
    seed: int,
):
    """Top-level picklable wrapper for ProcessPoolExecutor submissions."""
    return analyze_single_gene(gene_name, expression_vector, embedding, umis, config, n_perm, seed)


def plot_results(
    gene_name: str,
    adata_tal: anndata.AnnData,
    res: Dict,
    outdir: Path,
    config: BioRSPConfig,
    var_to_symbol: Dict[str, str],
    plot_mode: str = "absolute",
):
    """
    Generate Embedding and Radar plots for a gene.
    """
    # Prepare data
    embedding = get_embedding(adata_tal)
    x = adata_tal[:, gene_name].X
    if scipy.sparse.issparse(x):
        x = x.toarray().flatten()
    else:
        x = x.flatten()

    threshold = np.quantile(x, 0.90)
    fg_mask = x >= threshold if threshold > 0 else x > 0

    center = np.median(embedding, axis=0)

    symbol = var_to_symbol.get(gene_name, gene_name)

    # Setup figure
    if plot_mode == "absolute":
        fig = plt.figure(figsize=(18, 6))
        gs = fig.add_gridspec(1, 3)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1], projection="polar")
        ax3 = fig.add_subplot(gs[0, 2], projection="polar")
    else:
        fig = plt.figure(figsize=(12, 6))
        gs = fig.add_gridspec(1, 2)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1], projection="polar")

    # 1. Embedding Plot
    # Background
    ax1.scatter(
        embedding[~fg_mask, 0],
        embedding[~fg_mask, 1],
        c="lightgray",
        s=5,
        alpha=0.5,
        label="Background",
    )
    # Foreground
    ax1.scatter(
        embedding[fg_mask, 0],
        embedding[fg_mask, 1],
        c="red",
        s=10,
        alpha=0.8,
        label="Foreground (Top 10%)",
    )
    ax1.scatter(center[0], center[1], c="black", marker="x", s=100, label="Vantage Point")
    ax1.set_title(f"{symbol} Expression\n(TAL Cells)", fontsize=20)
    ax1.axis("off")
    ax1.legend(loc="lower left", fontsize=18)

    # 2. Radar Plot
    # Recompute radar for plotting
    r, theta = polar_coordinates(embedding, center)
    radar_res = compute_rsp_radar(
        r,
        theta,
        fg_mask,
        B=config.n_angles,
        delta_deg=config.sector_width_deg,
        min_fg_sector=config.min_fg_sector,
        min_bg_sector=config.min_bg_sector,
    )

    # Ensure rsp is a 1D numpy array
    radar_res.rsp = np.asarray(radar_res.rsp).flatten()

    # Diagnostic logging: report NaN fraction and range
    rsp = radar_res.rsp
    n_nan = int(np.sum(np.isnan(rsp)))
    n_valid = int(np.sum(~np.isnan(rsp)))
    n_zero = int(np.sum(np.isfinite(rsp) & (np.abs(rsp) <= 1e-12)))
    logger.debug(
        f"Radar stats for {symbol}: n_sectors={len(rsp)}, n_valid={n_valid}, n_nan={n_nan}, n_zero={n_zero}, min={np.nanmin(rsp) if n_valid>0 else 'NA'}, max={np.nanmax(rsp) if n_valid>0 else 'NA'}"
    )

    if n_valid == 0:
        logger.warning(f"Radar for {symbol} has no valid sectors (all NaN); plot will be empty")
        # Provide a helpful plot annotation instead of an empty polar chart
        if plot_mode == "absolute":
            msg = "No valid sectors\n" "(insufficient foreground/background counts)"
            ax2.text(0.5, 0.5, msg, transform=ax2.transAxes, ha="center", va="center", fontsize=10)
            ax3.text(0.5, 0.5, msg, transform=ax3.transAxes, ha="center", va="center", fontsize=10)
        else:
            msg = "No valid sectors\n" "(insufficient foreground/background counts)"
            ax2.text(0.5, 0.5, msg, transform=ax2.transAxes, ha="center", va="center", fontsize=10)
    else:
        if plot_mode == "absolute":
            plot_radar(radar_res, ax=ax2, title="Enrichment RSP ($R > 0$)", mode="enrichment")
            plot_radar(radar_res, ax=ax3, title="Depletion RSP ($R < 0$)", mode="depletion")
        elif plot_mode == "relative":
            plot_radar(radar_res, ax=ax2, title="BioRSP RSP (Relative)", mode="relative")
        else:  # combined
            plot_radar(radar_res, ax=ax2, title="BioRSP RSP (Combined)", mode="combined")

    # Annotate metrics (safe formatting)
    A_g = res.get("A_g", np.nan)
    P_g = res.get("P_g", np.nan)
    theta_g = res.get("theta_g", np.nan)
    p_strat = res.get("p_strat", np.nan)

    def _fmt(x, fmt="{:.3f}"):
        try:
            if x is None or (isinstance(x, float) and np.isnan(x)):
                return "NA"
            return fmt.format(x)
        except Exception:
            return str(x)

    theta_text = "NA"
    try:
        if not (theta_g is None or np.isnan(theta_g)):
            theta_text = f"{np.degrees(theta_g):.1f}°"
    except Exception:
        theta_text = str(theta_g)

    info_text = (
        f"$A_g$: {_fmt(A_g)}\n"
        f"$P_g$: {_fmt(P_g)}\n"
        f"$\\theta_g$: {theta_text}\n"
        f"$p$: {_fmt(p_strat, '{:.1e}')}"
    )
    plt.figtext(
        0.99,
        0.9,
        info_text,
        ha="right",
        va="top",
        fontsize=20,
        bbox=dict(facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    plt.savefig(outdir / f"plot_{symbol}.png", dpi=150, bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()

    # Allow user to control logging verbosity
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, args.log_level.upper(), logging.INFO))

    # When debugging, register faulthandler so we can dump live thread traces on demand
    if args.log_level.upper() == "DEBUG":
        try:
            faulthandler.register(signal.SIGUSR1, all_threads=True)
            logger.info(
                "faulthandler registered for SIGUSR1 (send kill -SIGUSR1 <PID> to dump threads)"
            )
        except Exception as e:
            logger.warning(f"Could not register faulthandler: {e}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting BioRSP Case Study 1: TAL Cells")

    # Save run metadata
    run_meta = {
        "args": vars(args),
        "timestamp": datetime.now().isoformat(),
    }

    # 1. Load Data
    logger.info("Stage 1: Loading reference dataset")
    logger.info(f"Loading reference from {args.ref_data}...")
    adata = load_reference(args.ref_data)
    logger.info(f"Loaded data: {adata.shape}")
    logger.info(f"Available obs columns: {list(adata.obs.columns)}")
    logger.info(f"Available var columns: {list(adata.var.columns)}")
    logger.info("Stage 1 complete")

    # Create mapping from gene symbols to var_names
    var_to_symbol = {}
    if "feature_name" in adata.var.columns:
        var_to_symbol = adata.var["feature_name"].to_dict()
        logger.info("Using 'feature_name' column for gene symbols")
    elif "gene_symbol" in adata.var.columns:
        var_to_symbol = adata.var["gene_symbol"].to_dict()
        logger.info("Using 'gene_symbol' column for gene symbols")
    elif "gene_name" in adata.var.columns:
        var_to_symbol = adata.var["gene_name"].to_dict()
        logger.info("Using 'gene_name' column for gene symbols")
    else:
        # Assume var_names are symbols
        var_to_symbol = {name: name for name in adata.var_names}
        logger.info("No symbol column found; assuming var_names are gene symbols")
    symbol_to_var = {v: k for k, v in var_to_symbol.items()}
    logger.info(f"Symbol to var mapping created for {len(symbol_to_var)} genes")

    # 2. Subset TAL
    logger.info("Stage 2: Subsetting to TAL cells")
    logger.info(f"Subsetting to TAL cells (key='{args.celltype_key}', labels={args.tal_labels})...")
    if args.celltype_key not in adata.obs.columns:
        logger.info(f"Available obs columns: {list(adata.obs.columns)}")
        raise ValueError(f"Column '{args.celltype_key}' not found in metadata.")

    # Debug: print unique values in celltype_key
    unique_labels = adata.obs[args.celltype_key].unique()
    logger.info(f"Unique values in '{args.celltype_key}': {sorted(unique_labels)}")

    tal_mask = adata.obs[args.celltype_key].isin(args.tal_labels)
    logger.info(f"Subset mask: {tal_mask.sum()} cells selected out of {len(tal_mask)}")
    adata_tal = adata[tal_mask].copy()

    # Subsampling for robustness testing
    if args.subsample and args.subsample < adata_tal.n_obs:
        logger.info(f"Subsampling to {args.subsample} cells (seed={args.seed})...")
        np.random.seed(args.seed)
        idx = np.random.choice(adata_tal.n_obs, args.subsample, replace=False)
        adata_tal = adata_tal[idx].copy()

    # Donor-aware statistics
    if args.donor_key in adata_tal.obs.columns:
        donor_counts = adata_tal.obs[args.donor_key].value_counts()
        logger.info(f"Donor distribution in TAL subset:\n{donor_counts}")
        run_meta["donor_distribution"] = donor_counts.to_dict()

        # Identify donors with sufficient cells
        valid_donors = donor_counts[donor_counts >= args.min_cells_per_donor].index.tolist()
        logger.info(f"Donors with >= {args.min_cells_per_donor} cells: {valid_donors}")
        run_meta["valid_donors"] = valid_donors
    else:
        logger.warning(f"Donor key '{args.donor_key}' not found in metadata.")

    n_tal = adata_tal.n_obs
    logger.info(f"Final analysis set: {n_tal} TAL cells.")
    if n_tal < 100:
        logger.warning("Very few TAL cells found. Results may be unstable.")

    run_meta["n_tal_cells"] = int(n_tal)
    logger.info("Stage 2 complete")

    # 3. Prepare Inputs
    logger.info("Stage 3: Preparing inputs (embedding and UMIs)")
    embedding = get_embedding(adata_tal)
    logger.info(f"Embedding shape: {embedding.shape}")
    umis = get_umis(adata_tal)
    logger.info(f"UMIs shape: {umis.shape}")
    logger.info("Stage 3 complete")

    # 4. Select Genes
    logger.info("Stage 4: Selecting genes for analysis")
    all_genes = adata_tal.var_names.tolist()
    genes_to_analyze = []

    # Controls
    auto_controls = []
    if args.controls:
        controls = [g.strip() for g in args.controls.split(",")]
        # Validate existence against symbols -> var_names mapping
        valid_controls = []
        missing_controls = []
        for c in controls:
            if c in symbol_to_var:
                v = symbol_to_var[c]
                if v not in genes_to_analyze:
                    valid_controls.append(v)
            else:
                missing_controls.append(c)
        if missing_controls:
            logger.warning(
                f"The following control symbols were not found in reference and will be skipped: {missing_controls}"
            )
        genes_to_analyze.extend(valid_controls)
        logger.info(f"Added {len(valid_controls)} control genes from --controls.")
        run_meta["controls_selected"] = [var_to_symbol.get(c, c) for c in valid_controls]
    else:
        # Auto-select top mean-expression genes as controls, excluding mitochondrial/ribosomal genes
        logger.info(
            "Selecting control genes automatically (top mean-expression, excluding MT/RPS/RPL)..."
        )
        try:
            if scipy.sparse.issparse(adata_tal.X):
                means = np.asarray(adata_tal.X.mean(axis=0)).reshape(-1)
            else:
                means = np.asarray(adata_tal.X.mean(axis=0)).reshape(-1)
        except Exception:
            means = np.asarray(np.asarray(adata_tal.X).mean(axis=0)).reshape(-1)
        gene_means = list(zip(all_genes, means))
        # filter
        import re

        bad_re = re.compile(r"^(MT-|mt-|RPS|RPL)")
        gene_means = [gm for gm in gene_means if not bad_re.match(gm[0])]
        gene_means.sort(key=lambda x: -x[1])
        n_controls = min(6, len(gene_means))
        auto_controls = [g for g, _ in gene_means[:n_controls]]
        genes_to_analyze.extend(auto_controls)
        auto_controls_symbols = [var_to_symbol.get(g, g) for g in auto_controls]
        logger.info(f"Auto-selected {len(auto_controls)} control genes: {auto_controls_symbols}")
        run_meta["controls_selected"] = auto_controls_symbols

    # Discovery
    # Calculate prevalence
    logger.info("Calculating gene prevalence...")
    try:
        if scipy.sparse.issparse(adata_tal.X):
            present = np.asarray((adata_tal.X > 0).sum(axis=0)).reshape(-1)
        else:
            present = np.asarray((adata_tal.X > 0).sum(axis=0)).reshape(-1)
    except Exception:
        present = np.asarray((np.asarray(adata_tal.X) > 0).sum(axis=0)).reshape(-1)

    prevalence = present / float(n_tal)
    logger.info(f"Calculated prevalence for {len(prevalence)} genes.")

    # Filter
    discovery_mask = prevalence >= float(args.min_pct)
    discovery_genes = np.array(all_genes)[discovery_mask]

    # Remove controls from discovery to avoid dupes
    discovery_genes = [g for g in discovery_genes if g not in genes_to_analyze]

    # Limit max genes
    if args.max_genes and len(discovery_genes) > args.max_genes:
        # Simple heuristic: pick highest variance or just random/first?
        # Prompt suggests HVGs or first N. Let's do highest prevalence for simplicity/robustness
        # (most information) if scanpy not available, or HVG if available.
        if sc is not None:
            logger.info("Computing HVGs to select top genes...")
            # Normalize/log1p needed for HVG? Assuming input is normalized.
            # We'll just use dispersion on existing data.
            # Actually, let's just sort by prevalence to be safe and fast.
            # High prevalence = more signal for spatial structure.
            pass

        # Sort by prevalence descending
        # Get indices of discovery genes in all_genes
        disc_indices = [all_genes.index(g) for g in discovery_genes]
        disc_prev = prevalence[disc_indices]
        sorted_indices = np.argsort(-disc_prev)
        discovery_genes = np.array(discovery_genes)[sorted_indices[: args.max_genes]].tolist()

    genes_to_analyze.extend(discovery_genes)
    genes_to_analyze = sorted(list(set(genes_to_analyze)))  # Unique

    logger.info(
        f"Total genes to analyze: {len(genes_to_analyze)} (controls: {len(controls) if args.controls else 0}, discovery: {len(discovery_genes)})"
    )
    logger.info("Stage 4 complete")
    run_meta["n_genes_analyzed"] = len(genes_to_analyze)

    # Record dataset-level stats
    run_meta["n_cells_total"] = int(adata.n_obs)
    run_meta["n_genes_total"] = int(adata.n_vars)

    # If an annoy index is provided, try to load it (optional)
    if args.annoy_index:
        try:
            load_annoy_index(args.annoy_index, dim=embedding.shape[1])
            logger.info(f"Loaded Annoy index from {args.annoy_index}")
            run_meta["annoy_index"] = {"path": args.annoy_index, "loaded": True}
        except Exception as e:
            logger.warning(f"Could not load Annoy index: {e}")
            run_meta["annoy_index"] = {"path": args.annoy_index, "loaded": False, "error": str(e)}
    else:
        run_meta["annoy_index"] = None

    # 5. Run BioRSP
    logger.info("Stage 5: Running BioRSP analysis on each gene")
    config = BioRSPConfig(min_adequacy_fraction=args.min_adequacy_fraction)
    results = []

    logger.info(f"Starting analysis with {args.n_workers} workers...")

    # Prepare dense vectors for thread-based parallel execution (safe pickling and low overhead)
    def _process_gene_tuple(args_tuple):
        g, vec = args_tuple
        return analyze_single_gene(g, vec, embedding, umis, config, args.n_perm, args.seed)

    # Build list of (gene, 1D-numpy-vector)
    gene_vecs = []
    skipped_genes = []
    logger.info(
        f"Building expression vectors for {len(genes_to_analyze)} genes from adata_tal with shape {adata_tal.shape}"
    )
    for g in tqdm(genes_to_analyze, desc="Building vectors"):
        try:
            raw = adata_tal[:, g].X
        except KeyError:
            logger.warning(f"Gene {g} not found in subsetted AnnData; skipping")
            skipped_genes.append(g)
            continue
        if scipy.sparse.issparse(raw):
            vec = np.asarray(raw.toarray()).reshape(-1)
        else:
            vec = np.asarray(raw).reshape(-1)
        if vec.size == 0 or vec.shape[0] != adata_tal.n_obs:
            logger.warning(
                f"Gene {g} produced empty or mismatched expression vector (size={vec.size}, expected {adata_tal.n_obs}); skipping"
            )
            skipped_genes.append(g)
            continue
        gene_vecs.append((g, vec))
    logger.info(
        f"Successfully built vectors for {len(gene_vecs)} genes; skipped {len(skipped_genes)}"
    )
    if skipped_genes:
        run_meta.setdefault("skipped_genes", {})
        run_meta["skipped_genes"]["during_vector_build"] = skipped_genes

    # Run (sliding-window executor to avoid large internal queues and make stuck tasks visible)
    if args.n_workers > 1:
        # Support for per-gene timeout and choosing executor type (auto => use process pool for parallel runs)
        import time
        from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

        executor_type = args.executor if hasattr(args, "executor") else "auto"
        if executor_type == "auto":
            # Default: prefer process executor when running >1 worker to avoid nested-thread starvation
            chosen_executor = "process" if args.n_workers > 1 else "thread"
        else:
            chosen_executor = executor_type

        logger.info(f"Using {chosen_executor} executor with {args.n_workers} workers")

        ExecClass = ProcessPoolExecutor if chosen_executor == "process" else ThreadPoolExecutor

        per_gene_timeout = getattr(args, "per_gene_timeout", None)
        if per_gene_timeout is not None:
            logger.info(f"Per-gene timeout enabled: {per_gene_timeout}s")

        with ExecClass(max_workers=args.n_workers) as executor:
            it = iter(gene_vecs)
            running = {}  # future -> (gene_name, start_time)

            # Submit initial batch
            for _ in range(args.n_workers):
                try:
                    gv = next(it)
                except StopIteration:
                    break
                # For process executor we need a picklable callable and explicit args
                if chosen_executor == "process":
                    fut = executor.submit(
                        _process_gene_for_executor,
                        gv[0],
                        gv[1],
                        embedding,
                        umis,
                        config,
                        args.n_perm,
                        args.seed,
                    )
                else:
                    fut = executor.submit(_process_gene_tuple, gv)
                running[fut] = (gv[0], time.monotonic())

            # As futures finish, submit new ones to keep the pool busy
            with tqdm(total=len(gene_vecs)) as pbar:
                completed = 0
                while running:
                    try:
                        for fut in as_completed(list(running.keys()), timeout=30):
                            gene_name, _ = running.pop(fut, None)
                            try:
                                # Wait unconditionally for the future to complete (no per-gene cancellation)
                                res = fut.result()
                            except Exception as e:
                                logger.exception(f"Gene processing failed for {gene_name}: {e}")
                                res = {
                                    "gene": gene_name,
                                    "A_g": np.nan,
                                    "P_g": np.nan,
                                    "theta_g": np.nan,
                                    "p_strat": np.nan,
                                    "n_fg": 0,
                                    "adequacy_fraction": 0.0,
                                    "is_adequate": False,
                                    "error": str(e),
                                }

                            results.append(res)
                            completed += 1
                            pbar.update(1)

                            # Submit next job if available
                            try:
                                gv = next(it)
                                if chosen_executor == "process":
                                    nf = executor.submit(
                                        _process_gene_for_executor,
                                        gv[0],
                                        gv[1],
                                        embedding,
                                        umis,
                                        config,
                                        args.n_perm,
                                        args.seed,
                                    )
                                else:
                                    nf = executor.submit(_process_gene_tuple, gv)
                                running[nf] = (gv[0], time.monotonic())
                            except StopIteration:
                                pass
                    except TimeoutError:
                        # No future completed within 30s — tasks are still running; wait for completion
                        now = time.monotonic()
                        msg = "No gene finished within 30s — tasks still running; waiting for completion"
                        logger.info(msg)

                        # Log currently running tasks and their durations for diagnostics
                        for f, (gname, start_t) in running.items():
                            duration = now - start_t
                            logger.debug(f"Running gene '{gname}' for {duration:.1f}s")

                        # Continue and wait for tasks to finish naturally
                        continue

    else:
        for g, vec in tqdm(gene_vecs):
            res = analyze_single_gene(g, vec, embedding, umis, config, args.n_perm, args.seed)
            results.append(res)
    logger.info("Stage 5 complete")

    # 5.5 Stability Diagnostics (Donor-aware)
    logger.info("Stage 5.5: Stability Diagnostics (Donor-aware)")
    stability_results = []
    if args.donor_key in adata_tal.obs.columns:
        valid_donors = run_meta.get("valid_donors", [])
        if valid_donors:
            # Pick top 5 genes by anisotropy for stability check
            # (Need to create df_res first or just sort results list)
            temp_df = pd.DataFrame([r for r in results if r.get("is_adequate")])
            if not temp_df.empty:
                top_genes_for_stability = (
                    temp_df.sort_values("A_g", ascending=False).head(5)["gene"].tolist()
                )
                # Also include controls
                if args.controls:
                    controls = [g.strip() for g in args.controls.split(",")]
                    control_vars = [symbol_to_var.get(c) for c in controls if c in symbol_to_var]
                    top_genes_for_stability = list(set(top_genes_for_stability + control_vars))

                logger.info(
                    f"Running stability check for {len(top_genes_for_stability)} genes across {len(valid_donors)} donors..."
                )
                for g in tqdm(top_genes_for_stability, desc="Stability genes"):
                    symbol = var_to_symbol.get(g, g)
                    # Get expression vector for this gene
                    raw = adata_tal[:, g].X
                    vec = (
                        np.asarray(raw.toarray()).reshape(-1)
                        if scipy.sparse.issparse(raw)
                        else np.asarray(raw).reshape(-1)
                    )

                    for donor in tqdm(valid_donors, desc=f"Donors for {symbol}", leave=False):
                        donor_mask = (adata_tal.obs[args.donor_key] == donor).values
                        if donor_mask.sum() < args.min_cells_per_donor:
                            continue

                        # Run BioRSP on donor subset
                        d_emb = embedding[donor_mask]
                        d_vec = vec[donor_mask]
                        d_umis = umis[donor_mask]

                        d_res = analyze_single_gene(
                            g, d_vec, d_emb, d_umis, config, n_perm=0, seed=args.seed
                        )
                        d_res["donor"] = donor
                        d_res["gene_symbol"] = symbol
                        stability_results.append(d_res)

            if stability_results:
                df_stability = pd.DataFrame(stability_results)
                stability_csv = outdir / "stability_diagnostics.csv"
                df_stability.to_csv(stability_csv, index=False)
                logger.info(
                    f"Saved stability diagnostics to {stability_csv} ({len(stability_results)} donor-gene combinations)"
                )
                run_meta["stability_diagnostics"] = str(stability_csv)
        else:
            logger.info("No donors with sufficient cells for stability check.")
    else:
        logger.info("Skipping stability diagnostics (no donor key).")

    # 6. Save Results
    logger.info("Stage 6: Saving results")

    # Sanitize and validate results list
    csv_path = outdir / "tal_gene_results.csv"
    expected_keys = [
        "gene",
        "A_g",
        "P_g",
        "theta_g",
        "p_strat",
        "n_fg",
        "adequacy_fraction",
        "is_adequate",
        "error",
    ]

    sanitized = []
    for r in results:
        if not isinstance(r, dict):
            logger.warning(f"Skipping non-dict result entry: {r}")
            continue
        gene = r.get("gene")
        if gene is None:
            logger.warning(f"Skipping result with missing gene key: {r}")
            continue
        sanitized.append(
            {
                "gene": gene,
                "A_g": r.get("A_g", np.nan),
                "P_g": r.get("P_g", np.nan),
                "theta_g": r.get("theta_g", np.nan),
                "p_strat": r.get("p_strat", np.nan),
                "n_fg": int(r.get("n_fg", 0)),
                "adequacy_fraction": float(r.get("adequacy_fraction", 0.0)),
                "is_adequate": bool(r.get("is_adequate", False)),
                "error": r.get("error", None),
            }
        )

    df_res = pd.DataFrame(sanitized, columns=expected_keys)

    if df_res.empty:
        # Write empty CSV with header and exit gracefully
        df_res.to_csv(csv_path, index=False)
        logger.warning("No valid results were produced; wrote empty results CSV and exiting.")
        run_meta["n_results"] = 0
        run_meta["end_time"] = datetime.now().isoformat()
        run_meta["duration_seconds"] = (
            datetime.fromisoformat(run_meta["end_time"])
            - datetime.fromisoformat(run_meta["timestamp"])
        ).total_seconds()
        run_meta["output_files"] = {
            "results_csv": str(csv_path),
        }
        with open(outdir / "run_metadata.json", "w", encoding="utf-8") as f:
            json.dump(run_meta, f, indent=2)
        logger.info("Analysis finished with no valid results.")
        return

    # Map gene symbols for readability
    df_res["gene_symbol"] = df_res["gene"].map(var_to_symbol)

    # Rank genes by anisotropy score (A_g)
    # We only rank genes that meet adequacy criteria for the primary summary
    if "A_g" in df_res.columns:
        df_res = df_res.sort_values(["is_adequate", "A_g"], ascending=[False, False])

    df_res.to_csv(csv_path, index=False)
    logger.info(f"Saved ranked results to {csv_path} ({len(df_res)} genes)")
    logger.info("Stage 6 complete")

    # 7. Top Genes & Plots
    logger.info("Stage 7: Generating plots for top genes")
    # Filter for adequate genes (safe when column may be missing)
    if "is_adequate" in df_res.columns:
        valid_df = df_res[df_res["is_adequate"].fillna(False)]
    else:
        valid_df = df_res.iloc[0:0]
    top_genes = valid_df.head(args.top_k_plots)["gene"].tolist()

    # Add controls to plot list
    if args.controls:
        controls = [g.strip() for g in args.controls.split(",")]
        plot_controls = [symbol_to_var.get(c) for c in controls if c in symbol_to_var]
        plot_genes = list(set(top_genes + plot_controls))
    else:
        plot_genes = top_genes

    top_genes_symbols = valid_df.head(args.top_k_plots)["gene_symbol"].tolist()

    txt_path = outdir / "tal_top_genes.txt"
    with open(txt_path, "w") as f:
        for g in top_genes_symbols:
            f.write(f"{g}\n")

    logger.info(f"Generating plots for {len(plot_genes)} genes...")
    plots_dir = outdir / "plots"
    plots_dir.mkdir(exist_ok=True)

    for g in tqdm(plot_genes, desc="Generating plots"):
        # Ensure g is a var_name present in adata_tal; if symbol was supplied, map it
        var_g = g
        if var_g not in all_genes and g in symbol_to_var:
            var_g = symbol_to_var[g]
        if var_g not in all_genes:
            logger.warning(f"Plot gene {g} not found in dataset (after mapping), skipping")
            continue
        # Find result row (skip if not found)
        sub = df_res[df_res["gene"] == var_g]
        if sub.shape[0] == 0:
            logger.warning(f"No result row found for {var_g}, skipping plot")
            continue
        # Quick check: ensure expression vector is non-empty before plotting
        raw = adata_tal[:, var_g].X
        if scipy.sparse.issparse(raw):
            vec = np.asarray(raw.toarray()).reshape(-1)
        else:
            vec = np.asarray(raw).reshape(-1)
        if vec.size == 0:
            logger.warning(f"Expression vector for {var_g} is empty; skipping plot")
            continue
        row = sub.iloc[0].to_dict()
        plot_results(
            var_g, adata_tal, row, plots_dir, config, var_to_symbol, plot_mode=args.plot_mode
        )

    logger.info("Stage 7 complete")

    # Finalize run metadata
    run_meta["end_time"] = datetime.now().isoformat()
    run_meta["duration_seconds"] = (
        datetime.fromisoformat(run_meta["end_time"]) - datetime.fromisoformat(run_meta["timestamp"])
    ).total_seconds()
    run_meta["output_files"] = {
        "results_csv": str(csv_path),
        "top_genes": str(txt_path),
        "plots_dir": str(plots_dir),
    }

    with open(outdir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)

    logger.info("Analysis complete.")
    logger.info(f"Total runtime: {run_meta['duration_seconds']:.1f} seconds")
    logger.info(f"Results saved to: {outdir}")


if __name__ == "__main__":
    main()
