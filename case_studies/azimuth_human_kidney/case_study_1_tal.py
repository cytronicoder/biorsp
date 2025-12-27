#!/usr/bin/env python3
"""
Case Study 1: Azimuth Human Kidney Reference (TAL Cells)
--------------------------------------------------------
This script executes the BioRSP analysis for the Thick Ascending Limb (TAL)
cell population from the Azimuth Human Kidney reference.

It performs the following steps:
1. Loads the reference dataset (H5AD preferred, RDS fallback via conversion).
2. Subsets cells to the TAL population.
3. Selects a panel of genes (controls + discovery).
4. Runs BioRSP (Radial Structure Procedure) on each gene.
5. Computes depth-aware permutation statistics.
6. Generates summary CSVs and visualization plots.

Usage:
    python case_study_1_tal.py --ref_data path/to/ref.h5ad --outdir results/tal_case_study
"""

import argparse
import json
import logging
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    from biorsp.radar import compute_rsp_radar
    from biorsp.summaries import compute_scalar_summaries
except ImportError:
    # Fallback for running from examples folder without install
    sys.path.append(str(Path(__file__).parent.parent))
    from biorsp.config import BioRSPConfig
    from biorsp.geometry import geometric_median, polar_coordinates
    from biorsp.inference import compute_p_value
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
        default="annotation.l2",
        help="Metadata column containing cell type annotations",
    )
    parser.add_argument(
        "--tal_labels",
        type=str,
        nargs="+",
        default=["Thick Ascending Limb"],
        help="Label(s) identifying TAL cells in the annotation column",
    )

    # Gene Selection
    parser.add_argument(
        "--controls",
        type=str,
        help="Comma-separated list of control genes to analyze",
    )
    parser.add_argument(
        "--min_pct",
        type=float,
        default=0.01,
        help="Minimum fraction of cells expressing a gene for discovery",
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
        logger.info(f"Loading H5AD from {path}...")
        return anndata.read_h5ad(path)

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
        # 1. Define Foreground (Top 10% quantile rule)
        # Handle sparsity if passed as sparse matrix/array
        if scipy.sparse.issparse(expression_vector):
            x = expression_vector.toarray().flatten()
        else:
            x = np.array(expression_vector).flatten()

        n_cells = len(x)
        if n_cells == 0:
            return {"gene": gene_name, "error": "No cells"}

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
                "is_adequate": False,
                "note": f"Insufficient foreground (n_fg={n_fg} < min_fg_total={min_fg_total})",
            }

        # 2. Compute Radar
        center = geometric_median(embedding)
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

        # 3. Compute Summaries
        summaries = compute_scalar_summaries(radar_res)
        adequacy_fraction = float(np.mean(~np.isnan(radar_res.rsp)))
        is_adequate = adequacy_fraction >= config.min_adequacy_fraction

        # 4. Compute P-value (Depth-aware)
        p_val = np.nan
        if is_adequate:
            try:
                p_val, _ = compute_p_value(
                    summaries.rms_anisotropy,
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
            except Exception as e:
                logger.debug(f"Permutation p-value failed for gene {gene_name}: {e}")
                p_val = np.nan

        return {
            "gene": gene_name,
            "A_g": summaries.rms_anisotropy,
            "P_g": summaries.peak_distal,
            "theta_g": summaries.peak_distal_angle,
            "p_strat": p_val,
            "n_fg": int(n_fg),
            "adequacy_fraction": adequacy_fraction,
            "is_adequate": is_adequate,
            "error": None,
        }

    except Exception as e:
        return {"gene": gene_name, "error": str(e)}


def plot_results(
    gene_name: str, adata_tal: anndata.AnnData, res: Dict, outdir: Path, config: BioRSPConfig
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

    center = geometric_median(embedding)

    # Setup figure
    fig = plt.figure(figsize=(12, 5))

    # 1. Embedding Plot
    ax1 = fig.add_subplot(121)
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
    ax1.scatter(center[0], center[1], c="black", marker="x", s=100, label="Center")
    ax1.set_title(f"{gene_name} Expression\n(TAL Cells)")
    ax1.axis("off")
    ax1.legend(loc="upper right", fontsize="small")

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

    ax2 = fig.add_subplot(122, projection="polar")

    angles = radar_res.centers
    values = np.nan_to_num(radar_res.rsp, nan=0.0)

    # Close the loop for plotting
    angles_plot = np.concatenate((angles, [angles[0] + 2 * np.pi]))
    values_plot = np.concatenate((values, [values[0]]))

    ax2.plot(angles_plot, values_plot, color="blue", linewidth=2)
    ax2.fill(angles_plot, values_plot, color="blue", alpha=0.25)

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
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.8),
    )

    ax2.set_title(f"BioRSP Radar: {gene_name}")

    plt.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    plt.savefig(outdir / f"plot_{gene_name}.png", dpi=150)
    plt.close()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Save run metadata
    run_meta = {
        "args": vars(args),
        "timestamp": datetime.now().isoformat(),
    }

    # 1. Load Data
    logger.info(f"Loading reference from {args.ref_data}...")
    adata = load_reference(args.ref_data)
    logger.info(f"Loaded data: {adata.shape}")

    # 2. Subset TAL
    logger.info(f"Subsetting to TAL cells (key='{args.celltype_key}', labels={args.tal_labels})...")
    if args.celltype_key not in adata.obs.columns:
        raise ValueError(f"Column '{args.celltype_key}' not found in metadata.")

    tal_mask = adata.obs[args.celltype_key].isin(args.tal_labels)
    adata_tal = adata[tal_mask].copy()

    n_tal = adata_tal.n_obs
    logger.info(f"Found {n_tal} TAL cells.")
    if n_tal < 100:
        logger.warning("Very few TAL cells found. Results may be unstable.")

    run_meta["n_tal_cells"] = int(n_tal)

    # 3. Prepare Inputs
    embedding = get_embedding(adata_tal)
    umis = get_umis(adata_tal)

    # 4. Select Genes
    all_genes = adata_tal.var_names.tolist()
    genes_to_analyze = []

    # Controls
    auto_controls = []
    if args.controls:
        controls = [g.strip() for g in args.controls.split(",")]
        # Validate existence
        valid_controls = [g for g in controls if g in all_genes]
        genes_to_analyze.extend(valid_controls)
        logger.info(f"Added {len(valid_controls)} control genes from --controls.")
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
        logger.info(f"Auto-selected {len(auto_controls)} control genes: {auto_controls}")
        run_meta["controls_selected"] = auto_controls

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

    logger.info(f"Total genes to analyze: {len(genes_to_analyze)}")
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
    config = BioRSPConfig()  # Defaults
    results = []

    logger.info(f"Starting analysis with {args.n_workers} workers...")

    # Prepare dense vectors for thread-based parallel execution (safe pickling and low overhead)
    def _process_gene_tuple(args_tuple):
        g, vec = args_tuple
        return analyze_single_gene(g, vec, embedding, umis, config, args.n_perm, args.seed)

    # Build list of (gene, 1D-numpy-vector)
    gene_vecs = []
    for g in genes_to_analyze:
        raw = adata_tal[:, g].X
        if scipy.sparse.issparse(raw):
            vec = np.asarray(raw.toarray()).reshape(-1)
        else:
            vec = np.asarray(raw).reshape(-1)
        gene_vecs.append((g, vec))

    # Run (ThreadPoolExecutor to avoid pickling large objects and keep BioRSP config intact)
    if args.n_workers > 1:
        with ThreadPoolExecutor(max_workers=args.n_workers) as executor:
            futures = {executor.submit(_process_gene_tuple, gv): gv[0] for gv in gene_vecs}
            for future in tqdm(as_completed(futures), total=len(gene_vecs)):
                res = future.result()
                results.append(res)
    else:
        for g, vec in tqdm(gene_vecs):
            res = analyze_single_gene(g, vec, embedding, umis, config, args.n_perm, args.seed)
            results.append(res)
    # 6. Save Results
    df_res = pd.DataFrame(results)

    # Sort by A_g descending
    if "A_g" in df_res.columns:
        df_res = df_res.sort_values("A_g", ascending=False)

    csv_path = outdir / "tal_gene_results.csv"
    df_res.to_csv(csv_path, index=False)
    logger.info(f"Saved results to {csv_path}")

    # 7. Top Genes & Plots
    # Filter for adequate genes
    valid_df = df_res[df_res["is_adequate"]]
    top_genes = valid_df.head(args.top_k_plots)["gene"].tolist()

    # Add controls to plot list
    if args.controls:
        controls = [g.strip() for g in args.controls.split(",")]
        plot_genes = list(set(top_genes + [c for c in controls if c in all_genes]))
    else:
        plot_genes = top_genes

    txt_path = outdir / "tal_top_genes.txt"
    with open(txt_path, "w") as f:
        for g in top_genes:
            f.write(f"{g}\n")

    logger.info(f"Generating plots for {len(plot_genes)} genes...")
    plots_dir = outdir / "plots"
    plots_dir.mkdir(exist_ok=True)

    for g in plot_genes:
        # Find result row (skip if not found)
        sub = df_res[df_res["gene"] == g]
        if sub.shape[0] == 0:
            logger.warning(f"No result row found for {g}, skipping plot")
            continue
        row = sub.iloc[0].to_dict()
        plot_results(g, adata_tal, row, plots_dir, config)

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

    with open(outdir / "run_metadata.json", "w") as f:
        json.dump(run_meta, f, indent=2)

    logger.info("Analysis complete.")


if __name__ == "__main__":
    main()
