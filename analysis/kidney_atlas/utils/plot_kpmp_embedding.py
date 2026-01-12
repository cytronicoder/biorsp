"""
Plot 2D embedding for KPMP AnnData object with auto-discovery and multiple variants.

This script generates publication-ready embedding plots (UMAP/t-SNE/PCA) with
intelligent discovery of cell type labels, conditions, and QC metrics.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from matplotlib.colors import Normalize

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42

EMBEDDING_KEYS = ["X_umap", "X_UMAP", "X_tsne", "X_pca", "X_PCA"]
CELLTYPE_KEYS = [
    "cell_type",
    "celltype",
    "cell_type_fine",
    "cell_type_coarse",
    "subclass",
    "cluster",
    "leiden",
    "louvain",
    "annot",
    "label",
]
CONDITION_KEYS = ["condition", "disease", "dx", "phenotype", "injury_state", "status"]
SAMPLE_KEYS = [
    "sample",
    "sample_id",
    "donor",
    "donor_id",
    "patient",
    "patient_id",
    "batch",
    "dataset",
    "library",
]
QC_KEYS = [
    "n_genes_by_counts",
    "total_counts",
    "pct_counts_mt",
    "percent_mito",
    "log1p_total_counts",
]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot 2D embeddings for KPMP AnnData with auto-discovery",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--h5ad", type=str, default="data/kpmp.h5ad", help="Path to KPMP AnnData h5ad file"
    )
    parser.add_argument(
        "--embedding-key",
        type=str,
        default=None,
        help="Key in adata.obsm for embedding (auto-discovers if not provided)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="results/embedding_plots",
        help="Output directory for plots and metadata",
    )
    parser.add_argument("--point-size", type=float, default=2.0, help="Size of scatter points")
    parser.add_argument("--alpha", type=float, default=0.6, help="Alpha transparency for points")
    parser.add_argument(
        "--max-points",
        type=int,
        default=250000,
        help="Maximum points to plot (subsample if dataset larger)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for deterministic subsampling"
    )
    parser.add_argument(
        "--color-by",
        type=str,
        default=None,
        help="Specific obs column to color by (auto-generates multiple if not provided)",
    )
    parser.add_argument(
        "--rasterize",
        action="store_true",
        default=False,
        help="Force rasterization for scatter plots (auto-enabled for large datasets)",
    )
    parser.add_argument("--dpi", type=int, default=300, help="DPI for saved figures")
    parser.add_argument(
        "--facet-by",
        type=str,
        default=None,
        help="Column for small multiples (up to 12 categories)",
    )
    parser.add_argument(
        "--density", action="store_true", default=False, help="Generate density hexbin plot"
    )

    return parser.parse_args()


def discover_embedding_key(adata: ad.AnnData) -> str:
    """
    Auto-discover embedding key in adata.obsm.

    Parameters
    ----------
    adata : AnnData
        Annotated data object

    Returns
    -------
    str
        Discovered embedding key

    Raises
    ------
    ValueError
        If no valid 2D embedding found
    """
    available_keys = list(adata.obsm.keys())

    for key in EMBEDDING_KEYS:
        if key in adata.obsm:
            embedding = adata.obsm[key]
            if embedding.shape[1] == 2:
                logger.info(f"Discovered embedding key: {key}")
                return key
            else:
                logger.warning(f"Found {key} but it has shape {embedding.shape}, need 2D")

    key_info = [f"{k}: shape {adata.obsm[k].shape}" for k in available_keys]
    raise ValueError(
        "No valid 2D embedding found in adata.obsm.\nAvailable keys:\n" + "\n".join(key_info)
    )


def subsample_data(
    adata: ad.AnnData, max_points: int, seed: int, outdir: Path
) -> Tuple[ad.AnnData, np.ndarray]:
    """
    Deterministically subsample AnnData if too large.

    Parameters
    ----------
    adata : AnnData
        Full dataset
    max_points : int
        Maximum points to keep
    seed : int
        Random seed
    outdir : Path
        Output directory to save sampled indices

    Returns
    -------
    Tuple[AnnData, np.ndarray]
        Subsampled data and indices used
    """
    n_obs = adata.n_obs

    if n_obs <= max_points:
        logger.info(f"Dataset has {n_obs} cells, no subsampling needed")
        return adata, np.arange(n_obs)

    logger.info(f"Subsampling {max_points} from {n_obs} cells (seed={seed})")
    rng = np.random.default_rng(seed)
    indices = rng.choice(n_obs, size=max_points, replace=False)
    indices = np.sort(indices)

    indices_file = outdir / "sampled_indices.csv"
    pd.DataFrame({"index": indices}).to_csv(indices_file, index=False)
    logger.info(f"Saved sampled indices to {indices_file}")

    return adata[indices].copy(), indices


def is_categorical(series: pd.Series, threshold: int = 30) -> bool:
    """
    Determine if a pandas Series should be treated as categorical.

    Parameters
    ----------
    series : pd.Series
        Series to check
    threshold : int
        Max unique values to treat as categorical

    Returns
    -------
    bool
        True if categorical
    """
    if pd.api.types.is_numeric_dtype(series):
        n_unique = series.nunique()
        return n_unique <= threshold
    return True


def discover_color_columns(adata: ad.AnnData) -> Dict[str, str]:
    """
    Discover good columns for coloring plots.

    Parameters
    ----------
    adata : AnnData
        Annotated data

    Returns
    -------
    Dict[str, str]
        Dictionary mapping plot type to column name
    """
    discovered = {}

    for key in CELLTYPE_KEYS:
        if key in adata.obs.columns:
            discovered["celltype"] = key
            logger.info(f"Discovered cell type column: {key}")
            break

    for key in CONDITION_KEYS:
        if key in adata.obs.columns:
            discovered["condition"] = key
            logger.info(f"Discovered condition column: {key}")
            break

    for key in SAMPLE_KEYS:
        if key in adata.obs.columns:
            discovered["sample"] = key
            logger.info(f"Discovered sample column: {key}")
            break

    for key in QC_KEYS:
        if key in adata.obs.columns:
            discovered["qc"] = key
            logger.info(f"Discovered QC column: {key}")
            break

    return discovered


def get_embedding_label(embedding_key: str, axis: int) -> str:
    """
    Get axis label based on embedding type.

    Parameters
    ----------
    embedding_key : str
        Key from obsm
    axis : int
        Axis number (1 or 2)

    Returns
    -------
    str
        Axis label
    """
    key_lower = embedding_key.lower()
    if "umap" in key_lower:
        return f"UMAP {axis}"
    elif "tsne" in key_lower or "t-sne" in key_lower:
        return f"t-SNE {axis}"
    elif "pca" in key_lower:
        return f"PC {axis}"
    else:
        return f"Dim {axis}"


def plot_embedding_uncolored(
    coords: np.ndarray,
    embedding_key: str,
    n_total: int,
    n_plotted: int,
    point_size: float,
    alpha: float,
    rasterize: bool,
    outdir: Path,
    dpi: int,
):
    """Plot uncolored embedding (density impression)."""
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        s=point_size,
        alpha=alpha,
        c="gray",
        edgecolors="none",
        rasterized=rasterize,
    )

    ax.set_xlabel(get_embedding_label(embedding_key, 1), fontsize=12)
    ax.set_ylabel(get_embedding_label(embedding_key, 2), fontsize=12)
    ax.set_title(
        f"KPMP Overview\n{n_plotted:,} cells"
        + (f" (of {n_total:,})" if n_plotted < n_total else ""),
        fontsize=14,
    )
    ax.set_aspect("equal", "box")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    for ext in ["png", "pdf"]:
        outfile = outdir / f"kpmp_embedding_overview.{ext}"
        fig.savefig(outfile, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved {outfile}")

    plt.close(fig)


def plot_embedding_categorical(
    coords: np.ndarray,
    labels: pd.Series,
    embedding_key: str,
    n_total: int,
    n_plotted: int,
    point_size: float,
    alpha: float,
    rasterize: bool,
    outdir: Path,
    dpi: int,
    column_name: str,
):
    """Plot embedding colored by categorical variable."""
    fig, ax = plt.subplots(figsize=(10, 8))

    categories = labels.unique()
    n_cats = len(categories)

    if n_cats <= 10:
        cmap = plt.cm.tab10
    elif n_cats <= 20:
        cmap = plt.cm.tab20
    else:
        cmap = plt.cm.nipy_spectral

    colors = [cmap(i / max(n_cats - 1, 1)) for i in range(n_cats)]

    for i, cat in enumerate(categories):
        mask = labels == cat
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=point_size,
            alpha=alpha,
            c=[colors[i]],
            label=str(cat),
            edgecolors="none",
            rasterized=rasterize,
        )

    ax.set_xlabel(get_embedding_label(embedding_key, 1), fontsize=12)
    ax.set_ylabel(get_embedding_label(embedding_key, 2), fontsize=12)
    ax.set_title(
        f"KPMP by {column_name}\n{n_plotted:,} cells"
        + (f" (of {n_total:,})" if n_plotted < n_total else ""),
        fontsize=14,
    )
    ax.set_aspect("equal", "box")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if n_cats <= 30:
        ax.legend(
            bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8, frameon=False, markerscale=2
        )
    else:
        ax.text(
            0.98,
            0.02,
            f"Legend omitted ({n_cats} categories)",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    plt.tight_layout()

    safe_name = column_name.replace(" ", "_").replace("/", "_")
    for ext in ["png", "pdf"]:
        outfile = outdir / f"kpmp_embedding_by_{safe_name}.{ext}"
        fig.savefig(outfile, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved {outfile}")

    plt.close(fig)


def plot_embedding_continuous(
    coords: np.ndarray,
    values: pd.Series,
    embedding_key: str,
    n_total: int,
    n_plotted: int,
    point_size: float,
    alpha: float,
    rasterize: bool,
    outdir: Path,
    dpi: int,
    column_name: str,
):
    """Plot embedding colored by continuous variable."""
    fig, ax = plt.subplots(figsize=(10, 8))

    valid_mask = ~pd.isna(values)
    values_clean = values[valid_mask]
    coords_clean = coords[valid_mask]

    vmin, vmax = np.percentile(values_clean, [1, 99])
    norm = Normalize(vmin=vmin, vmax=vmax)

    scatter = ax.scatter(
        coords_clean[:, 0],
        coords_clean[:, 1],
        s=point_size,
        alpha=alpha,
        c=values_clean,
        cmap="viridis",
        norm=norm,
        edgecolors="none",
        rasterized=rasterize,
    )

    if not valid_mask.all():
        ax.scatter(
            coords[~valid_mask, 0],
            coords[~valid_mask, 1],
            s=point_size,
            alpha=alpha * 0.3,
            c="lightgray",
            edgecolors="none",
            rasterized=rasterize,
        )

    ax.set_xlabel(get_embedding_label(embedding_key, 1), fontsize=12)
    ax.set_ylabel(get_embedding_label(embedding_key, 2), fontsize=12)
    ax.set_title(
        f"KPMP by {column_name}\n{n_plotted:,} cells"
        + (f" (of {n_total:,})" if n_plotted < n_total else ""),
        fontsize=14,
    )
    ax.set_aspect("equal", "box")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(column_name, fontsize=10)

    plt.tight_layout()

    safe_name = column_name.replace(" ", "_").replace("/", "_")
    for ext in ["png", "pdf"]:
        outfile = outdir / f"kpmp_embedding_by_{safe_name}.{ext}"
        fig.savefig(outfile, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved {outfile}")

    plt.close(fig)


def plot_embedding_density(
    coords: np.ndarray, embedding_key: str, n_total: int, n_plotted: int, outdir: Path, dpi: int
):
    """Plot density using hexbin."""
    fig, ax = plt.subplots(figsize=(8, 8))

    hexbin = ax.hexbin(
        coords[:, 0], coords[:, 1], gridsize=50, cmap="YlOrRd", mincnt=1, edgecolors="none"
    )

    ax.set_xlabel(get_embedding_label(embedding_key, 1), fontsize=12)
    ax.set_ylabel(get_embedding_label(embedding_key, 2), fontsize=12)
    ax.set_title(
        f"KPMP Density\n{n_plotted:,} cells"
        + (f" (of {n_total:,})" if n_plotted < n_total else ""),
        fontsize=14,
    )
    ax.set_aspect("equal", "box")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    cbar = plt.colorbar(hexbin, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Cell count", fontsize=10)

    plt.tight_layout()

    for ext in ["png", "pdf"]:
        outfile = outdir / f"kpmp_embedding_density.{ext}"
        fig.savefig(outfile, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved {outfile}")

    plt.close(fig)


def plot_embedding_faceted(
    coords: np.ndarray,
    facet_labels: pd.Series,
    color_values: Optional[pd.Series],
    embedding_key: str,
    n_total: int,
    n_plotted: int,
    point_size: float,
    alpha: float,
    rasterize: bool,
    outdir: Path,
    dpi: int,
    facet_column: str,
    color_column: Optional[str] = None,
):
    """Plot small multiples faceted by a categorical variable."""
    categories = facet_labels.unique()
    n_cats = len(categories)

    if n_cats > 12:
        logger.warning(f"Too many categories ({n_cats}) for faceting, skipping")
        return

    ncols = min(4, n_cats)
    nrows = int(np.ceil(n_cats / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)

    for idx, cat in enumerate(categories):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row, col]

        mask = facet_labels == cat
        coords_sub = coords[mask]

        if color_values is not None:
            color_sub = color_values[mask]
            if is_categorical(color_sub):
                cats_sub = color_sub.unique()
                cmap = plt.cm.tab10 if len(cats_sub) <= 10 else plt.cm.tab20
                colors = [cmap(i / max(len(cats_sub) - 1, 1)) for i in range(len(cats_sub))]
                for i, c in enumerate(cats_sub):
                    m = color_sub == c
                    ax.scatter(
                        coords_sub[m, 0],
                        coords_sub[m, 1],
                        s=point_size * 0.5,
                        alpha=alpha,
                        c=[colors[i]],
                        edgecolors="none",
                        rasterized=rasterize,
                    )
            else:
                ax.scatter(
                    coords_sub[:, 0],
                    coords_sub[:, 1],
                    s=point_size * 0.5,
                    alpha=alpha,
                    c=color_sub,
                    cmap="viridis",
                    edgecolors="none",
                    rasterized=rasterize,
                )
        else:
            ax.scatter(
                coords_sub[:, 0],
                coords_sub[:, 1],
                s=point_size * 0.5,
                alpha=alpha,
                c="gray",
                edgecolors="none",
                rasterized=rasterize,
            )

        ax.set_title(f"{cat} (n={mask.sum()})", fontsize=10)
        ax.set_aspect("equal", "box")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if row == nrows - 1:
            ax.set_xlabel(get_embedding_label(embedding_key, 1), fontsize=9)
        if col == 0:
            ax.set_ylabel(get_embedding_label(embedding_key, 2), fontsize=9)

    for idx in range(n_cats, nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        axes[row, col].axis("off")

    color_str = f"_{color_column}" if color_column else ""
    fig.suptitle(
        f"KPMP by {facet_column}{color_str}\n{n_plotted:,} cells"
        + (f" (of {n_total:,})" if n_plotted < n_total else ""),
        fontsize=14,
    )

    plt.tight_layout()

    safe_facet = facet_column.replace(" ", "_").replace("/", "_")
    safe_color = color_column.replace(" ", "_").replace("/", "_") if color_column else "uncolored"
    for ext in ["png", "pdf"]:
        outfile = outdir / f"kpmp_embedding_facet_{safe_facet}_{safe_color}.{ext}"
        fig.savefig(outfile, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved {outfile}")

    plt.close(fig)


def main():
    """Main execution function."""
    args = parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {outdir}")

    logger.info(f"Loading AnnData from {args.h5ad}")
    try:
        adata = ad.read_h5ad(args.h5ad)
        logger.info(f"Loaded {adata.n_obs} cells × {adata.n_vars} genes")
    except Exception as e:
        logger.error(f"Failed to load AnnData: {e}")
        sys.exit(1)

    if args.embedding_key:
        if args.embedding_key not in adata.obsm:
            logger.error(
                f"Embedding key '{args.embedding_key}' not found in adata.obsm.\n"
                f"Available keys: {list(adata.obsm.keys())}"
            )
            sys.exit(1)
        embedding_key = args.embedding_key
        logger.info(f"Using specified embedding key: {embedding_key}")
    else:
        try:
            embedding_key = discover_embedding_key(adata)
        except ValueError as e:
            logger.error(str(e))
            sys.exit(1)

    embedding = adata.obsm[embedding_key]
    if embedding.shape[1] != 2:
        logger.error(f"Embedding '{embedding_key}' has shape {embedding.shape}, need 2D (n_obs, 2)")
        sys.exit(1)

    n_total = adata.n_obs
    adata_plot, sampled_indices = subsample_data(adata, args.max_points, args.seed, outdir)
    n_plotted = adata_plot.n_obs

    coords = adata_plot.obsm[embedding_key]

    rasterize = args.rasterize or (n_plotted > 50000)
    if rasterize:
        logger.info("Rasterization enabled for scatter plots")

    metadata = {
        "h5ad_path": args.h5ad,
        "embedding_key": embedding_key,
        "n_obs_total": n_total,
        "n_obs_plotted": n_plotted,
        "subsampled": n_plotted < n_total,
        "subsample_seed": args.seed if n_plotted < n_total else None,
        "point_size": args.point_size,
        "alpha": args.alpha,
        "rasterized": rasterize,
        "dpi": args.dpi,
        "plots_generated": [],
    }

    if args.color_by:
        if args.color_by not in adata_plot.obs.columns:
            logger.error(
                f"Column '{args.color_by}' not found in adata.obs.\n"
                f"Available columns: {list(adata_plot.obs.columns)}"
            )
            sys.exit(1)

        values = adata_plot.obs[args.color_by]

        if is_categorical(values):
            logger.info(f"Plotting categorical: {args.color_by}")
            plot_embedding_categorical(
                coords,
                values,
                embedding_key,
                n_total,
                n_plotted,
                args.point_size,
                args.alpha,
                rasterize,
                outdir,
                args.dpi,
                args.color_by,
            )
        else:
            logger.info(f"Plotting continuous: {args.color_by}")
            plot_embedding_continuous(
                coords,
                values,
                embedding_key,
                n_total,
                n_plotted,
                args.point_size,
                args.alpha,
                rasterize,
                outdir,
                args.dpi,
                args.color_by,
            )

        metadata["plots_generated"].append(f"by_{args.color_by}")

    else:
        logger.info("Generating multiple plot variants")

        logger.info("Plotting uncolored overview")
        plot_embedding_uncolored(
            coords,
            embedding_key,
            n_total,
            n_plotted,
            args.point_size,
            args.alpha,
            rasterize,
            outdir,
            args.dpi,
        )
        metadata["plots_generated"].append("overview")

        color_cols = discover_color_columns(adata_plot)

        if not color_cols:
            logger.warning(
                "No common metadata columns found for coloring. Only uncolored plot generated."
            )

        for plot_type, col_name in color_cols.items():
            values = adata_plot.obs[col_name]

            if is_categorical(values):
                logger.info(f"Plotting {plot_type}: {col_name} (categorical)")
                plot_embedding_categorical(
                    coords,
                    values,
                    embedding_key,
                    n_total,
                    n_plotted,
                    args.point_size,
                    args.alpha,
                    rasterize,
                    outdir,
                    args.dpi,
                    col_name,
                )
            else:
                logger.info(f"Plotting {plot_type}: {col_name} (continuous)")
                plot_embedding_continuous(
                    coords,
                    values,
                    embedding_key,
                    n_total,
                    n_plotted,
                    args.point_size,
                    args.alpha,
                    rasterize,
                    outdir,
                    args.dpi,
                    col_name,
                )

            metadata["plots_generated"].append(f"by_{col_name}")
            metadata[f"{plot_type}_column"] = col_name

    if args.density:
        logger.info("Generating density hexbin plot")
        plot_embedding_density(coords, embedding_key, n_total, n_plotted, outdir, args.dpi)
        metadata["plots_generated"].append("density")

    if args.facet_by:
        if args.facet_by not in adata_plot.obs.columns:
            logger.error(f"Facet column '{args.facet_by}' not found in adata.obs")
        else:
            facet_values = adata_plot.obs[args.facet_by]
            if not is_categorical(facet_values):
                logger.warning(
                    f"Facet column '{args.facet_by}' is continuous, converting to categorical"
                )
                facet_values = pd.cut(facet_values, bins=6)

            color_values = None
            color_col_name = None
            if args.color_by and args.color_by in adata_plot.obs.columns:
                color_values = adata_plot.obs[args.color_by]
                color_col_name = args.color_by

            logger.info(f"Generating faceted plot by {args.facet_by}")
            plot_embedding_faceted(
                coords,
                facet_values,
                color_values,
                embedding_key,
                n_total,
                n_plotted,
                args.point_size,
                args.alpha,
                rasterize,
                outdir,
                args.dpi,
                args.facet_by,
                color_col_name,
            )
            metadata["plots_generated"].append(f"facet_{args.facet_by}")
            metadata["facet_column"] = args.facet_by

    metadata_file = outdir / "metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {metadata_file}")

    logger.info("=" * 60)
    logger.info("SUCCESS: All plots generated")
    logger.info(f"Output directory: {outdir.resolve()}")
    logger.info(f"Generated {len(metadata['plots_generated'])} plot(s)")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
