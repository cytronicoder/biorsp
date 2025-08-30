"""Toy demo comparing standard UMAP with densMAP on single-cell data.

This script downloads a small single-cell dataset and walks through:
1. Basic shape and density summaries of the data.
2. Standard UMAP vs density-aware densMAP embeddings.
3. Parameter sweeps showing how neighborhood size and ``min_dist``
   influence each method.

The goal is to highlight how densMAP attempts to preserve local density
from the original high-dimensional space, whereas standard UMAP focuses
mainly on topological structure. All results are saved under
``experiments/umap_vs_densmap``.
"""

from __future__ import annotations

import os
import time
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
import requests
from tqdm import tqdm
from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CONFIG = {
    "MAX_CELLS": 5000,
    "N_PCS": 30,
    "N_NEIGHBORS_GRID": [15, 30, 50],
    "MIN_DIST_GRID": [0.1, 0.5],
    "BASELINE_N": 30,
    "BASELINE_MIN_DIST": 0.3,
    "SEED": 42,
    "SUBSAMPLE_STRATEGY": "random",
    "COLOR_KEY": None,
    "ROOT": os.path.dirname(os.path.abspath(__file__)),
}
CONFIG["FIG_DIR"] = os.path.join(CONFIG["ROOT"], "figures")
CONFIG["DATA_DIR"] = os.path.join(CONFIG["ROOT"], "data")

SAMPLE_COL_CANDIDATES = [
    "cell_type", "CellType", "celltype", "cluster", "sample", "louvain",
]

DATASET_ID = "45a06603-f923-45af-b4c3-8ead77aa2e78"
H5AD_URL = f"https://datasets.cellxgene.cziscience.com/{DATASET_ID}.h5ad"
OUT_H5AD = os.path.join(CONFIG["DATA_DIR"], f"{DATASET_ID}.h5ad")

np.random.seed(CONFIG["SEED"])

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def download_with_progress(url: str, dest_path: str, chunk_size: int = 1 << 20) -> None:
    """Download a file with a progress bar.

    Parameters
    ----------
    url : str
        HTTP(S) url to download from.
    dest_path : str
        Local path to save the file.
    chunk_size : int
        Size of chunks to stream in bytes.
    """
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    print(f"[INFO] Downloading {url} -> {dest_path}")
    t0 = time.time()
    with requests.get(url, stream=True, timeout=60) as r:  # type: ignore
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        progress = tqdm(total=total, unit="B", unit_scale=True, desc="download")
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    progress.update(len(chunk))
        progress.close()
    print(f"[INFO] Download finished in {time.time() - t0:.1f}s")


def load_anndata_in_memory() -> sc.AnnData:
    """Load the demo dataset into memory, downloading it if needed."""
    if not os.path.exists(OUT_H5AD):
        import requests  # local import to avoid unused dependency if file exists

        download_with_progress(H5AD_URL, OUT_H5AD)
    else:
        print(f"[INFO] Found existing file {OUT_H5AD} (skip download)")
    print(f"[INFO] Loading AnnData from {OUT_H5AD}")
    adata = sc.read_h5ad(OUT_H5AD)
    return adata


def detect_color_key(adata: sc.AnnData) -> str:
    """Detect a reasonable color key from ``adata.obs``.

    Preference order is given by ``SAMPLE_COL_CANDIDATES``; otherwise the
    first categorical column is used.
    """
    for key in SAMPLE_COL_CANDIDATES:
        if key in adata.obs.columns:
            return key
    # Fall back to first categorical column
    for col in adata.obs.columns:
        if pd.api.types.is_categorical_dtype(adata.obs[col]):
            return col
        if adata.obs[col].dtype == object and adata.obs[col].nunique() < 50:
            return col
    raise ValueError("Could not detect a suitable color key")


def maybe_subsample(adata: sc.AnnData, max_cells: int, seed: int, strategy: str) -> sc.AnnData:
    """Subsample the AnnData object if it exceeds ``max_cells``.

    The selected indices are stored at ``{ROOT}/subsample_indices.npy`` for
    reproducibility.
    """
    if adata.n_obs <= max_cells:
        return adata
    if strategy != "random":
        raise NotImplementedError("Only random subsampling is implemented")
    rng = np.random.default_rng(seed)
    idx = rng.choice(adata.n_obs, size=max_cells, replace=False)
    subsampled = adata[idx].copy()
    os.makedirs(CONFIG["ROOT"], exist_ok=True)
    np.save(os.path.join(CONFIG["ROOT"], "subsample_indices.npy"), idx)
    print(f"[INFO] Subsampled {adata.n_obs} -> {max_cells} cells")
    return subsampled


def compute_pca_neighbors(adata: sc.AnnData, n_neighbors: int, n_pcs: int) -> None:
    """Compute PCA and neighbor graph for ``adata``."""
    if "highly_variable" not in adata.var.columns:
        sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True, flavor="seurat")
    sc.tl.pca(adata, n_comps=n_pcs, svd_solver="arpack")
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)


def local_density_proxy_hd(adata: sc.AnnData) -> np.ndarray:
    """Mean distance to the k=10 nearest neighbors in PCA space."""
    X = adata.obsm["X_pca"]
    nn = NearestNeighbors(n_neighbors=11).fit(X)
    dists, _ = nn.kneighbors(X)
    mean_dist = dists[:, 1:].mean(axis=1)  # skip self-distance at index 0
    return mean_dist


def run_umap(adata: sc.AnnData, densmap: bool, min_dist: float, random_state: int) -> np.ndarray:
    """Run (dens)UMAP on ``adata`` using precomputed graph."""
    try:
        sc.tl.umap(
            adata,
            min_dist=min_dist,
            random_state=random_state,
            densmap=densmap,
            copy=False,
            init="spectral",
        )
        return adata.obsm["X_umap"].copy()
    except Exception as e:  # pragma: no cover - densMAP may not be available
        if densmap:
            print(f"[WARN] densMAP failed: {e}. Skipping densMAP embedding.")
            return np.zeros((adata.n_obs, 2))
        raise


def local_density_proxy_ld(X: np.ndarray) -> np.ndarray:
    """Distance to the 10th nearest neighbor in 2D embedding."""
    nn = NearestNeighbors(n_neighbors=11).fit(X)
    dists, _ = nn.kneighbors(X)
    return dists[:, -1]


def plot_embedding_scatter(
    adata: sc.AnnData, emb_key: str, color_key: str, title: str, filename: str
) -> None:
    """Scatter plot of a 2D embedding."""
    X = adata.obsm[emb_key]
    plt.figure(figsize=(5, 5))
    plt.scatter(X[:, 0], X[:, 1], c=adata.obs[color_key], s=3, cmap="tab20", linewidths=0)
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_density_scatter(X: np.ndarray, density: np.ndarray, title: str, filename: str) -> None:
    """Scatter plot colored by local density proxy."""
    plt.figure(figsize=(5, 5))
    plt.scatter(X[:, 0], X[:, 1], c=density, s=3, cmap="viridis", linewidths=0)
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(label="10th-NN dist")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_density_hist(d_std: np.ndarray, d_dens: np.ndarray, filename: str) -> None:
    """Histogram comparison of 2D densities for UMAP vs densMAP."""
    plt.figure(figsize=(5, 4))
    plt.hist(d_std, bins=50, alpha=0.5, label="UMAP")
    plt.hist(d_dens, bins=50, alpha=0.5, label="densMAP")
    plt.xlabel("10th-NN distance")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_hd_ld_corr(hd: np.ndarray, ld: np.ndarray, filename: str, title: str) -> float:
    """Plot HD vs LD density correlation and return Spearman rho."""
    rho, pval = spearmanr(hd, ld)
    plt.figure(figsize=(4, 4))
    plt.scatter(hd, ld, s=3)
    plt.xlabel("HD mean 10NN dist")
    plt.ylabel("LD 10th-NN dist")
    plt.title(f"{title}\nSpearman ρ={rho:.2f}")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    return float(rho)


def plot_trustworthiness_grid(metrics: pd.DataFrame, filename: str) -> None:
    """Parameter dashboard for trustworthiness across settings."""
    plt.figure(figsize=(6, 4))
    for method in ["std", "dens"]:
        subset = metrics[metrics["method"] == method]
        for m in CONFIG["MIN_DIST_GRID"]:
            sub = subset[subset["min_dist"] == m]
            label = f"{method}-min{m}"
            plt.plot(sub["n_neighbors"], sub["trustworthiness"], marker="o", label=label)
    plt.xlabel("n_neighbors")
    plt.ylabel("Trustworthiness")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def save_metrics(metrics: List[dict], filename: str) -> pd.DataFrame:
    """Save metrics to CSV and return the DataFrame."""
    df = pd.DataFrame(metrics)
    df.to_csv(filename, index=False)
    return df


# ---------------------------------------------------------------------------
# Main analysis pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(CONFIG["FIG_DIR"], exist_ok=True)

    adata = load_anndata_in_memory()
    adata = maybe_subsample(
        adata, CONFIG["MAX_CELLS"], CONFIG["SEED"], CONFIG["SUBSAMPLE_STRATEGY"]
    )

    color_key = CONFIG["COLOR_KEY"] or detect_color_key(adata)
    print(f"[INFO] Using color key: {color_key}")

    # Compute PCA and base neighbors
    compute_pca_neighbors(adata, CONFIG["BASELINE_N"], CONFIG["N_PCS"])
    hd_density = local_density_proxy_hd(adata)
    adata.obs["hd_density"] = hd_density

    # Basic QC plots
    sc.pp.calculate_qc_metrics(adata, inplace=True)
    plt.figure(figsize=(5, 4))
    plt.hist(np.log10(adata.obs["total_counts"] + 1), bins=50)
    plt.xlabel("log10 library size")
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["FIG_DIR"], "libsize_hist.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(5, 4))
    plt.hist(adata.obs["n_genes_by_counts"], bins=50)
    plt.xlabel("detected genes")
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["FIG_DIR"], "n_genes_hist.png"), dpi=150)
    plt.close()

    X_pca = adata.obsm["X_pca"]
    plt.figure(figsize=(5, 5))
    plt.hexbin(X_pca[:, 0], X_pca[:, 1], gridsize=50, cmap="viridis")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["FIG_DIR"], "pca_hexbin.png"), dpi=150)
    plt.close()

    metrics: List[dict] = []

    # Baseline embeddings and summary panel
    for densmap_flag, method in [(False, "std"), (True, "dens")]:
        run_umap(
            adata,
            densmap=densmap_flag,
            min_dist=CONFIG["BASELINE_MIN_DIST"],
            random_state=CONFIG["SEED"],
        )
        key = f"X_umap_{method}__n{CONFIG['BASELINE_N']}_m{CONFIG['BASELINE_MIN_DIST']}"
        adata.obsm[key] = adata.obsm["X_umap"].copy()
        ld_density = local_density_proxy_ld(adata.obsm["X_umap"])
        adata.obs[f"ld_density_{method}"] = ld_density
        trust = trustworthiness(adata.obsm["X_pca"], adata.obsm["X_umap"], n_neighbors=10)
        rho = spearmanr(hd_density, ld_density)[0]
        metrics.append(
            {
                "method": method,
                "n_neighbors": CONFIG["BASELINE_N"],
                "min_dist": CONFIG["BASELINE_MIN_DIST"],
                "trustworthiness": trust,
                "spearman_hd_ld_density": rho,
                "mean_2d_knn10_radius": ld_density.mean(),
                "std_2d_knn10_radius": ld_density.std(),
                "n_cells_used": adata.n_obs,
            }
        )

    # Baseline plots
    plot_embedding_scatter(
        adata,
        f"X_umap_std__n{CONFIG['BASELINE_N']}_m{CONFIG['BASELINE_MIN_DIST']}",
        color_key,
        "UMAP baseline",
        os.path.join(CONFIG["FIG_DIR"], "baseline_umap.png"),
    )
    plot_embedding_scatter(
        adata,
        f"X_umap_dens__n{CONFIG['BASELINE_N']}_m{CONFIG['BASELINE_MIN_DIST']}",
        color_key,
        "densMAP baseline",
        os.path.join(CONFIG["FIG_DIR"], "baseline_densmap.png"),
    )

    d_std = adata.obs["ld_density_std"].values
    d_dens = adata.obs["ld_density_dens"].values
    plot_density_hist(
        d_std,
        d_dens,
        os.path.join(CONFIG["FIG_DIR"], "baseline_density_hist.png"),
    )

    # Summary panel
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 2, 1)
    plt.scatter(
        adata.obsm[f"X_umap_std__n{CONFIG['BASELINE_N']}_m{CONFIG['BASELINE_MIN_DIST']}"][:, 0],
        adata.obsm[f"X_umap_std__n{CONFIG['BASELINE_N']}_m{CONFIG['BASELINE_MIN_DIST']}"][:, 1],
        c=adata.obs[color_key],
        s=3,
        cmap="tab20",
        linewidths=0,
    )
    plt.title("UMAP")
    plt.xticks([])
    plt.yticks([])
    plt.subplot(2, 2, 2)
    plt.scatter(
        adata.obsm[f"X_umap_dens__n{CONFIG['BASELINE_N']}_m{CONFIG['BASELINE_MIN_DIST']}"][:, 0],
        adata.obsm[f"X_umap_dens__n{CONFIG['BASELINE_N']}_m{CONFIG['BASELINE_MIN_DIST']}"][:, 1],
        c=adata.obs[color_key],
        s=3,
        cmap="tab20",
        linewidths=0,
    )
    plt.title("densMAP")
    plt.xticks([])
    plt.yticks([])
    plt.subplot(2, 2, 3)
    plt.hist(d_std, bins=50, alpha=0.5, label="UMAP")
    plt.hist(d_dens, bins=50, alpha=0.5, label="densMAP")
    plt.xlabel("10th-NN distance")
    plt.ylabel("Count")
    plt.legend()
    plt.subplot(2, 2, 4)
    plt.bar([0, 1], [metrics[0]["trustworthiness"], metrics[1]["trustworthiness"]])
    plt.xticks([0, 1], ["UMAP", "densMAP"])
    plt.ylabel("Trustworthiness")
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["FIG_DIR"], "baseline_summary_panel.png"), dpi=150)
    plt.close()

    # Grid sweep
    for n in CONFIG["N_NEIGHBORS_GRID"]:
        for m in CONFIG["MIN_DIST_GRID"]:
            compute_pca_neighbors(adata, n, CONFIG["N_PCS"])
            hd_density = local_density_proxy_hd(adata)
            for densmap_flag, method in [(False, "std"), (True, "dens")]:
                run_umap(adata, densmap_flag, m, CONFIG["SEED"])
                key = f"X_umap_{method}__n{n}_m{m}"
                adata.obsm[key] = adata.obsm["X_umap"].copy()
                ld_density = local_density_proxy_ld(adata.obsm["X_umap"])
                rho = spearmanr(hd_density, ld_density)[0]
                trust = trustworthiness(
                    adata.obsm["X_pca"], adata.obsm["X_umap"], n_neighbors=10
                )
                metrics.append(
                    {
                        "method": method,
                        "n_neighbors": n,
                        "min_dist": m,
                        "trustworthiness": trust,
                        "spearman_hd_ld_density": rho,
                        "mean_2d_knn10_radius": ld_density.mean(),
                        "std_2d_knn10_radius": ld_density.std(),
                        "n_cells_used": adata.n_obs,
                    }
                )
                # Plots
                plot_embedding_scatter(
                    adata,
                    key,
                    color_key,
                    f"{method} n={n} m={m}",
                    os.path.join(CONFIG["FIG_DIR"], f"scatter_{method}_n{n}_m{m}.png"),
                )
                plot_density_scatter(
                    adata.obsm["X_umap"],
                    ld_density,
                    f"{method} n={n} m={m}",
                    os.path.join(CONFIG["FIG_DIR"], f"density_{method}_n{n}_m{m}.png"),
                )
            plot_density_hist(
                local_density_proxy_ld(
                    adata.obsm[f"X_umap_std__n{n}_m{m}"]
                ),
                local_density_proxy_ld(
                    adata.obsm[f"X_umap_dens__n{n}_m{m}"]
                ),
                os.path.join(CONFIG["FIG_DIR"], f"hist_n{n}_m{m}.png"),
            )
            plot_hd_ld_corr(
                hd_density,
                local_density_proxy_ld(adata.obsm[f"X_umap_std__n{n}_m{m}"]),
                os.path.join(CONFIG["FIG_DIR"], f"corr_std_n{n}_m{m}.png"),
                f"UMAP n={n} m={m}",
            )
            plot_hd_ld_corr(
                hd_density,
                local_density_proxy_ld(adata.obsm[f"X_umap_dens__n{n}_m{m}"]),
                os.path.join(CONFIG["FIG_DIR"], f"corr_dens_n{n}_m{m}.png"),
                f"densMAP n={n} m={m}",
            )

    metrics_df = save_metrics(
        metrics, os.path.join(CONFIG["ROOT"], "umap_densmap_metrics.csv")
    )
    plot_trustworthiness_grid(
        metrics_df[metrics_df["min_dist"].isin(CONFIG["MIN_DIST_GRID"])],
        os.path.join(CONFIG["FIG_DIR"], "parameter_dashboard.png"),
    )

    print("[INFO] Finished. Metrics saved to", os.path.join(CONFIG["ROOT"], "umap_densmap_metrics.csv"))


if __name__ == "__main__":  # pragma: no cover
    main()
