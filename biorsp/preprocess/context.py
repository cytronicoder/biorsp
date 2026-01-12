"""Reusable geometry context for BioRSP scoring.

This module provides a BioRSPContext dataclass that precomputes and stores
all geometric information (vantage, polar coordinates, sector indices) once,
allowing efficient scoring across many genes without redundant computation.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from anndata import AnnData

from biorsp.core.geometry import compute_vantage, get_sector_indices, polar_coordinates
from biorsp.preprocess.normalization import normalize_radii
from biorsp.utils.config import BioRSPConfig


@dataclass
class BioRSPContext:
    """Precomputed geometric context for BioRSP scoring.

    Stores all cell-level and sector-level information that is constant
    across genes, enabling efficient multi-gene scoring loops.

    Attributes
    ----------
    coords : np.ndarray
        (N, 2) array of embedding coordinates.
    center : np.ndarray
        (2,) vantage point coordinates.
    r_norm : np.ndarray
        (N,) normalized radial distances.
    theta : np.ndarray
        (N,) angular positions in [-pi, pi).
    sector_indices : List[np.ndarray]
        Precomputed cell indices for each angular sector.
    sector_sort_indices : List[Optional[np.ndarray]]
        Precomputed sort indices for r_norm within each sector.
    norm_stats : Dict[str, Any]
        Statistics from radial normalization (median, iqr, etc.).
    n_cells : int
        Number of cells in the context.
    embedding_key : str
        Key used to extract embedding from AnnData.
    subset_query : Optional[str]
        Query string used to subset cells (for provenance).
    config : BioRSPConfig
        Configuration used to build this context.
    cell_indices : Optional[np.ndarray]
        Original indices into the source AnnData (if subsampled).
    stratify_labels : Optional[np.ndarray]
        Stratification labels for permutation tests.
    """

    coords: np.ndarray
    center: np.ndarray
    r_norm: np.ndarray
    theta: np.ndarray
    sector_indices: List[np.ndarray]
    sector_sort_indices: List[Optional[np.ndarray]]
    norm_stats: Dict[str, Any]
    n_cells: int
    embedding_key: str
    subset_query: Optional[str] = None
    config: Optional[BioRSPConfig] = None
    cell_indices: Optional[np.ndarray] = None
    stratify_labels: Optional[np.ndarray] = None
    _metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Export context metadata for manifest/provenance."""
        return {
            "n_cells": self.n_cells,
            "embedding_key": self.embedding_key,
            "subset_query": self.subset_query,
            "center": self.center.tolist(),
            "norm_stats": {
                k: float(v) if isinstance(v, (np.floating, float)) else v
                for k, v in self.norm_stats.items()
            },
            "B": self.config.B if self.config else None,
            "delta_deg": self.config.delta_deg if self.config else None,
            **self._metadata,
        }


def discover_embedding_key(adata: AnnData) -> str:
    """Auto-discover the best available 2D embedding key.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.

    Returns
    -------
    str
        The embedding key to use.

    Raises
    ------
    ValueError
        If no valid 2D embedding is found.
    """
    candidates = ["X_umap", "X_UMAP", "X_tsne", "X_tSNE", "X_pca", "X_PCA"]

    for key in candidates:
        if key in adata.obsm:
            emb = adata.obsm[key]
            if emb.shape[1] >= 2:
                return key

    for key in adata.obsm:
        emb = adata.obsm[key]
        if emb.shape[1] == 2:
            return key

    available = [(k, adata.obsm[k].shape) for k in adata.obsm]
    raise ValueError(
        f"No valid 2D embedding found in adata.obsm.\nAvailable keys and shapes: {available}"
    )


def prepare_context(
    adata: AnnData,
    embedding_key: Optional[str] = None,
    subset: Optional[Union[dict, str, List[str]]] = None,
    config: Optional[BioRSPConfig] = None,
    max_cells: Optional[int] = None,
    seed: int = 42,
    stratify_key: Optional[str] = None,
    n_strata: int = 10,
) -> Tuple[AnnData, BioRSPContext]:
    """Prepare reusable geometric context for BioRSP scoring.

    This function performs all cell-level preprocessing once:
    - Applies subset filter if specified
    - Subsamples if dataset exceeds max_cells
    - Computes vantage point
    - Transforms to polar coordinates
    - Normalizes radii
    - Precomputes sector indices

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    embedding_key : str, optional
        Key in adata.obsm for the embedding. If None, auto-detects.
    subset : Union[dict, str, List[str]], optional
        Subset specification:
        - dict: {column: value} filter
        - str: pandas eval query on adata.obs
        - List[str]: explicit cell names/indices
    config : BioRSPConfig, optional
        BioRSP configuration. Uses defaults if None.
    max_cells : int, optional
        Maximum cells to use. If exceeded, deterministically subsamples.
    seed : int, optional
        Random seed for subsampling, by default 42.
    stratify_key : str, optional
        Column in adata.obs for stratified permutation tests.
    n_strata : int, optional
        Number of strata for numeric stratification, by default 10.

    Returns
    -------
    Tuple[AnnData, BioRSPContext]
        The (possibly subsetted/subsampled) AnnData and precomputed context.
    """
    if config is None:
        config = BioRSPConfig(seed=seed)

    if embedding_key is None:
        embedding_key = discover_embedding_key(adata)

    subset_query = None
    if subset is not None:
        if isinstance(subset, dict):
            for k, v in subset.items():
                adata = adata[adata.obs[k] == v].copy()
            subset_query = str(subset)
        elif isinstance(subset, str):
            adata = adata[adata.obs.eval(subset)].copy()
            subset_query = subset
        else:
            adata = adata[subset].copy()
            subset_query = f"explicit_indices[{len(subset)}]"

    if adata.n_obs == 0:
        raise ValueError("Subset resulted in 0 cells.")

    cell_indices = None
    if max_cells is not None and adata.n_obs > max_cells:
        rng = np.random.default_rng(seed)
        indices = rng.choice(adata.n_obs, size=max_cells, replace=False)
        indices = np.sort(indices)
        cell_indices = indices
        adata = adata[indices].copy()

    coords = adata.obsm[embedding_key]
    if coords.shape[1] != 2:
        coords = coords[:, :2]

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

    r_norm, norm_stats = normalize_radii(r)

    sector_indices = get_sector_indices(theta, config.B, config.delta_deg)

    sector_sort_indices = []
    for idx_s in sector_indices:
        if idx_s.size > 0:
            sector_sort_indices.append(np.argsort(r_norm[idx_s]))
        else:
            sector_sort_indices.append(None)

    stratify_labels = None
    if stratify_key is not None and stratify_key in adata.obs:
        vals = adata.obs[stratify_key].values
        if np.issubdtype(vals.dtype, np.number):
            stratify_labels = pd.qcut(vals, n_strata, labels=False, duplicates="drop")
        else:
            stratify_labels = vals

    context = BioRSPContext(
        coords=coords,
        center=center,
        r_norm=r_norm,
        theta=theta,
        sector_indices=sector_indices,
        sector_sort_indices=sector_sort_indices,
        norm_stats=norm_stats,
        n_cells=adata.n_obs,
        embedding_key=embedding_key,
        subset_query=subset_query,
        config=config,
        cell_indices=cell_indices,
        stratify_labels=stratify_labels,
    )

    return adata, context


def score_gene_with_context(
    adata: AnnData,
    gene: str,
    context: BioRSPContext,
    config: Optional[BioRSPConfig] = None,
    compute_pvalue: bool = False,
    n_permutations: int = 500,
) -> Dict[str, Any]:
    """Score a single gene using precomputed context.

    This is the core scoring function that uses a precomputed BioRSPContext
    to avoid redundant geometry calculations.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix (should match the context).
    gene : str
        Gene name to score.
    context : BioRSPContext
        Precomputed geometric context.
    config : BioRSPConfig, optional
        Configuration override. Uses context's config if None.
    compute_pvalue : bool, optional
        Whether to compute permutation p-value, by default False.
    n_permutations : int, optional
        Number of permutations for p-value, by default 500.

    Returns
    -------
    Dict[str, Any]
        Dictionary with gene scores and metadata.
    """
    from scipy import sparse

    from biorsp.core.engine import compute_rsp_radar
    from biorsp.preprocess.foreground import define_foreground

    if config is None:
        config = context.config or BioRSPConfig()

    try:
        idx = adata.var_names.get_loc(gene)
        if sparse.issparse(adata.X):
            x = adata.X[:, idx].toarray().flatten()
        else:
            x = adata.X[:, idx].copy()
    except KeyError:
        return {
            "gene": gene,
            "error": f"Gene {gene} not found",
            "spatial_score": np.nan,
            "coverage": np.nan,
        }

    is_integers = (x.dtype.kind in "iu") or (np.allclose(x, np.round(x)) and np.max(x) > 1.0)
    if config.expr_threshold_mode == "fixed" and config.expr_threshold_value is not None:
        t_g = config.expr_threshold_value
        t_mode = "fixed"
    elif config.expr_threshold_mode == "detect":
        t_g = 1.0 if is_integers else 1e-6
        t_mode = "detect_count" if is_integers else "detect_continuous"
    else:
        t_g = 1.0 if is_integers else 1e-6
        t_mode = "default"

    coverage = float(np.mean(x >= t_g))

    y, fg_info = define_foreground(
        x,
        mode=config.foreground_mode,
        q=config.foreground_quantile,
        abs_threshold=config.foreground_threshold,
        min_fg=config.min_fg_total,
        rng=np.random.default_rng(config.seed),
    )

    warnings = []
    if coverage < 0.01:
        warnings.append("low_coverage")

    if y is None:
        return {
            "gene": gene,
            "n_cells_total": len(x),
            "expr_threshold_mode": t_mode,
            "expr_threshold_value": t_g,
            "coverage": coverage,
            "spatial_score": 0.0,
            "spatial_sign": 0,
            "r_mean_bg": 0.0,
            "coverage_geom": 0.0,
            "coverage_fg": 0.0,
            "p_value": np.nan,
            "q_value": np.nan,
            "warnings": ";".join(warnings + ["insufficient_internal_fg"]),
        }

    radar = compute_rsp_radar(
        context.r_norm,
        context.theta,
        y,
        config=config,
        sector_indices=context.sector_indices,
        sector_sort_indices=context.sector_sort_indices,
    )

    geom_mask = radar.geom_supported_mask
    if geom_mask is None or not np.any(geom_mask):
        s_g = 0.0
        r_mean = 0.0
    else:
        valid_rsp = np.nan_to_num(radar.rsp[geom_mask], nan=0.0)
        w = radar.sector_weights[geom_mask]
        sum_w = np.sum(w)
        if sum_w > 0:
            s_g = float(np.sqrt(np.sum(w * valid_rsp**2) / sum_w))
            r_mean = float(np.sum(w * valid_rsp) / sum_w)
        else:
            s_g = 0.0
            r_mean = 0.0

    cov_geom = float(np.mean(geom_mask)) if geom_mask is not None else 0.0
    if cov_geom < 0.8:
        warnings.append("low_coverage_geom")

    if geom_mask is not None and np.any(geom_mask):
        n_fg_sector = radar.n_fg_per_sector[geom_mask]
        cov_fg = float(np.mean(n_fg_sector >= config.min_fg_sector))
    else:
        cov_fg = 0.0

    result = {
        "gene": gene,
        "n_cells_total": len(x),
        "expr_threshold_mode": t_mode,
        "expr_threshold_value": t_g,
        "coverage": coverage,
        "spatial_score": s_g,
        "spatial_sign": int(np.sign(r_mean)),
        "r_mean_bg": r_mean,
        "coverage_geom": cov_geom,
        "coverage_fg": cov_fg,
        "warnings": ";".join(warnings),
    }

    if compute_pvalue and n_permutations > 0:
        p_val, null_mean, null_sd = _permute_spatial_score(context, y, s_g, config, n_permutations)
        result["p_value"] = p_val
        result["null_mean"] = null_mean
        result["null_sd"] = null_sd
    else:
        result["p_value"] = np.nan

    return result


def _permute_spatial_score(
    context: BioRSPContext,
    y_observed: np.ndarray,
    observed_s: float,
    config: BioRSPConfig,
    n_perm: int,
) -> Tuple[float, float, float]:
    """Compute permutation p-value for spatial score.

    Shuffles foreground labels while preserving the number of foreground cells,
    recomputing the spatial score under the null hypothesis.
    """
    from biorsp.core.engine import compute_rsp_radar

    rng = np.random.default_rng(config.seed)
    n_cells = len(y_observed)

    if context.stratify_labels is not None:
        unique_labels = np.unique(context.stratify_labels)
        strata_indices = [np.where(context.stratify_labels == label)[0] for label in unique_labels]
    else:
        strata_indices = [np.arange(n_cells)]

    null_scores = []
    for _ in range(n_perm):
        y_perm = np.copy(y_observed)
        for idxs in strata_indices:
            perm_idxs = rng.permutation(idxs)
            y_perm[idxs] = y_perm[perm_idxs]

        radar_null = compute_rsp_radar(
            context.r_norm,
            context.theta,
            y_perm,
            config=config,
            sector_indices=context.sector_indices,
            sector_sort_indices=context.sector_sort_indices,
        )

        geom_mask = radar_null.geom_supported_mask
        if geom_mask is None or not np.any(geom_mask):
            s_null = 0.0
        else:
            valid_rsp = np.nan_to_num(radar_null.rsp[geom_mask], nan=0.0)
            w = radar_null.sector_weights[geom_mask]
            sum_w = np.sum(w)
            s_null = float(np.sqrt(np.sum(w * valid_rsp**2) / sum_w)) if sum_w > 0 else 0.0

        null_scores.append(s_null)

    null_scores = np.array(null_scores)
    p_val = (1 + np.sum(null_scores >= observed_s)) / (1 + n_perm)
    null_mean = float(np.mean(null_scores))
    null_sd = float(np.std(null_scores))

    return p_val, null_mean, null_sd
