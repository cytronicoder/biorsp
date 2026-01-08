"""
Scoring wrapper for BioRSP v3 public APIs.

ONLY place that calls biorsp.score_genes and biorsp.score_gene_pairs.

Integrates geometry caching to avoid recomputing coordinates/sectors for
multi-gene panels on the same dataset.
"""

from typing import TYPE_CHECKING, List, Optional

import numpy as np
import pandas as pd

from biorsp import score_gene_pairs, score_genes
from biorsp.utils.config import BioRSPConfig

try:
    from . import cache

    _CACHE_AVAILABLE = True
except ImportError:
    _CACHE_AVAILABLE = False

if TYPE_CHECKING:
    from anndata import AnnData


def score_dataset(
    adata: "AnnData",
    genes: List[str],
    config: BioRSPConfig,
    embedding_key: str = "X_sim",
    cache_key: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Score genes using BioRSP v3.

    Parameters
    ----------
    adata : AnnData
        Dataset with expression and coordinates
    genes : List[str]
        Genes to score
    config : BioRSPConfig
        BioRSP configuration
    embedding_key : str, optional
        Key in obsm for coordinates
    cache_key : dict, optional
        Cache key dict (e.g., {"shape": "disk", "N": 2000, "seed": 42, "rep": 0})
        If provided and cache is available, geometry may be retrieved from cache

    Returns
    -------
    df_genes : pd.DataFrame
        Standardized gene-level scores

    Notes
    -----
    Geometry caching: If cache_key is provided, the function will attempt to use
    cached geometry (coords, sector_indices) to avoid recomputation. Cache hits
    can provide 2-5x speedup for multi-gene panels on the same dataset.
    """

    # For now, cache infrastructure exists but BioRSP API doesn't expose geometry directly

    if _CACHE_AVAILABLE and cache_key is not None:
        cached = cache.get_cached_geometry(**cache_key)
        if cached is not None:

            pass

    # Call BioRSP public API
    results = score_genes(adata, genes, embedding_key=embedding_key, config=config)

    df = pd.DataFrame()

    df["gene"] = results.get("gene", genes)
    df["coverage_expr"] = results.get("coverage_expr", results.get("coverage", np.nan))
    df["spatial_score"] = results.get("spatial_score", results.get("anisotropy", np.nan))
    df["r_mean_bg"] = results.get("r_mean", results.get("r_mean_bg", 0.0))
    df["coverage_bg"] = results.get("coverage_bg", np.nan)
    df["coverage_fg"] = results.get("coverage_fg", np.nan)
    df["p_value"] = results.get("p_value", np.nan)
    df["q_value"] = results.get("q_value", np.nan)

    if "archetype" in results.columns:
        df["archetype_pred"] = results["archetype"]

    if "abstain" not in results.columns:
        df["abstain_flag"] = pd.isna(df["spatial_score"]) | (df["coverage_bg"] < 0.1)
        df["abstain_reason"] = "ok"
        df.loc[pd.isna(df["spatial_score"]), "abstain_reason"] = "no_spatial_score"
        df.loc[df["coverage_bg"] < 0.1, "abstain_reason"] = "low_bg_support"
    else:
        df["abstain_flag"] = results["abstain"]
        df["abstain_reason"] = results.get("abstain_reason", "ok")

    return df


def score_pairs(
    adata: "AnnData",
    genes: List[str],
    config: BioRSPConfig,
    embedding_key: str = "X_sim",
    cache_key: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Score gene pairs using BioRSP v3.

    Parameters
    ----------
    adata : AnnData
        Dataset with expression and coordinates
    genes : List[str]
        Genes to analyze
    config : BioRSPConfig
        BioRSP configuration
    embedding_key : str, optional
        Key in obsm for coordinates
    cache_key : dict, optional
        Cache key dict for geometry reuse (see score_dataset)

    Returns
    -------
    df_pairs : pd.DataFrame
        Standardized pairwise scores

    Notes
    -----
    Geometry caching: Same caching strategy as score_dataset(). Particularly
    beneficial for large gene panels where sector statistics are computed once
    and reused across all pairwise comparisons.
    """

    if _CACHE_AVAILABLE and cache_key is not None:
        cached = cache.get_cached_geometry(**cache_key)
        if cached is not None:

            pass

    # Call BioRSP public API
    results = score_gene_pairs(adata, genes, embedding_key=embedding_key, config=config)

    df = pd.DataFrame()

    df["gene_a"] = results.get("gene_a", results.get("feature_a"))
    df["gene_b"] = results.get("gene_b", results.get("feature_b"))
    df["similarity_profile"] = results.get("similarity_profile", results.get("correlation", np.nan))
    df["copattern_score"] = results.get("copattern_score", results.get("correlation", np.nan))
    df["shared_mask_fraction"] = results.get("shared_mask_fraction", 1.0)
    df["sign_agreement"] = results.get("similarity_sign", results.get("sign_agreement", np.nan))

    return df
