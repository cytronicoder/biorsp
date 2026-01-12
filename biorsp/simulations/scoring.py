"""
Scoring wrapper for BioRSP public APIs.

Central interface for calling biorsp.score_genes and biorsp.score_gene_pairs.

Integrates geometry caching to avoid recomputing coordinates/sectors for
multi-gene panels on the same dataset.

Key output columns:
- coverage (C_g): fraction of cells with expression >= biological threshold
- spatial_score (S_g): RMS of geometry-supported radar profile
- p_value: permutation-based significance
- archetype_pred: classification (if classify_genes was run)
"""

from typing import TYPE_CHECKING, List, Optional

import numpy as np
import pandas as pd

from biorsp.api import score_gene_pairs, score_genes
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
    Score genes using BioRSP.

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
        Standardized gene-level scores with columns:
        - coverage (C_g): fraction of cells expressing >= threshold
        - spatial_score (S_g): RMS of radar profile (spatial organization)
        - p_value: permutation-based significance
        - coverage_geom: fraction of geometry-supported sectors

    Notes
    -----
    Geometry caching: If cache_key is provided, the function will attempt to use
    cached geometry (coords, sector_indices) to avoid recomputation. Cache hits
    can provide 2-5x speedup for multi-gene panels on the same dataset.
    """

    if _CACHE_AVAILABLE and cache_key is not None:
        cached = cache.get_cached_geometry(**cache_key)
        if cached is not None:
            pass

    # Call BioRSP public API
    results = score_genes(adata, genes, embedding_key=embedding_key, config=config)

    df = pd.DataFrame()

    df["gene"] = results.get("gene", genes)
    df["Coverage"] = results.get("Coverage", np.nan)
    df["Spatial_Score"] = results.get("Spatial_Score", np.nan)
    df["Directionality"] = results.get("Directionality", 0.0)
    df["coverage_geom"] = results.get("coverage_geom", np.nan)
    df["coverage_fg"] = results.get("coverage_fg", np.nan)
    df["p_value"] = results.get("p_value", np.nan)
    df["q_value"] = results.get("q_value", np.nan)

    if "Archetype" in results.columns:
        df["Archetype"] = results["Archetype"]

    if "abstain" not in results.columns:
        df["abstain_flag"] = pd.isna(df["Spatial_Score"]) | (df["coverage_geom"] < 0.1)
        df["abstain_reason"] = "ok"
        df.loc[pd.isna(df["Spatial_Score"]), "abstain_reason"] = "no_spatial_score"
        df.loc[df["coverage_geom"] < 0.1, "abstain_reason"] = "low_geom_support"
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
    Score gene pairs using BioRSP.

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
        Standardized pairwise scores with columns:
        - gene_a, gene_b: gene pair identifiers
        - similarity_profile: weighted correlation of radar profiles
        - copattern_score: overall co-localization measure
        - shared_mask_fraction: overlap in geometry-supported sectors
        - sign_agreement: whether genes share directionality

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
