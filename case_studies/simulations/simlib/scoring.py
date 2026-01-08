"""
Scoring wrapper for BioRSP v3 public APIs.

ONLY place that calls biorsp.score_genes and biorsp.score_gene_pairs.
"""

from typing import TYPE_CHECKING, List

import numpy as np
import pandas as pd

from biorsp import score_gene_pairs, score_genes
from biorsp.utils.config import BioRSPConfig

if TYPE_CHECKING:
    from anndata import AnnData


def score_dataset(
    adata: "AnnData",
    genes: List[str],
    config: BioRSPConfig,
    embedding_key: str = "X_sim",
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

    Returns
    -------
    df_genes : pd.DataFrame
        Standardized gene-level scores
    """
    # Call BioRSP public API
    results = score_genes(adata, genes, embedding_key=embedding_key, config=config)

    # Standardize output columns
    df = pd.DataFrame()

    # Required columns
    df["gene"] = results.get("gene", genes)
    df["coverage_expr"] = results.get("coverage", np.nan)
    df["spatial_score"] = results.get("anisotropy", np.nan)  # v3 name
    df["r_mean_bg"] = results.get("r_mean", 0.0)
    df["coverage_bg"] = results.get("coverage_bg", np.nan)
    df["coverage_fg"] = results.get("coverage_fg", np.nan)
    df["p_value"] = results.get("p_value", np.nan)
    df["q_value"] = results.get("q_value", np.nan)

    # Optional columns
    if "archetype" in results.columns:
        df["archetype_pred"] = results["archetype"]

    # Synthesize abstain flag if not present
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

    Returns
    -------
    df_pairs : pd.DataFrame
        Standardized pairwise scores
    """
    # Call BioRSP public API
    results = score_gene_pairs(adata, genes, embedding_key=embedding_key, config=config)

    # Standardize output columns
    df = pd.DataFrame()

    df["gene_a"] = results.get("gene_a", results.get("feature_a"))
    df["gene_b"] = results.get("gene_b", results.get("feature_b"))
    df["similarity_profile"] = results.get("correlation", np.nan)
    df["copattern_score"] = results.get("correlation", np.nan)  # Alias
    df["shared_mask_fraction"] = results.get("shared_mask_fraction", 1.0)
    df["sign_agreement"] = results.get("sign_agreement", np.nan)

    return df
