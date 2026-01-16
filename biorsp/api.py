"""Public API for BioRSP.

This module provides the primary entry points for gene scoring and gene-pair analysis.
"""

from typing import List, Optional, Union

import numpy as np
import pandas as pd
from anndata import AnnData

from biorsp.core.scoring import classify_genes_impl, score_gene_pairs_impl, score_genes_impl
from biorsp.utils.config import BioRSPConfig

COLUMN_MAP = {
    "coverage": "Coverage",
    "spatial_score": "Spatial_Bias_Score",
    "r_mean": "Directionality",
    "archetype": "Archetype",
}
INTERNAL_MAP = {v: k for k, v in COLUMN_MAP.items()}
PUBLIC_COLUMNS = ["Coverage", "Spatial_Bias_Score", "Directionality", "Archetype"]
LEGACY_COLUMNS = {
    "coverage_expr",
    "pct_cells",
    "alpha",
    "anisotropy",
    "rms",
    "S_g",
    "class",
    "type",
}


def _standardize_gene_table(df: pd.DataFrame) -> pd.DataFrame:
    """Rename to public schema, drop legacy columns, and ensure required fields exist."""
    df = df.rename(columns=COLUMN_MAP)

    df = df.drop(columns=[c for c in LEGACY_COLUMNS if c in df.columns], errors="ignore")

    for col in PUBLIC_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    ordered = []
    if "gene" in df.columns:
        ordered.append("gene")
    ordered.extend([c for c in PUBLIC_COLUMNS if c not in ordered])

    remaining = [c for c in df.columns if c not in ordered]
    return df[ordered + remaining]


def score_genes(
    adata: AnnData,
    genes: List[str],
    embedding_key: str = "X_umap",
    subset: Optional[Union[dict, str, List[str]]] = None,
    config: Optional[BioRSPConfig] = None,
    **kwargs,
) -> pd.DataFrame:
    """Score genes for coverage and spatial organization.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    genes : List[str]
        List of genes to score.
    embedding_key : str, optional
        Key in adata.obsm storing the embedding, by default "X_umap".
    subset : Optional[Union[dict, str, List[str]]], optional
        Subset of cells to analyze (e.g., specific cell type), by default None.
    config : Optional[BioRSPConfig], optional
        BioRSP configuration, by default None.
    **kwargs
        Additional configuration parameters to override defaults.

    Returns
    -------
    pd.DataFrame
        GeneScoreTable with standardized columns: Coverage, Spatial_Bias_Score, Directionality.
    """
    if config is None:
        config = BioRSPConfig(**kwargs)
    elif kwargs:
        from dataclasses import fields, replace

        valid_fields = {f.name for f in fields(BioRSPConfig)}
        updates = {k: v for k, v in kwargs.items() if k in valid_fields}
        if updates:
            config = replace(config, **updates)

    df = score_genes_impl(adata, genes, embedding_key, subset, config)

    return _standardize_gene_table(df)


def classify_genes(
    df: pd.DataFrame,
    c_cut: Optional[float] = None,
    s_cut: Optional[float] = None,
    fdr_cut: float = 0.05,
) -> pd.DataFrame:
    """Classify genes into archetypes based on Coverage and Spatial_Bias_Score.

    Parameters
    ----------
    df : pd.DataFrame
        Result from score_genes().
    c_cut : float, optional
         Coverage cutoff. If None, defaults to 0.10.
    s_cut : float, optional
         Spatial Bias Score cutoff. If None, determined automatically.
    fdr_cut : float, optional
         FDR cutoff used if 'q_value' is present and s_cut is None.

    Returns
    -------
    pd.DataFrame
        DataFrame with an additional 'Archetype' column.
    """
    internal_df = df.rename(columns=INTERNAL_MAP)

    out = classify_genes_impl(internal_df, c_cut, s_cut, fdr_cut)

    return _standardize_gene_table(out)


def score_gene_pairs(
    adata: AnnData,
    genes: List[str],
    embedding_key: str,
    subset: Optional[Union[dict, str, List[str]]] = None,
    config: Optional[BioRSPConfig] = None,
    **kwargs,
) -> pd.DataFrame:
    """Score gene-gene co-patterns."""
    if config is None:
        config = BioRSPConfig(**kwargs)

    return score_gene_pairs_impl(adata, genes, embedding_key, subset, config)
