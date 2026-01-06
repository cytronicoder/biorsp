"""
Public API for BioRSP.

This module provides the primary entry points for gene scoring and gene-pair analysis.
"""

from typing import List, Optional, Union

import pandas as pd
from anndata import AnnData

from biorsp.core.scoring import classify_genes_impl, score_gene_pairs_impl, score_genes_impl
from biorsp.utils.config import BioRSPConfig


def score_genes(
    adata: AnnData,
    genes: List[str],
    embedding_key: str = "X_umap",
    subset: Optional[Union[dict, str, List[str]]] = None,
    config: Optional[BioRSPConfig] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Score genes for coverage and spatial organization.

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
        GeneScoreTable with coverage and spatial scores.
    """
    if config is None:
        config = BioRSPConfig(**kwargs)
    elif kwargs:
        from dataclasses import fields, replace

        config_fields = {f.name for f in fields(BioRSPConfig)}
        config_overrides = {k: v for k, v in kwargs.items() if k in config_fields}
        if config_overrides:
            config = replace(config, **config_overrides)

    return score_genes_impl(adata, genes, embedding_key, subset, config)


def score_gene_pairs(
    adata: AnnData,
    genes: List[str],
    embedding_key: str = "X_umap",
    subset: Optional[Union[dict, str, List[str]]] = None,
    config: Optional[BioRSPConfig] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Score gene pairs for spatial relationship/copatterning.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    genes : List[str]
        List of genes to analyze.
    embedding_key : str, optional
        Key in adata.obsm storing the embedding, by default "X_umap".
    subset : Optional[Union[dict, str, List[str]]], optional
        Subset of cells to analyze, by default None.
    config : Optional[BioRSPConfig], optional
        BioRSP configuration, by default None.

    Returns
    -------
    pd.DataFrame
        GenePairTable with pairwise similarity metrics.
    """
    if config is None:
        config = BioRSPConfig(**kwargs)
    elif kwargs:
        from dataclasses import fields, replace

        config_fields = {f.name for f in fields(BioRSPConfig)}
        config_overrides = {k: v for k, v in kwargs.items() if k in config_fields}
        if config_overrides:
            config = replace(config, **config_overrides)

    return score_gene_pairs_impl(adata, genes, embedding_key, subset, config)


def classify_genes(
    gene_table: pd.DataFrame,
    c_cut: Optional[float] = None,
    s_cut: Optional[float] = None,
    fdr_cut: float = 0.05,
) -> pd.DataFrame:
    """
    Classify genes into archetypes based on coverage and spatial scores.

    Parameters
    ----------
    gene_table : pd.DataFrame
        Output from score_genes.
    c_cut : Optional[float], optional
        Coverage cutoff, by default 0.10.
    s_cut : Optional[float], optional
        Spatial score cutoff, by default determined by FDR or empirical null.
    fdr_cut : float, optional
        FDR cutoff for significance, by default 0.05.

    Returns
    -------
    pd.DataFrame
        Gene table with added 'archetype' column.
    """
    return classify_genes_impl(gene_table, c_cut, s_cut, fdr_cut)
