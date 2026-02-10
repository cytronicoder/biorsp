"""Shared utilities for BioRSP workflows."""

from __future__ import annotations

import os
from typing import Iterable

import numpy as np
import scipy.sparse as sp


def ensure_dir(path: str) -> None:
    """Create a directory (and parents) if it does not exist."""
    if path == "":
        return
    os.makedirs(path, exist_ok=True)


def get_gene_vector(adata, gene: str) -> np.ndarray:
    """Extract a single-gene expression vector as a dense 1D numpy array.

    Args:
        adata: AnnData object.
        gene: Gene name to extract.

    Returns:
        Expression vector of length N cells.
    """
    if "var_names" not in adata.__dict__ and not hasattr(adata, "var_names"):
        raise ValueError("AnnData object missing var_names; cannot select genes.")
    if gene not in adata.var_names:
        raise KeyError(f"Gene '{gene}' not found in adata.var_names.")

    vec = adata[:, gene].X
    if sp.issparse(vec):
        vec_dense = vec.toarray().ravel()
    else:
        vec_dense = np.asarray(vec).ravel()

    if vec_dense.ndim != 1:
        # Handles cases where slicing returns shape (N, 1)
        vec_dense = vec_dense.reshape(-1)

    if vec_dense.size != adata.n_obs:
        raise ValueError(
            f"Gene vector length mismatch: expected {adata.n_obs}, got {vec_dense.size}."
        )
    if not np.all(np.isfinite(vec_dense)):
        raise ValueError(f"Expression vector for gene '{gene}' contains NaN/inf.")
    return vec_dense


def select_gene(adata, preferred: Iterable[str], fallback_index: int = 0) -> str:
    """Select the first available gene from a preference list, else fallback by index.

    Args:
        adata: AnnData object.
        preferred: Iterable of gene symbols to try in order.
        fallback_index: Index into adata.var_names to use if none of the preferred genes are present.

    Returns:
        Selected gene name.
    """
    var_names = list(adata.var_names)
    for gene in preferred:
        if gene in var_names:
            return gene
    if fallback_index < 0 or fallback_index >= len(var_names):
        raise IndexError(
            f"Fallback index {fallback_index} is out of bounds for var_names of length {len(var_names)}."
        )
    return var_names[fallback_index]
