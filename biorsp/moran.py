"""Moran's I spatial autocorrelation baseline."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def extract_weights(adata) -> sp.spmatrix:
    """Extract row-standardized connectivity weights from AnnData.

    Args:
        adata: AnnData object with obsp["connectivities"].

    Returns:
        Row-standardized CSR matrix.
    """
    if "connectivities" not in adata.obsp:
        raise KeyError("adata.obsp['connectivities'] is required for Moran's I.")
    W = adata.obsp["connectivities"]
    if not sp.issparse(W):
        raise TypeError("adata.obsp['connectivities'] must be a sparse matrix.")
    W = W.tocsr(copy=True)
    row_sums = np.asarray(W.sum(axis=1)).ravel()
    scale = np.zeros_like(row_sums, dtype=float)
    nonzero = row_sums > 0
    scale[nonzero] = 1.0 / row_sums[nonzero]
    if np.any(nonzero):
        D_inv = sp.diags(scale)
        W = D_inv.dot(W)
    return W


def _row_standardize(W: sp.spmatrix) -> sp.spmatrix:
    """Row-standardize a sparse matrix."""
    W_csr = W.tocsr(copy=True)
    row_sums = np.asarray(W_csr.sum(axis=1)).ravel()
    scale = np.zeros_like(row_sums, dtype=float)
    nonzero = row_sums > 0
    scale[nonzero] = 1.0 / row_sums[nonzero]
    if np.any(nonzero):
        D_inv = sp.diags(scale)
        W_csr = D_inv.dot(W_csr)
    return W_csr


def morans_i(x: np.ndarray, W: sp.spmatrix, row_standardize: bool = True) -> float:
    """Compute Moran's I spatial autocorrelation statistic.

    Args:
        x: Numeric vector of length N.
        W: Sparse weights matrix (N x N).
        row_standardize: Whether to row-standardize W before computing I.

    Returns:
        Moran's I value.
    """
    x_arr = np.asarray(x, dtype=float).ravel()
    if x_arr.ndim != 1:
        raise ValueError("x must be a 1D array.")
    if not np.isfinite(x_arr).all():
        raise ValueError("x contains NaN or infinite values.")

    if not sp.issparse(W):
        raise TypeError("W must be a scipy sparse matrix.")
    W_use = _row_standardize(W) if row_standardize else W.tocsr()

    N = x_arr.size
    if W_use.shape != (N, N):
        raise ValueError(f"W shape {W_use.shape} does not match x length {N}.")

    z = x_arr - x_arr.mean()
    den = float(np.sum(z**2))
    if den <= 0:
        raise ValueError("Variance of x is zero; Moran's I undefined.")

    S0 = float(W_use.sum())
    if S0 == 0:
        raise ValueError("Sum of weights S0 is zero; Moran's I undefined.")

    Wz = W_use.dot(z)
    num = float(z.dot(Wz))
    moran_i = (N / S0) * (num / den)
    return float(moran_i)
