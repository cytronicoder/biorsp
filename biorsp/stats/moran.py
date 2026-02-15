"""Moran's I spatial autocorrelation statistics."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def extract_weights(adata) -> sp.spmatrix:
    if "connectivities" not in adata.obsp:
        raise KeyError("adata.obsp['connectivities'] is required for Moran's I.")
    w = adata.obsp["connectivities"]
    if not sp.issparse(w):
        raise TypeError("adata.obsp['connectivities'] must be sparse.")
    w = w.tocsr(copy=True)
    row_sum = np.asarray(w.sum(axis=1)).ravel()
    scale = np.zeros_like(row_sum, dtype=float)
    nz = row_sum > 0
    scale[nz] = 1.0 / row_sum[nz]
    if np.any(nz):
        w = sp.diags(scale).dot(w)
    return w


def _row_standardize(w: sp.spmatrix) -> sp.spmatrix:
    w_csr = w.tocsr(copy=True)
    row_sum = np.asarray(w_csr.sum(axis=1)).ravel()
    scale = np.zeros_like(row_sum, dtype=float)
    nz = row_sum > 0
    scale[nz] = 1.0 / row_sum[nz]
    if np.any(nz):
        w_csr = sp.diags(scale).dot(w_csr)
    return w_csr


def morans_i(x: np.ndarray, w: sp.spmatrix, row_standardize: bool = True) -> float:
    x_arr = np.asarray(x, dtype=float).ravel()
    if x_arr.ndim != 1 or x_arr.size == 0:
        raise ValueError("x must be non-empty 1D.")
    if not np.isfinite(x_arr).all():
        raise ValueError("x contains NaN/inf.")
    if not sp.issparse(w):
        raise TypeError("w must be sparse.")

    w_use = _row_standardize(w) if row_standardize else w.tocsr(copy=False)
    n = int(x_arr.size)
    if w_use.shape != (n, n):
        raise ValueError("w shape mismatch with x length.")

    z = x_arr - float(np.mean(x_arr))
    den = float(np.sum(z**2))
    if den <= 0.0:
        raise ValueError("Variance of x is zero; Moran's I undefined.")

    s0 = float(w_use.sum())
    if s0 <= 0.0:
        raise ValueError("Sum of weights is zero; Moran's I undefined.")

    wz = w_use.dot(z)
    num = float(np.dot(z, wz))
    return float((n / s0) * (num / den))
