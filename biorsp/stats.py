"""
General statistical utilities for BioRSP.
"""

import numpy as np


def bh_fdr(p_values: np.ndarray) -> np.ndarray:
    """
    Benjamini-Hochberg FDR correction.

    Parameters
    ----------
    p_values : np.ndarray
        Array of p-values (NaNs allowed).

    Returns
    -------
    np.ndarray
        Array of q-values with NaNs preserved.
    """
    p_values = np.asarray(p_values, dtype=float)
    q_values = np.full_like(p_values, np.nan, dtype=float)

    finite_mask = np.isfinite(p_values)
    if not np.any(finite_mask):
        return q_values

    pvals = p_values[finite_mask]
    order = np.argsort(pvals)
    ranked = pvals[order]
    n = len(ranked)
    q = ranked * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)

    q_values[finite_mask] = q[np.argsort(order)]
    return q_values


__all__ = [
    "bh_fdr",
]
