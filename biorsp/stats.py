"""
General statistical utilities for BioRSP.
"""

from typing import Optional, Tuple

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


def compute_localization(
    R: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    eps: float = 1e-12,
    method: str = "entropy",
) -> Tuple[float, dict]:
    """
    Compute localization index for a radar profile.

    Quantifies how concentrated the absolute radar energy is in a small subset of angles.
    L near 0: diffuse/global structure.
    L near 1: strongly localized structure.

    Parameters
    ----------
    R : np.ndarray
        Radar profile (sector statistics).
    valid_mask : np.ndarray, optional
        Boolean mask of valid sectors. If None, uses non-NaN values.
    eps : float, optional
        Small constant for numerical stability.
    method : str, optional
        Metric to use: "entropy" (default) or "gini".

    Returns
    -------
    float
        Localization index L.
    dict
        Diagnostic information (M, sum_abs, entropy, status, etc.).
    """
    R = np.asarray(R)
    if valid_mask is None:
        valid_mask = np.isfinite(R)

    R_valid = R[valid_mask]
    M = len(R_valid)

    info = {
        "M": M,
        "sum_abs": 0.0,
        "status": "ok",
        "entropy": np.nan,
        "gini": np.nan,
    }

    if M <= 1:
        info["status"] = "insufficient_sectors"
        return np.nan, info

    a = np.abs(R_valid)
    sum_a = np.sum(a)
    info["sum_abs"] = float(sum_a)

    if sum_a < eps:
        info["status"] = "no_signal"
        return 0.0, info

    p = a / (sum_a + eps)

    # Entropy-based localization
    # H = -sum(p * log(p))
    # L = 1 - H / log(M)
    h = -np.sum(p * np.log(p + eps))
    info["entropy"] = float(h)
    L_entropy = 1.0 - h / np.log(M)

    # Gini-based localization (optional alternative)
    # G = (sum_{i=1}^M sum_{j=1}^M |a_i - a_j|) / (2 * M^2 * mean(a))
    diff_matrix = np.abs(a[:, None] - a[None, :])
    gini = np.sum(diff_matrix) / (2 * M**2 * np.mean(a))
    info["gini"] = float(gini)

    if method == "gini":
        return float(gini), info

    return float(L_entropy), info


__all__ = [
    "bh_fdr",
    "compute_localization",
]
