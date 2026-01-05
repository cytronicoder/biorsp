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

    h = -np.sum(p * np.log(p + eps))
    info["entropy"] = float(h)
    L_entropy = 1.0 - h / np.log(M)

    diff_matrix = np.abs(a[:, None] - a[None, :])
    gini = np.sum(diff_matrix) / (2 * M**2 * np.mean(a))
    info["gini"] = float(gini)

    if method == "gini":
        return float(gini), info

    return float(L_entropy), info


def compute_signed_summaries(
    R: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    eps: float = 1e-12,
) -> dict:
    """
    Compute signed summary statistics for a radar profile.

    Distinguishes between core (proximal) and rim (distal) patterns.
    R_mean > 0: overall core bias.
    R_mean < 0: overall rim bias.
    Polarity near +/- 1: consistently one-signed.
    Polarity near 0: mixed or localized.

    Parameters
    ----------
    R : np.ndarray
        Radar profile (sector statistics).
    valid_mask : np.ndarray, optional
        Boolean mask of valid sectors. If None, uses non-NaN values.
    eps : float, optional
        Small constant for numerical stability.

    Returns
    -------
    dict
        Dictionary containing:
        - R_mean: Mean signed shift.
        - R_median: Median signed shift.
        - polarity: Signed energy ratio.
        - A_signed: Signed anisotropy (sign(R_mean) * RMS).
        - frac_pos: Fraction of positive sectors.
        - frac_neg: Fraction of negative sectors.
        - M_valid: Number of valid sectors.
        - status: Status of computation.
    """
    R = np.asarray(R)
    if valid_mask is None:
        valid_mask = np.isfinite(R)

    R_valid = R[valid_mask]
    M = len(R_valid)

    res = {
        "R_mean": np.nan,
        "R_median": np.nan,
        "polarity": np.nan,
        "A_signed": np.nan,
        "frac_pos": 0.0,
        "frac_neg": 0.0,
        "M_valid": M,
        "status": "ok",
    }

    if M < 1:
        res["status"] = "no_valid_sectors"
        return res

    if M < 2:
        # M < 2: metrics like polarity are less meaningful, but report available statistics
        res["status"] = "insufficient_sectors"

    r_mean = np.mean(R_valid)
    r_median = np.median(R_valid)
    sum_r = np.sum(R_valid)
    sum_abs_r = np.sum(np.abs(R_valid))
    rms = np.sqrt(np.mean(R_valid**2))

    res["R_mean"] = float(r_mean)
    res["R_median"] = float(r_median)
    res["A_signed"] = float(np.sign(r_mean) * rms)

    if sum_abs_r < eps:
        res["polarity"] = 0.0
        res["status"] = "no_signal"
    else:
        res["polarity"] = float(sum_r / (sum_abs_r + eps))

    res["frac_pos"] = float(np.sum(R_valid > 0) / M)
    res["frac_neg"] = float(np.sum(R_valid < 0) / M)

    return res


__all__ = [
    "bh_fdr",
    "compute_localization",
    "compute_signed_summaries",
]
