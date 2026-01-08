"""Preprocessing module for BioRSP.

Implements radial normalization and other data preparation steps.
"""

from typing import Tuple

import numpy as np
from scipy.stats import iqr


def normalize_radii(
    r: np.ndarray, eps: float = 1e-8, method: str = "robust_iqr"
) -> Tuple[np.ndarray, dict]:
    """Perform within-set robust radial normalization.

    r_hat_i = (r_i - median(r)) / (IQR(r) + eps)

    Parameters
    ----------
    r : np.ndarray
        (N,) array of radial distances.
    eps : float, optional
        Small constant to prevent division by zero, by default 1e-8.
    method : str, optional
        Normalization method (currently only "robust_iqr" supported), by default "robust_iqr".

    Returns
    -------
    r_hat : np.ndarray
        (N,) array of normalized radii.
    stats : dict
        Dictionary containing 'median_r', 'iqr_r', 'eps', and 'n_non_finite'.

    """
    if not np.all(np.isfinite(r)):
        n_non_finite = int(np.sum(~np.isfinite(r)))
        raise ValueError(f"Input radii contain {n_non_finite} non-finite values (NaN/inf).")

    median_r = np.median(r)
    iqr_r = iqr(r)

    if iqr_r < eps:
        mad = np.median(np.abs(r - median_r))
        iqr_r = 1.4826 * mad
        if iqr_r < eps:
            iqr_r = np.std(r)

    r_hat = (r - median_r) / (iqr_r + eps)

    stats = {
        "median_r": float(median_r),
        "iqr_r": float(iqr_r),
        "eps": float(eps),
        "n_non_finite": 0,
    }

    return r_hat, stats


__all__ = [
    "normalize_radii",
]
