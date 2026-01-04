"""
Utility functions for BioRSP.
"""

from typing import Optional

import numpy as np


def weighted_wasserstein_1d(
    values_a: np.ndarray,
    weights_a: np.ndarray,
    values_b: np.ndarray,
    weights_b: np.ndarray,
) -> float:
    """
    Compute 1D Wasserstein-1 distance between two weighted samples.

    EMD = integral |CDF_u(t) - CDF_v(t)| dt

    Args:
        values_a: (N_a,) array of values.
        weights_a: (N_a,) array of weights.
        values_b: (N_b,) array of values.
        weights_b: (N_b,) array of weights.

    Returns:
        w1: Wasserstein-1 distance.
    """
    if values_a.size == 0 or values_b.size == 0:
        return np.nan

    # Normalize weights
    sum_a = np.sum(weights_a)
    sum_b = np.sum(weights_b)
    if sum_a <= 0 or sum_b <= 0:
        return np.nan

    u = weights_a / sum_a
    v = weights_b / sum_b

    # Combine and sort all values
    all_values = np.concatenate([values_a, values_b])
    all_weights_u = np.concatenate([u, np.zeros_like(v)])
    all_weights_v = np.concatenate([np.zeros_like(u), v])

    sort_idx = np.argsort(all_values)
    all_values = all_values[sort_idx]
    all_weights_u = all_weights_u[sort_idx]
    all_weights_v = all_weights_v[sort_idx]

    # Compute CDFs
    cdf_u = np.cumsum(all_weights_u)
    cdf_v = np.cumsum(all_weights_v)

    # Compute integral |CDF_u - CDF_v| dt
    diff_values = np.diff(all_values)
    w1 = np.sum(np.abs(cdf_u[:-1] - cdf_v[:-1]) * diff_values)

    return float(w1)


def weighted_quantile(
    values: np.ndarray, weights: np.ndarray, q: float, rng: Optional[np.random.Generator] = None
) -> float:
    """
    Compute weighted quantile with deterministic tie-handling.

    Args:
        values: (N,) array of values.
        weights: (N,) array of weights.
        q: Quantile in [0, 1].
        rng: Optional random generator for tie-breaking.

    Returns:
        The weighted quantile.
    """
    if values.size == 0:
        return np.nan
    order = np.argsort(values)
    return weighted_quantile_sorted(values[order], weights[order], q)


def weighted_quantile_sorted(
    values_sorted: np.ndarray, weights_sorted: np.ndarray, q: float
) -> float:
    """
    Compute weighted quantile from pre-sorted values.

    Args:
        values_sorted: (N,) array of values, sorted ascending.
        weights_sorted: (N,) array of weights corresponding to values_sorted.
        q: Quantile in [0, 1].

    Returns:
        The weighted quantile.
    """
    if values_sorted.size == 0:
        return np.nan
    sum_w = np.sum(weights_sorted)
    if sum_w <= 0:
        return np.nan

    # If weights are binary, use standard quantile for exact match with scipy/numpy
    if np.all((weights_sorted == 0) | (weights_sorted == 1)):
        target_values = values_sorted[weights_sorted == 1]
        if target_values.size == 0:
            return np.nan
        # np.quantile is still O(N) but faster than full sort if we already have sorted input
        # Actually, for sorted input, we can just pick the element.
        n = target_values.size
        idx = q * (n - 1)
        i = int(idx)
        f = idx - i
        if i == n - 1:
            return float(target_values[-1])
        return float(target_values[i] * (1 - f) + target_values[i + 1] * f)

    cdf = np.cumsum(weights_sorted).astype(float)
    cdf -= 0.5 * weights_sorted
    cdf /= sum_w

    return float(np.interp(q, cdf, values_sorted))


def weighted_mad(values: np.ndarray, weights: np.ndarray, scale_factor: float = 1.4826) -> float:
    """
    Compute weighted Median Absolute Deviation (MAD).

    Args:
        values: (N,) array of values.
        weights: (N,) array of weights.
        scale_factor: Factor to match normal distribution (default 1.4826).

    Returns:
        The weighted MAD.
    """
    if values.size == 0:
        return np.nan
    med = weighted_quantile(values, weights, 0.5)
    if np.isnan(med):
        return np.nan
    abs_dev = np.abs(values - med)
    mad = weighted_quantile(abs_dev, weights, 0.5)
    return float(scale_factor * mad)


def bh_fdr(p_values: np.ndarray) -> np.ndarray:
    """
    Benjamini-Hochberg FDR correction.

    Args:
        p_values: Array of p-values (NaNs allowed).

    Returns:
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
    "weighted_wasserstein_1d",
    "weighted_quantile",
    "bh_fdr",
]
