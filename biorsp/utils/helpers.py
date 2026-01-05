"""
Utility functions for BioRSP.
"""

import numpy as np


def _as_1d(a) -> np.ndarray:
    """Ensure input is a 1D numpy array."""
    return np.asarray(a).reshape(-1)


def weighted_wasserstein_1d(
    values_a: np.ndarray,
    weights_a: np.ndarray,
    values_b: np.ndarray,
    weights_b: np.ndarray,
) -> float:
    r"""
    Compute 1D Wasserstein-1 distance between two weighted samples.

    The 1D Wasserstein-1 distance (Earth Mover's Distance) is defined as:
    $$W_1(P, Q) = \int_{-\infty}^{\infty} |F_P(t) - F_Q(t)| dt$$
    where $F_P$ and $F_Q$ are the cumulative distribution functions of $P$ and $Q$.

    Parameters
    ----------
    values_a : np.ndarray
        (N_a,) array of values for distribution A.
    weights_a : np.ndarray
        (N_a,) array of weights for distribution A.
    values_b : np.ndarray
        (N_b,) array of values for distribution B.
    weights_b : np.ndarray
        (N_b,) array of weights for distribution B.

    Returns
    -------
    float
        The Wasserstein-1 distance. Returns NaN if either input is empty or has zero total weight.
    """
    if values_a.size == 0 or values_b.size == 0:
        return np.nan

    # Normalize weights
    sum_a = np.sum(weights_a)
    sum_b = np.sum(weights_b)
    if sum_a <= 0 or sum_b <= 0:
        return np.nan

    all_values = np.concatenate([values_a, values_b])
    u = weights_a / sum_a
    v = weights_b / sum_b

    all_weights_u = np.concatenate([u, np.zeros(values_b.size)])
    all_weights_v = np.concatenate([np.zeros(values_a.size), v])

    sort_idx = np.argsort(all_values)
    all_values = all_values[sort_idx]
    all_weights_u = all_weights_u[sort_idx]
    all_weights_v = all_weights_v[sort_idx]

    cdf_u = np.cumsum(all_weights_u)
    cdf_v = np.cumsum(all_weights_v)

    diff_values = np.diff(all_values)
    w1 = np.sum(np.abs(cdf_u[:-1] - cdf_v[:-1]) * diff_values)

    return float(w1)


def weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    q: float,
) -> float:
    """
    Compute weighted quantile with deterministic tie-handling.

    Parameters
    ----------
    values : np.ndarray
        (N,) array of values.
    weights : np.ndarray
        (N,) array of weights.
    q : float
        Quantile in [0, 1].

    Returns
    -------
    float
        The weighted quantile. Returns NaN if input is empty or has zero total weight.
    """
    if values.size == 0:
        return np.nan
    order = np.argsort(values)
    return weighted_quantile_sorted(values[order], weights[order], q)


def weighted_quantile_sorted(
    values_sorted: np.ndarray, weights_sorted: np.ndarray, q: float
) -> float:
    r"""
    Compute weighted quantile from pre-sorted values.

    Uses the definition:
    $$Q(q) = \inf \{x : F(x) \geq q\}$$
    where $F(x)$ is the weighted CDF. We use linear interpolation of the CDF
    shifted by half-weights for smoothness, matching R's type 7 quantile
    behavior for unweighted data.

    Parameters
    ----------
    values_sorted : np.ndarray
        (N,) array of values, sorted ascending.
    weights_sorted : np.ndarray
        (N,) array of weights corresponding to values_sorted.
    q : float
        Quantile in [0, 1].

    Returns
    -------
    float
        The weighted quantile. Returns NaN if input is empty or has zero total weight.
    """
    if values_sorted.size == 0:
        return np.nan
    sum_w = np.sum(weights_sorted)
    if sum_w <= 0:
        return np.nan

    # If weights are binary, use standard quantile for exact match with scipy/numpy
    # This is faster and more robust for the common binary case.
    if np.all((weights_sorted == 0) | (weights_sorted == 1)):
        target_values = values_sorted[weights_sorted == 1]
        if target_values.size == 0:
            return np.nan
        n = target_values.size
        if n == 1:
            return float(target_values[0])
        idx = q * (n - 1)
        i = int(idx)
        f = idx - i
        if i >= n - 1:
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


def compute_sector_weight(
    nF: float,
    nB: float,
    mode: str = "none",
    k: float = 5.0,
) -> float:
    """
    Compute a support-based weight for a sector.

    Parameters
    ----------
    nF : float
        Foreground support (count or mass).
    nB : float
        Background support (count or mass).
    mode : str, optional
        Weighting mode: "none", "sqrt_frac", "effective_min", "logistic_support".
        By default "none".
    k : float, optional
        Tunable parameter for "effective_min" or "logistic_support", by default 5.0.

    Returns
    -------
    float
        Weight in [0, 1].
    """
    if mode == "none":
        return 1.0

    if nF <= 0 or nB <= 0:
        return 0.0

    if mode == "sqrt_frac":
        return float(np.sqrt(nF / (nF + nB)))

    if mode == "effective_min":
        m = min(nF, nB)
        return float(m / (m + k))

    if mode == "logistic_support":
        m = min(nF, nB)
        z = (m - k) / (max(k, 1e-8) / 4.0)
        if z > 100:
            return 1.0
        if z < -100:
            return 0.0
        return float(1.0 / (1.0 + np.exp(-z)))

    return 1.0


__all__ = [
    "weighted_wasserstein_1d",
    "weighted_quantile",
    "bh_fdr",
    "compute_sector_weight",
]
