"""Confidence interval helpers shared across benchmarks."""

from __future__ import annotations

from typing import Callable, Iterable

import numpy as np


def binomial_wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Parameters
    ----------
    k : int
        Number of successes.
    n : int
        Number of trials.
    alpha : float
        Two-sided error rate (e.g., 0.05 for 95% CI).
    """

    if n <= 0:
        raise ValueError("n must be positive for binomial CI")
    if k < 0 or k > n:
        raise ValueError("k must be within [0, n]")

    z = scipy_norm_quantile(1 - alpha / 2)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def bootstrap_ci(
    values: Iterable[float],
    func: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int | None = None,
) -> tuple[float, float]:
    """Non-parametric bootstrap confidence interval.

    Parameters
    ----------
    values : Iterable[float]
        Sample values.
    func : callable
        Statistic to bootstrap (defaults to mean).
    n_boot : int
        Number of bootstrap replicates.
    alpha : float
        Two-sided error rate.
    seed : int, optional
        Random seed for reproducibility.
    """

    arr = np.asarray(list(values), dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        raise ValueError("bootstrap_ci requires at least one finite value")

    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot, dtype=float)
    n = arr.size
    for i in range(n_boot):
        sample = rng.choice(arr, size=n, replace=True)
        stats[i] = func(sample)

    lower = np.quantile(stats, alpha / 2)
    upper = np.quantile(stats, 1 - alpha / 2)
    return float(lower), float(upper)


def scipy_norm_quantile(q: float) -> float:
    """Small helper to avoid importing scipy globally."""

    try:
        from scipy.stats import norm

        return float(norm.ppf(q))
    except Exception as exc:  # pragma: no cover - fallback path
        raise ImportError("scipy is required for norm quantiles; please install scipy") from exc
