"""
Foreground definition module for BioRSP.

Implements foreground selection logic from the Methods section:
- Binary foreground via quantile thresholding
- Optional soft foreground weighting
"""

from typing import Tuple

import numpy as np


def binary_foreground(
    x: np.ndarray,
    quantile: float = 0.90,
) -> Tuple[np.ndarray, float, float]:
    """
    Define binary foreground indicator y_i^(g) = 1(x_i^(g) > t_g).

    t_g = Q_{0.90}({x_i^(g) : i in S})

    Args:
        x: (N,) array of expression values x_i^(g).
        quantile: Quantile for threshold (default 0.90).

    Returns:
        y: (N,) boolean array where True indicates foreground.
        threshold: The computed threshold t_g.
        coverage: Realized foreground fraction c_g.
    """
    # Compute threshold t_g = Q_{0.90}
    threshold = np.quantile(x, quantile)

    # Strict inequality: x_i > t_g
    y = x > threshold

    # Realized coverage c_g = (1/|S|) * sum(y_i)
    coverage = np.mean(y)

    return y, threshold, coverage


def soft_foreground_weights(
    x: np.ndarray,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, float, float]:
    """
    Compute continuous foreground weights (optional robustness mode).

    w_i^(g) = sigma((x_i^(g) - mu_g) / s_g)

    where:
    - sigma is the logistic function
    - mu_g is the median expression
    - s_g is the median absolute deviation + epsilon

    Args:
        x: (N,) array of expression values.
        eps: Small constant to avoid division by zero.

    Returns:
        w: (N,) array of weights in (0, 1).
        mu: Median expression.
        s: Scaled MAD.
    """
    # Median mu_g
    mu = np.median(x)

    # Median absolute deviation
    mad = np.median(np.abs(x - mu))
    s = mad + eps

    # Z-score-like term
    z = (x - mu) / s

    # Logistic function sigma(z) = 1 / (1 + exp(-z))
    w = 1.0 / (1.0 + np.exp(-z))

    return w, mu, s


def check_gene_power(coverage: float) -> bool:
    """
    Check if gene is underpowered based on coverage.

    Args:
        coverage: Realized foreground fraction.

    Returns:
        True if gene is adequately powered (coverage > 0), False otherwise.
    """
    return coverage > 0


__all__ = ["binary_foreground", "soft_foreground_weights", "check_gene_power"]
