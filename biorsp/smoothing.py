"""Smoothing helpers shared across BioRSP scoring/permutation code."""

from __future__ import annotations

import numpy as np


def circular_moving_average(x: np.ndarray, w: int) -> np.ndarray:
    """Circular moving average with odd window length ``w``.

    Args:
        x: 1D numeric array.
        w: Odd window size. ``w=1`` returns an unchanged copy.

    Returns:
        Smoothed array with the same length as ``x``.
    """

    arr = np.asarray(x, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError("x must contain at least one value.")
    if int(w) != w or int(w) < 1:
        raise ValueError("w must be an integer >= 1.")

    w_i = int(w)
    if w_i % 2 == 0:
        raise ValueError("w must be odd.")
    if w_i == 1 or arr.size == 1:
        return arr.copy()

    if w_i > arr.size:
        w_i = arr.size if arr.size % 2 == 1 else arr.size - 1
        if w_i <= 1:
            return arr.copy()

    k = w_i // 2
    padded = np.concatenate([arr[-k:], arr, arr[:k]])
    kernel = np.ones(w_i, dtype=float) / float(w_i)
    out = np.convolve(padded, kernel, mode="valid")
    if out.size != arr.size:
        raise AssertionError("circular_moving_average output length mismatch.")
    return out
