"""Circular peak utilities for BioRSP profile shape analysis.

This module provides deterministic circular local-max detection and a simple,
reproducible prominence calculation.
"""

from __future__ import annotations

import numpy as np


def _as_1d_float(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError("x must contain at least one value.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("x must be finite.")
    return arr


def circular_local_maxima(x: np.ndarray) -> np.ndarray:
    """Return indices of circular local maxima.

    A local maximum at index ``i`` satisfies:
    ``x[i] > x[i-1]`` and ``x[i] >= x[i+1]`` (with wrap-around).
    """
    arr = _as_1d_float(x)
    n = int(arr.size)
    if n < 3:
        return np.zeros(0, dtype=int)

    left = np.roll(arr, 1)
    right = np.roll(arr, -1)
    mask = (arr > left) & (arr >= right)
    return np.flatnonzero(mask).astype(int)


def _is_local_min(arr: np.ndarray, idx: int) -> bool:
    n = arr.size
    left = arr[(idx - 1) % n]
    mid = arr[idx]
    right = arr[(idx + 1) % n]
    return bool((mid < left and mid <= right) or (mid <= left and mid < right))


def _nearest_local_min_value(arr: np.ndarray, peak_idx: int, step_sign: int) -> float:
    n = arr.size
    for step in range(1, n):
        idx = (peak_idx + step_sign * step) % n
        if _is_local_min(arr, idx):
            return float(arr[idx])
    return float(np.min(arr))


def circular_peak_prominences(
    x: np.ndarray,
    peak_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Compute circular peak prominences for selected peaks.

    Prominence for a peak at ``p`` is:
    ``x[p] - max(nearest_left_min, nearest_right_min)``.
    """
    arr = _as_1d_float(x)
    if peak_indices is None:
        peaks = circular_local_maxima(arr)
    else:
        peaks = np.asarray(peak_indices, dtype=int).ravel()

    if peaks.size == 0:
        return np.zeros(0, dtype=float)

    n = arr.size
    if np.any(peaks < 0) or np.any(peaks >= n):
        raise ValueError("peak_indices out of range.")

    prom = np.zeros(peaks.size, dtype=float)
    for i, p in enumerate(peaks.tolist()):
        left_min = _nearest_local_min_value(arr, int(p), -1)
        right_min = _nearest_local_min_value(arr, int(p), +1)
        base = max(left_min, right_min)
        prom[i] = max(0.0, float(arr[int(p)] - base))
    return prom


def find_circular_peaks(
    x: np.ndarray,
    prominence_threshold: float = 0.0,
) -> dict[str, np.ndarray]:
    """Find circular peaks and return indices/prominences/heights.

    Args:
        x: 1D circular profile.
        prominence_threshold: minimum prominence for retained peaks.

    Returns:
        Dict with ``indices``, ``prominences``, and ``heights`` arrays.
    """
    arr = _as_1d_float(x)
    peaks = circular_local_maxima(arr)
    prom = circular_peak_prominences(arr, peaks)
    keep = prom >= float(prominence_threshold)
    peaks_kept = peaks[keep]
    prom_kept = prom[keep]
    heights = arr[peaks_kept] if peaks_kept.size else np.zeros(0, dtype=float)
    return {
        "indices": peaks_kept.astype(int),
        "prominences": prom_kept.astype(float),
        "heights": np.asarray(heights, dtype=float),
    }
