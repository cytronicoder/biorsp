"""Small pure helpers for core computations."""

from __future__ import annotations

import numpy as np


def finite_1d(name: str, values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must be finite.")
    return arr


def circular_argmax(values: np.ndarray) -> int:
    arr = finite_1d("values", values)
    return int(np.argmax(arr))
