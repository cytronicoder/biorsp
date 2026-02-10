"""Geometry helpers for BioRSP."""

from __future__ import annotations

import numpy as np


def _validate_umap(umap_xy: np.ndarray) -> np.ndarray:
    """Validate a UMAP array of shape (N, 2) with finite values.

    Args:
        umap_xy: UMAP coordinate array.

    Returns:
        Validated UMAP array.

    Raises:
        ValueError: If array has incorrect shape or contains NaN/inf values.
    """
    if umap_xy is None:
        raise ValueError("UMAP coordinates are required but not found.")
    arr = np.asarray(umap_xy)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"UMAP array must have shape (N, 2); received {arr.shape}.")
    if arr.shape[0] < 3:
        raise ValueError(
            "UMAP array must contain at least 3 cells for angle computation."
        )
    if not np.isfinite(arr).all():
        raise ValueError("UMAP coordinates contain NaN or infinite values.")
    return arr


def compute_vantage(umap_xy: np.ndarray) -> np.ndarray:
    """Compute a vantage point as the per-axis median of UMAP coordinates.

    Args:
        umap_xy: Array of shape (N, 2).

    Returns:
        Array of shape (2,) containing median x and y.
    """
    arr = _validate_umap(umap_xy)
    median_xy = np.median(arr, axis=0)
    return median_xy.astype(float)


def compute_angles(umap_xy: np.ndarray, vantage_xy: np.ndarray) -> np.ndarray:
    """Compute angles from the vantage point to each cell in [0, 2π).

    Args:
        umap_xy: Array of shape (N, 2).
        vantage_xy: Array-like of shape (2,).

    Returns:
        Angles in radians, shape (N,).
    """
    arr = _validate_umap(umap_xy)
    v = np.asarray(vantage_xy, dtype=float).ravel()
    if v.size != 2 or not np.isfinite(v).all():
        raise ValueError("vantage_xy must be a finite array-like of length 2.")

    dx = arr[:, 0] - v[0]
    dy = arr[:, 1] - v[1]
    angles = np.arctan2(dy, dx)
    angles = (angles + 2 * np.pi) % (2 * np.pi)

    if angles.min(initial=0.0) < -1e-12:
        raise ValueError("Angles computation produced values below 0.")
    if angles.max(initial=0.0) >= 2 * np.pi + 1e-12:
        raise ValueError("Angles computation produced values outside [0, 2π).")

    return angles.astype(float)


def validate_angles(angles: np.ndarray) -> np.ndarray:
    """Lightweight validation for angles array.

    Args:
        angles: Array of angles in radians.

    Returns:
        Validated angles array.

    Raises:
        ValueError: If angles are not in [0, 2π) or contain NaN/inf.
    """
    arr = np.asarray(angles, dtype=float).ravel()
    if arr.ndim != 1:
        raise ValueError("angles must be a 1D array.")
    if arr.size == 0:
        raise ValueError("angles array is empty.")
    if not np.isfinite(arr).all():
        raise ValueError("angles contain NaN or infinite values.")
    if arr.min(initial=0.0) < 0 or arr.max(initial=0.0) >= 2 * np.pi:
        raise ValueError("angles must be within [0, 2π).")
    return arr
