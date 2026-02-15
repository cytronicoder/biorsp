"""Geometry helpers and theta conventions for BioRSP."""

from __future__ import annotations

import numpy as np


def _validate_xy(embedding_xy: np.ndarray) -> np.ndarray:
    arr = np.asarray(embedding_xy, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"embedding_xy must have shape (N, 2+), received {arr.shape}.")
    if arr.shape[0] < 3:
        raise ValueError("embedding_xy must include at least 3 rows.")
    if not np.isfinite(arr[:, :2]).all():
        raise ValueError("embedding_xy contains NaN/inf values.")
    return arr[:, :2]


def compute_vantage_point(
    embedding_xy: np.ndarray,
    method: str = "median",
) -> np.ndarray:
    """Compute vantage point from 2D embedding coordinates."""
    xy = _validate_xy(embedding_xy)
    method_norm = str(method).strip().lower()
    if method_norm == "median":
        out = np.median(xy, axis=0)
    elif method_norm == "mean":
        out = np.mean(xy, axis=0)
    else:
        raise ValueError("Unsupported center method. Use 'median' or 'mean'.")
    return np.asarray(out, dtype=float)


def compute_theta(
    embedding_xy: np.ndarray,
    center_xy: np.ndarray,
) -> np.ndarray:
    """Compute theta using UMAP-aligned convention.

    theta = mod(atan2(dy, dx), 2*pi)
    with 0 at +x (East) and counterclockwise increase.
    """
    xy = _validate_xy(embedding_xy)
    ctr = np.asarray(center_xy, dtype=float).ravel()
    if ctr.size != 2 or not np.isfinite(ctr).all():
        raise ValueError("center_xy must be a finite pair.")
    dx = xy[:, 0] - ctr[0]
    dy = xy[:, 1] - ctr[1]
    theta = np.mod(np.arctan2(dy, dx), 2.0 * np.pi)
    # Guard floating-point edge case where wrapped values hit 2*pi exactly.
    theta = np.where(np.isclose(theta, 2.0 * np.pi, atol=1e-12), 0.0, theta)
    return validate_theta(theta)


def validate_theta(theta: np.ndarray) -> np.ndarray:
    arr = np.asarray(theta, dtype=float).ravel()
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("theta must be a non-empty 1D array.")
    if not np.isfinite(arr).all():
        raise ValueError("theta contains NaN/inf values.")
    if float(arr.min()) < 0.0:
        raise ValueError("theta must lie in [0, 2*pi).")
    if float(arr.max()) >= 2.0 * np.pi:
        arr = np.where(np.isclose(arr, 2.0 * np.pi, atol=1e-12), 0.0, arr)
    if float(arr.max()) >= 2.0 * np.pi:
        raise ValueError("theta must lie in [0, 2*pi).")
    return arr


def bin_theta(theta: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Return theta bin edges and integer bin assignments."""
    arr = validate_theta(theta)
    n_bins = int(bins)
    if n_bins <= 0:
        raise ValueError("bins must be positive.")
    edges = np.linspace(0.0, 2.0 * np.pi, n_bins + 1, endpoint=True)
    bin_id = np.digitize(arr, edges, right=False) - 1
    bin_id = np.where(bin_id == n_bins, n_bins - 1, bin_id)
    return edges, bin_id.astype(np.int32)


def theta_bin_centers(bins: int) -> np.ndarray:
    n_bins = int(bins)
    if n_bins <= 0:
        raise ValueError("bins must be positive.")
    edges = np.linspace(0.0, 2.0 * np.pi, n_bins + 1, endpoint=True)
    return ((edges[:-1] + edges[1:]) / 2.0).astype(float)
