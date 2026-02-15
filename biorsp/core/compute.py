"""Core RSP computation functions (no plotting, no filesystem I/O)."""

from __future__ import annotations

from typing import Any

import numpy as np

from biorsp.core.geometry import (
    bin_theta,
    compute_theta,
    compute_vantage_point,
    theta_bin_centers,
    validate_theta,
)
from biorsp.core.types import RSPConfig, RSPResult
from biorsp.core.utils import finite_1d


def compute_rsp_profile_from_boolean(
    foreground: np.ndarray,
    theta: np.ndarray,
    bins: int,
    *,
    bin_id: np.ndarray | None = None,
    bin_counts_total: np.ndarray | None = None,
) -> tuple[np.ndarray, float, float, np.ndarray]:
    """Compute R(theta)=pF-pB from a boolean foreground mask."""
    f = np.asarray(foreground, dtype=bool).ravel()
    th = validate_theta(theta)
    if f.size != th.size:
        raise ValueError("foreground and theta must have same length.")

    n_bins = int(bins)
    if n_bins <= 0:
        raise ValueError("bins must be positive.")

    n_fg = int(f.sum())
    n_bg = int(f.size - n_fg)
    if n_fg == 0 or n_bg == 0:
        raise ValueError("RSP undefined when all or no cells are foreground.")

    if bin_id is None:
        _, b = bin_theta(th, n_bins)
        total = np.bincount(b, minlength=n_bins).astype(float)
    else:
        b = np.asarray(bin_id, dtype=np.int32).ravel()
        if b.size != f.size:
            raise ValueError("bin_id length mismatch.")
        if np.any(b < 0) or np.any(b >= n_bins):
            raise ValueError("bin_id values outside [0, bins).")
        if bin_counts_total is None:
            total = np.bincount(b, minlength=n_bins).astype(float)
        else:
            total = np.asarray(bin_counts_total, dtype=float).ravel()
            if total.size != n_bins:
                raise ValueError("bin_counts_total length must equal bins.")

    fg_counts = np.bincount(b[f], minlength=n_bins).astype(float)
    bg_counts = total - fg_counts
    p_fg = fg_counts / float(n_fg)
    p_bg = bg_counts / float(n_bg)
    r_theta = p_fg - p_bg

    centers = theta_bin_centers(n_bins)
    idx_max = int(np.argmax(r_theta))
    phi_max = float(centers[idx_max])
    e_max = float(np.max(r_theta))
    return r_theta.astype(float), phi_max, e_max, centers


def compute_rsp_profile(
    expr: np.ndarray,
    theta: np.ndarray,
    bins: int,
    *,
    threshold: float = 0.0,
    bin_id: np.ndarray | None = None,
    bin_counts_total: np.ndarray | None = None,
) -> tuple[np.ndarray, float, float, np.ndarray]:
    x = finite_1d("expr", expr)
    foreground = x > float(threshold)
    return compute_rsp_profile_from_boolean(
        foreground,
        theta,
        bins,
        bin_id=bin_id,
        bin_counts_total=bin_counts_total,
    )


def _peak_directions(theta_centers: np.ndarray, r_theta: np.ndarray, frac: float = 0.5) -> np.ndarray:
    arr = np.asarray(r_theta, dtype=float).ravel()
    th = np.asarray(theta_centers, dtype=float).ravel()
    if arr.size < 3:
        return th[:1]
    threshold = float(np.max(arr) * float(frac))
    peaks: list[float] = []
    for i in range(arr.size):
        left = arr[(i - 1) % arr.size]
        right = arr[(i + 1) % arr.size]
        if arr[i] >= threshold and arr[i] > left and arr[i] > right:
            peaks.append(float(th[i]))
    if not peaks:
        peaks = [float(th[int(np.argmax(arr))])]
    return np.asarray(peaks, dtype=float)


def compute_rsp(
    *,
    expr: np.ndarray,
    embedding_xy: np.ndarray,
    config: RSPConfig,
    center_xy: np.ndarray | None = None,
    feature_label: str | None = None,
    feature_index: int | None = None,
) -> RSPResult:
    """Compute RSPResult from expression and embedding."""
    emb = np.asarray(embedding_xy, dtype=float)
    if emb.ndim != 2 or emb.shape[1] < 2:
        raise ValueError(f"embedding_xy must have shape (N, 2+), got {emb.shape}.")

    if center_xy is None:
        center = compute_vantage_point(emb[:, :2], method=config.center_method)
    else:
        center = np.asarray(center_xy, dtype=float).ravel()
        if center.size != 2 or not np.isfinite(center).all():
            raise ValueError("center_xy must be a finite pair.")

    theta = compute_theta(emb[:, :2], center)
    r_theta, phi_max, e_max, theta_centers = compute_rsp_profile(
        expr,
        theta,
        config.bins,
        threshold=config.threshold,
    )

    anisotropy = float(np.max(np.abs(r_theta)))
    peak_dirs = _peak_directions(theta_centers, r_theta)
    breadth = float(np.mean(r_theta >= (0.5 * np.max(r_theta))))
    coverage = float(np.mean(r_theta > 0.0))

    label = feature_label or config.feature_label or "feature"
    meta: dict[str, Any] = {
        "basis": config.basis,
        "bins": int(config.bins),
        "center_method": str(config.center_method),
        "threshold": float(config.threshold),
        "center_xy": [float(center[0]), float(center[1])],
        "n_cells": int(emb.shape[0]),
    }
    return RSPResult(
        theta=np.asarray(theta_centers, dtype=float),
        R_theta=np.asarray(r_theta, dtype=float),
        anisotropy=anisotropy,
        peak_direction=float(phi_max),
        peak_directions=peak_dirs,
        breadth=breadth,
        coverage=coverage,
        E_max=float(e_max),
        feature_label=str(label),
        feature_index=(None if feature_index is None else int(feature_index)),
        metadata=meta,
    )
