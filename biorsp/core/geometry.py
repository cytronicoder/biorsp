"""Geometry module for BioRSP.

Implements geometric definitions and coordinate transformations:
- Geometric median vantage point
- Polar coordinate transformation
- Canonical polar preparation (prepare_polar)
- Wrapped angular distance on S1
- Efficient sliding window angular indexing
"""

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np


@dataclass
class PolarPrep:
    """Canonical polar preparation output.

    Attributes:
        theta: (N,) array of angles in radians.
        r_norm: (N,) array of normalized radii in [0, 1].
        vantage_xy: (2,) vantage point used for centering.
        norm_stats: Dict of normalization metadata (scale, eps, and flags).
    """

    theta: np.ndarray
    r_norm: np.ndarray
    vantage_xy: np.ndarray
    norm_stats: Dict[str, float]


def geometric_median(
    points: np.ndarray, tol: float = 1e-5, max_iter: int = 100
) -> Tuple[np.ndarray, int, bool]:
    r"""Compute the geometric median of a set of 2D points using Weiszfeld's algorithm.

    The geometric median v minimizes the sum of Euclidean distances to the points z_i:
        v = argmin_v \sum_i ||z_i - v||_2

    Parameters
    ----------
    points : np.ndarray
        (N, 2) array of coordinates z_i.
    tol : float, optional
        Convergence tolerance, by default 1e-5.
    max_iter : int, optional
        Maximum number of iterations, by default 100.

    Returns
    -------
    v : np.ndarray
        (2,) array representing the vantage point v.
    n_iter : int
        Number of iterations performed.
    converged : bool
        Boolean indicating if convergence was reached.

    """

    y = np.mean(points, axis=0)
    converged = False
    n_iter = 0

    for i in range(max_iter):
        n_iter = i + 1
        distances = np.linalg.norm(points - y, axis=1)

        non_zeros = distances > 1e-10

        if not np.any(non_zeros):
            converged = True
            return y, n_iter, converged

        inv_distances = 1.0 / distances[non_zeros]
        weights = inv_distances / np.sum(inv_distances)

        y_next = np.sum(points[non_zeros] * weights[:, np.newaxis], axis=0)

        if np.linalg.norm(y_next - y) < tol:
            converged = True
            return y_next, n_iter, converged

        y = y_next

    return y, n_iter, converged


def compute_vantage(
    coords: np.ndarray,
    method: Literal["geometric_median", "mean"] = "geometric_median",
    tol: float = 1e-5,
    max_iter: int = 100,
    knn_k: int = 15,
    density_percentile: float = 5.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Compute the vantage point for polar transformation.

    Parameters
    ----------
    coords : np.ndarray
        (N, 2) array of coordinates.
    method : str, optional
        Vantage method ("geometric_median" or "mean"), by default "geometric_median".
    tol : float, optional
        Convergence tolerance for geometric median, by default 1e-5.
    max_iter : int, optional
        Maximum iterations for geometric median, by default 100.
    knn_k : int, optional
        k for kNN density check, by default 15.
    density_percentile : float, optional
        Percentile threshold for density check, by default 5.0.
    seed : int, optional
        Seed for the random sampling during density check, by default None.

    Returns
    -------
    np.ndarray
        (2,) array of vantage coordinates.

    """
    if method == "mean":
        return np.mean(coords, axis=0)

    v, _, _ = geometric_median(coords, tol=tol, max_iter=max_iter)

    dists_from_center = np.linalg.norm(coords - v, axis=1)
    k = min(knn_k, len(coords) - 1)
    if k <= 0:
        return v

    v_knn_dist = np.partition(dists_from_center, k)[k]

    rng = np.random.default_rng(seed)
    sample_size = min(1000, len(coords))
    sample_indices = rng.choice(len(coords), sample_size, replace=False)

    sample_knn_dists = []
    for i in sample_indices:
        dists = np.linalg.norm(coords - coords[i], axis=1)
        sample_knn_dists.append(np.partition(dists, k)[k])

    threshold = np.percentile(sample_knn_dists, 100 - density_percentile)

    if v_knn_dist > threshold:
        v = coords[np.argmin(dists_from_center)]

    return v


def polar_coordinates(z: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute polar coordinates relative to vantage point v.

    Parameters
    ----------
    z : np.ndarray
        (N, 2) array of cell coordinates z_i.
    v : np.ndarray
        (2,) array of vantage point coordinates.

    Returns
    -------
    r : np.ndarray
        (N,) array of radial distances ||z_i - v||_2.
    theta : np.ndarray
        (N,) array of angles in [-pi, pi).

    """
    centered = z - v
    r = np.linalg.norm(centered, axis=1)
    theta = np.arctan2(centered[:, 1], centered[:, 0])
    return r, theta


def prepare_polar(
    embedding: np.ndarray,
    *,
    seed: int,
    vantage: Literal["median", "centroid", "random", "fixed"],
    fixed_vantage: Optional[np.ndarray],
    radius_norm: Literal["max", "quantile", "std"],
    radius_q: float = 0.99,
) -> PolarPrep:
    """Prepare canonical polar coordinates from a 2D embedding.

    The embedding is centered by a deterministic vantage point, then converted
    to polar coordinates. Radii are normalized to [0, 1] with a stable scale.

    Args:
        embedding: (N, 2) array of coordinates.
        seed: Random seed for reproducible selection (random vantage).
        vantage: Vantage strategy ("median", "centroid", "random", "fixed").
        fixed_vantage: Optional fixed vantage point used when ``vantage="fixed"``.
        radius_norm: Radial normalization mode ("max", "quantile", "std").
        radius_q: Quantile for radius scaling when ``radius_norm="quantile"``.

    Returns:
        PolarPrep with normalized radii, angles, and normalization metadata.
    """
    if embedding.ndim != 2 or embedding.shape[1] != 2:
        raise ValueError("Embedding must be an (N, 2) array.")

    n_cells = embedding.shape[0]
    eps = 1e-8
    if n_cells == 0:
        return PolarPrep(
            theta=np.array([]),
            r_norm=np.array([]),
            vantage_xy=np.array([0.0, 0.0]),
            norm_stats={
                "scale": 0.0,
                "eps": eps,
                "degenerate": True,
                "reason": "empty_embedding",
            },
        )

    if vantage == "median":
        vantage_xy, _, _ = geometric_median(embedding)
    elif vantage == "centroid":
        vantage_xy = np.mean(embedding, axis=0)
    elif vantage == "random":
        rng = np.random.default_rng(seed)
        vantage_xy = embedding[int(rng.integers(0, n_cells))]
    elif vantage == "fixed":
        if fixed_vantage is None:
            raise ValueError("fixed_vantage must be provided when vantage='fixed'.")
        vantage_xy = np.asarray(fixed_vantage, dtype=float)
    else:
        raise ValueError(f"Unknown vantage mode: {vantage}")

    r, theta = polar_coordinates(embedding, vantage_xy)

    if not np.all(np.isfinite(r)):
        raise ValueError("Embedding contains non-finite values after centering.")

    max_r = float(np.max(r)) if r.size else 0.0
    std_r = float(np.std(r)) if r.size else 0.0
    q_r = float(np.quantile(r, radius_q)) if r.size else 0.0

    if max_r <= eps:
        r_norm = np.zeros_like(r)
        theta = np.zeros_like(r)
        return PolarPrep(
            theta=theta,
            r_norm=r_norm,
            vantage_xy=vantage_xy,
            norm_stats={
                "scale": 0.0,
                "eps": eps,
                "degenerate": True,
                "reason": "all_points_identical",
                "max_r": max_r,
                "std_r": std_r,
                "q_r": q_r,
                "radius_norm": radius_norm,
            },
        )

    if radius_norm == "max":
        scale = max_r
    elif radius_norm == "quantile":
        scale = q_r if q_r > eps else max_r
    elif radius_norm == "std":
        scale = std_r * 3.0 if std_r > eps else max_r
    else:
        raise ValueError(f"Unknown radius normalization: {radius_norm}")

    if scale <= eps:
        r_norm = np.zeros_like(r)
        theta = np.zeros_like(r)
        degenerate = True
    else:
        r_norm = np.clip(r / (scale + eps), 0.0, 1.0)
        theta = theta.copy()
        degenerate = False

    norm_stats = {
        "scale": float(scale),
        "eps": eps,
        "degenerate": degenerate,
        "max_r": max_r,
        "std_r": std_r,
        "q_r": q_r,
        "radius_norm": radius_norm,
        "radius_q": float(radius_q),
    }

    return PolarPrep(theta=theta, r_norm=r_norm, vantage_xy=vantage_xy, norm_stats=norm_stats)


def wrapped_circular_distance(alpha: np.ndarray, beta: float) -> np.ndarray:
    """Compute wrapped angular distance on the unit circle S1.

    dist_S1(alpha, beta) = min_k |alpha - beta + 2*pi*k|

    Parameters
    ----------
    alpha : np.ndarray
        Array of angles (radians).
    beta : float
        Reference angle (radians).

    Returns
    -------
    np.ndarray
        Array of angular distances in [0, pi].

    """
    diff = alpha - beta
    diff = (diff + np.pi) % (2 * np.pi) - np.pi
    return np.abs(diff)


def angle_grid(n_sectors: int) -> np.ndarray:
    """Generate equally spaced angular grid on [-pi, pi).

    Parameters
    ----------
    n_sectors : int
        Number of grid points.

    Returns
    -------
    np.ndarray
        (n_sectors,) array of angles.

    """
    return np.linspace(-np.pi, np.pi, n_sectors, endpoint=False)


def get_sector_indices(
    theta: np.ndarray,
    n_sectors: int,
    delta_deg: float,
) -> List[np.ndarray]:
    """Compute cell indices for each angular sector using an efficient sliding window.

    Parameters
    ----------
    theta : np.ndarray
        (N,) array of angles in radians.
    n_sectors : int
        Number of sectors.
    delta_deg : float
        Window width in degrees.

    Returns
    -------
    List[np.ndarray]
        List of length n_sectors, where each element is an array of indices into `theta`.

    """
    two_pi = 2 * np.pi
    delta_rad = np.deg2rad(delta_deg)
    half_width = delta_rad / 2.0

    theta_mod = theta % two_pi
    order = np.argsort(theta_mod)
    theta_sorted = theta_mod[order]

    theta2 = np.concatenate([theta_sorted, theta_sorted + two_pi])
    idx2 = np.concatenate([order, order])

    centers = angle_grid(n_sectors)
    centers_mod = (centers + two_pi) % two_pi

    centers_use = np.where((centers_mod - half_width) < 0, centers_mod + two_pi, centers_mod)
    centers_order = np.argsort(centers_use)

    sector_indices = [np.array([], dtype=int) for _ in range(n_sectors)]
    left = 0
    right = 0
    n2 = len(theta2)

    for b_idx in centers_order:
        phi_use = centers_use[b_idx]
        start = phi_use - half_width
        end = phi_use + half_width

        while left < n2 and theta2[left] < start:
            left += 1
        if right < left:
            right = left
        while right < n2 and theta2[right] <= end:
            right += 1

        if right > left:
            window_idx = idx2[left:right]
            sector_indices[b_idx] = np.unique(window_idx)

    return sector_indices


__all__ = [
    "PolarPrep",
    "compute_vantage",
    "geometric_median",
    "polar_coordinates",
    "prepare_polar",
    "wrapped_circular_distance",
    "angle_grid",
    "get_sector_indices",
]
