"""
Geometry module for BioRSP.

Implements geometric definitions and coordinate transformations:
- Geometric median vantage point
- Polar coordinate transformation
- Wrapped angular distance on S1
- Efficient sliding window angular indexing
"""

from typing import List, Literal, Tuple

import numpy as np


def geometric_median(
    points: np.ndarray, tol: float = 1e-5, max_iter: int = 100
) -> Tuple[np.ndarray, int, bool]:
    """
    Compute the geometric median of a set of 2D points using Weiszfeld's algorithm.

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
    # Initial guess: centroid (mean)
    y = np.mean(points, axis=0)
    converged = False
    n_iter = 0

    for i in range(max_iter):
        n_iter = i + 1
        # Calculate distances from current guess
        distances = np.linalg.norm(points - y, axis=1)

        # Handle points coinciding with current guess to avoid division by zero
        non_zeros = distances > 1e-10

        if not np.any(non_zeros):
            converged = True
            return y, n_iter, converged

        # Weights w_i = 1 / ||z_i - y||
        inv_distances = 1.0 / distances[non_zeros]
        weights = inv_distances / np.sum(inv_distances)

        # Update y as weighted average: y_{k+1} = \sum w_i z_i
        y_next = np.sum(points[non_zeros] * weights[:, np.newaxis], axis=0)

        # Check convergence
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
) -> np.ndarray:
    """
    Compute the vantage point for polar transformation.

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

    Returns
    -------
    np.ndarray
        (2,) array of vantage coordinates.
    """
    if method == "geometric_median":
        v, _, _ = geometric_median(coords, tol=tol, max_iter=max_iter)
        return v
    if method == "mean":
        return np.mean(coords, axis=0)
    raise ValueError(f"Unknown vantage method: {method}")


def polar_coordinates(z: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute polar coordinates relative to vantage point v.

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


def wrapped_circular_distance(alpha: np.ndarray, beta: float) -> np.ndarray:
    """
    Compute wrapped angular distance on the unit circle S1.

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


def angle_grid(B: int) -> np.ndarray:
    """
    Generate equally spaced angular grid on [-pi, pi).

    Parameters
    ----------
    B : int
        Number of grid points.

    Returns
    -------
    np.ndarray
        (B,) array of angles.
    """
    return np.linspace(-np.pi, np.pi, B, endpoint=False)


def get_sector_indices(
    theta: np.ndarray,
    B: int,
    delta_deg: float,
) -> List[np.ndarray]:
    """
    Compute cell indices for each angular sector using an efficient sliding window.

    Parameters
    ----------
    theta : np.ndarray
        (N,) array of angles in radians.
    B : int
        Number of sectors.
    delta_deg : float
        Window width in degrees.

    Returns
    -------
    List[np.ndarray]
        List of length B, where each element is an array of indices into `theta`.
    """
    two_pi = 2 * np.pi
    delta_rad = np.deg2rad(delta_deg)
    half_width = delta_rad / 2.0

    # 1. Normalize angles to [0, 2pi)
    theta_mod = theta % two_pi
    order = np.argsort(theta_mod)
    theta_sorted = theta_mod[order]

    # 2. Duplicate for wrap-around handling
    theta2 = np.concatenate([theta_sorted, theta_sorted + two_pi])
    idx2 = np.concatenate([order, order])

    # 3. Define grid centers in [0, 2pi)
    centers = angle_grid(B)
    centers_mod = (centers + two_pi) % two_pi

    # 4. Map centers to monotonic space for two-pointer
    # If (phi - half_width) < 0, we shift it to (2pi, 3pi)
    centers_use = np.where((centers_mod - half_width) < 0, centers_mod + two_pi, centers_mod)
    centers_order = np.argsort(centers_use)

    sector_indices = [np.array([], dtype=int) for _ in range(B)]
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
            # Use unique to handle cases where delta > 360 (though unlikely)
            # and to ensure indices are sorted for downstream efficiency
            window_idx = idx2[left:right]
            sector_indices[b_idx] = np.unique(window_idx)

    return sector_indices


__all__ = [
    "compute_vantage",
    "geometric_median",
    "polar_coordinates",
    "wrapped_circular_distance",
    "angle_grid",
    "get_sector_indices",
]
