"""
Geometry module for BioRSP.

Implements geometric definitions from the Methods section:
- Geometric median vantage point
- Polar coordinate transformation
- Wrapped angular distance on S1
"""

from typing import Literal, Tuple

import numpy as np


def geometric_median(
    points: np.ndarray, tol: float = 1e-5, max_iter: int = 100
) -> Tuple[np.ndarray, int, bool]:
    """
    Compute the geometric median of a set of 2D points using Weiszfeld's algorithm.

    The geometric median v minimizes the sum of Euclidean distances to the points z_i:
        v = argmin_v \\sum_i ||z_i - v||_2

    Args:
        points: (N, 2) array of coordinates z_i.
        tol: Convergence tolerance for the iterative procedure.
        max_iter: Maximum number of iterations.

    Returns:
        v: (2,) array representing the vantage point v.
        n_iter: Number of iterations performed.
        converged: Boolean indicating if convergence was reached.
    """
    # Initial guess: centroid (mean)
    y = np.mean(points, axis=0)
    converged = False
    n_iter = 0

    for i in range(max_iter):
        n_iter = i + 1
        # Calculate distances from current guess
        # shape (N,)
        distances = np.linalg.norm(points - y, axis=1)

        # Handle points coinciding with current guess to avoid division by zero
        # Weiszfeld algorithm modification for points at the median
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

    Default uses the geometric median:
        v = argmin_v \\sum_i ||z_i - v||_2

    Args:
        coords: (N, 2) array of coordinates.
        method: Vantage method ("geometric_median" or "mean").
        tol: Convergence tolerance for geometric median.
        max_iter: Maximum iterations for geometric median.

    Returns:
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

    Args:
        z: (N, 2) array of cell coordinates z_i.
        v: (2,) array of vantage point coordinates.

    Returns:
        r: (N,) array of radial distances ||z_i - v||_2.
        theta: (N,) array of angles in [-pi, pi).
    """
    # Center points relative to vantage
    centered = z - v

    # r_i = ||z_i - v||_2
    r = np.linalg.norm(centered, axis=1)

    # theta_i = atan2(y - v_y, x - v_x)
    theta = np.arctan2(centered[:, 1], centered[:, 0])

    return r, theta


def wrapped_circular_distance(alpha: np.ndarray, beta: float) -> np.ndarray:
    """
    Compute wrapped angular distance on the unit circle S1.

    dist_S1(alpha, beta) = min_k |alpha - beta + 2*pi*k|

    Args:
        alpha: Array of angles (radians).
        beta: Reference angle (radians).

    Returns:
        Array of angular distances in [0, pi].
    """
    # Calculate difference in [-pi, pi]
    diff = alpha - beta
    # Wrap to [-pi, pi]
    diff = (diff + np.pi) % (2 * np.pi) - np.pi
    # Return absolute difference
    return np.abs(diff)


def angle_grid(B: int) -> np.ndarray:
    """
    Generate equally spaced angular grid on [-pi, pi).

    Args:
        B: Number of grid points.

    Returns:
        (B,) array of angles.
    """
    return np.linspace(-np.pi, np.pi, B, endpoint=False)


__all__ = [
    "compute_vantage",
    "geometric_median",
    "polar_coordinates",
    "wrapped_circular_distance",
    "angle_grid",
]
