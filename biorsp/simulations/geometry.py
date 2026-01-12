"""
Geometry computations for spatial coordinates.

Provides polar coordinate calculation and density estimation.
"""

from typing import Literal, Tuple

import numpy as np


def compute_polar(
    coords: np.ndarray, center: Literal["median", "mean", "centroid"] = "median"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute polar coordinates from Cartesian coordinates.

    Parameters
    ----------
    coords : np.ndarray
        Cartesian coordinates (n, 2)
    center : str, optional
        Center computation method: 'median', 'mean', or 'centroid'

    Returns
    -------
    vantage : np.ndarray
        Center point (2,)
    r : np.ndarray
        Radial distances (n,)
    theta : np.ndarray
        Angular coordinates in radians, range [-pi, pi] (n,)
    """
    if center == "median":
        vantage = np.median(coords, axis=0)
    elif center == "mean":
        vantage = np.mean(coords, axis=0)
    elif center == "centroid":
        vantage = np.median(coords, axis=0)
    else:
        raise ValueError(f"Unknown center method: {center}")

    rel = coords - vantage
    r = np.linalg.norm(rel, axis=1)
    theta = np.arctan2(rel[:, 1], rel[:, 0])

    return vantage, r, theta


def radial_density_proxy(r: np.ndarray) -> np.ndarray:
    """
    Fast density proxy based on radial rank.

    Parameters
    ----------
    r : np.ndarray
        Radial distances

    Returns
    -------
    density : np.ndarray
        Density proxy (high values = dense regions)
    """

    r_rank = np.argsort(np.argsort(r))
    density = 1.0 / (r_rank + 1.0)
    return density / density.mean()
