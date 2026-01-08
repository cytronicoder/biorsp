"""
Density estimation for confounded null generation.

Provides KDE and fast approximations for spatial density.
"""

import numpy as np


def kde_density(coords: np.ndarray, bandwidth: float = 0.3) -> np.ndarray:
    """
    Kernel density estimation using Gaussian kernel.

    Parameters
    ----------
    coords : np.ndarray
        Coordinates (n, 2)
    bandwidth : float, optional
        KDE bandwidth

    Returns
    -------
    density : np.ndarray
        Estimated density at each point (n,)
    """
    n = len(coords)
    density = np.zeros(n)

    for i in range(n):
        dists = np.linalg.norm(coords - coords[i], axis=1)
        kernel = np.exp(-0.5 * (dists / bandwidth) ** 2)
        density[i] = np.sum(kernel)

    # Normalize
    density = density / density.mean()
    return density


def knn_density(coords: np.ndarray, k: int = 20) -> np.ndarray:
    """
    Fast kNN-based density estimation.

    Parameters
    ----------
    coords : np.ndarray
        Coordinates (n, 2)
    k : int, optional
        Number of nearest neighbors

    Returns
    -------
    density : np.ndarray
        Density proxy (inverse of mean distance to k nearest neighbors)
    """
    try:
        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(n_neighbors=min(k, len(coords) - 1))
        nn.fit(coords)
        dists, _ = nn.kneighbors(coords)
        density = 1.0 / (np.mean(dists, axis=1) + 1e-6)
        return density / density.mean()
    except ImportError:
        # Fallback: use simple distance to centroid
        center = np.median(coords, axis=0)
        r = np.linalg.norm(coords - center, axis=1)
        density = 1.0 / (r + 1.0)
        return density / density.mean()
