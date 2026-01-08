"""
Coordinate generators for embedding cluster footprints.

Implements various spatial manifolds for testing BioRSP on different geometries.
"""

from typing import Any, Dict, Tuple

import numpy as np
from numpy.random import Generator


def generate_coords(
    shape: str, n: int, rng: Generator, params: Dict[str, Any] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Generate 2D spatial coordinates for a given manifold shape.

    Parameters
    ----------
    shape : str
        Shape identifier: disk, ellipse, crescent, annulus, peanut, disconnected_blobs
    n : int
        Number of points
    rng : Generator
        Numpy random generator
    params : Dict[str, Any], optional
        Shape-specific parameters

    Returns
    -------
    coords : np.ndarray
        Coordinates (n, 2)
    meta : Dict[str, Any]
        Metadata including shape parameters
    """
    params = params or {}

    if shape == "disk":
        return _disk(n, rng, params)
    elif shape == "ellipse":
        return _ellipse(n, rng, params)
    elif shape == "crescent":
        return _crescent(n, rng, params)
    elif shape == "annulus":
        return _annulus(n, rng, params)
    elif shape == "peanut":
        return _peanut(n, rng, params)
    elif shape == "disconnected_blobs":
        return _disconnected_blobs(n, rng, params)
    else:
        raise ValueError(f"Unknown shape: {shape}")


def _disk(n: int, rng: Generator, params: Dict) -> Tuple[np.ndarray, Dict]:
    """Uniform disk."""
    R = params.get("radius", 1.0)
    r = np.sqrt(rng.uniform(0, 1, n)) * R
    theta = rng.uniform(0, 2 * np.pi, n)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    coords = np.column_stack([x, y])
    meta = {"shape": "disk", "radius": R}
    return coords, meta


def _ellipse(n: int, rng: Generator, params: Dict) -> Tuple[np.ndarray, Dict]:
    """Anisotropic ellipse."""
    major = params.get("major", 1.5)
    minor = params.get("minor", 0.8)
    coords, _ = _disk(n, rng, {"radius": 1.0})
    coords[:, 0] *= major
    coords[:, 1] *= minor
    meta = {"shape": "ellipse", "major": major, "minor": minor}
    return coords, meta


def _crescent(n: int, rng: Generator, params: Dict) -> Tuple[np.ndarray, Dict]:
    """Non-convex crescent moon shape via rejection sampling."""
    R_outer = params.get("R_outer", 1.0)
    R_inner = params.get("R_inner", 0.7)
    offset = params.get("offset", 0.4)

    coords = []
    count = 0
    batch_size = n * 2

    while count < n:
        batch = rng.uniform(-R_outer, R_outer, (batch_size, 2))

        mask_outer = (batch[:, 0] ** 2 + batch[:, 1] ** 2) <= R_outer**2

        mask_inner = ((batch[:, 0] - offset) ** 2 + batch[:, 1] ** 2) <= R_inner**2
        valid = batch[mask_outer & ~mask_inner]
        coords.append(valid)
        count += len(valid)

    coords = np.vstack(coords)[:n]
    meta = {"shape": "crescent", "R_outer": R_outer, "R_inner": R_inner, "offset": offset}
    return coords, meta


def _annulus(n: int, rng: Generator, params: Dict) -> Tuple[np.ndarray, Dict]:
    """Ring with inner and outer radii."""
    r_in = params.get("r_in", 0.5)
    r_out = params.get("r_out", 1.0)
    r = np.sqrt(rng.uniform(r_in**2, r_out**2, n))
    theta = rng.uniform(0, 2 * np.pi, n)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    coords = np.column_stack([x, y])
    meta = {"shape": "annulus", "r_in": r_in, "r_out": r_out}
    return coords, meta


def _peanut(n: int, rng: Generator, params: Dict) -> Tuple[np.ndarray, Dict]:
    """Two overlapping lobes (peanut shape)."""
    lobe_radius = params.get("lobe_radius", 0.8)
    separation = params.get("separation", 1.0)

    n1 = n // 2
    n2 = n - n1

    r1 = np.sqrt(rng.uniform(0, 1, n1)) * lobe_radius
    t1 = rng.uniform(0, 2 * np.pi, n1)
    x1 = r1 * np.cos(t1) - separation / 2
    y1 = r1 * np.sin(t1)

    r2 = np.sqrt(rng.uniform(0, 1, n2)) * lobe_radius
    t2 = rng.uniform(0, 2 * np.pi, n2)
    x2 = r2 * np.cos(t2) + separation / 2
    y2 = r2 * np.sin(t2)

    coords = np.vstack([np.column_stack([x1, y1]), np.column_stack([x2, y2])])
    meta = {
        "shape": "peanut",
        "lobe_radius": lobe_radius,
        "separation": separation,
    }
    return coords, meta


def _disconnected_blobs(n: int, rng: Generator, params: Dict) -> Tuple[np.ndarray, Dict]:
    """Two spatially separated Gaussian blobs."""
    sigma = params.get("sigma", 0.4)
    separation = params.get("separation", 3.0)

    n1 = n // 2
    n2 = n - n1

    x1 = rng.normal(-separation / 2, sigma, n1)
    y1 = rng.normal(0, sigma, n1)

    x2 = rng.normal(separation / 2, sigma, n2)
    y2 = rng.normal(0, sigma, n2)

    coords = np.vstack([np.column_stack([x1, y1]), np.column_stack([x2, y2])])
    meta = {"shape": "disconnected_blobs", "sigma": sigma, "separation": separation}
    return coords, meta
