"""
Coordinate distortions for robustness testing.

All distortions preserve point count and return metadata about the transformation.
"""

from typing import Any, Dict, Tuple

import numpy as np
from numpy.random import Generator


def apply_distortion(
    coords: np.ndarray,
    kind: str,
    strength: float,
    rng: Generator,
    params: Dict[str, Any] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply spatial distortion to coordinates.

    Parameters
    ----------
    coords : np.ndarray
        Input coordinates (n, 2)
    kind : str
        Distortion type: none, rotate, aniso_scale, swirl, radial_warp, jitter, subsample
    strength : float
        Distortion magnitude (interpretation depends on kind)
    rng : Generator
        Random number generator
    params : Dict[str, Any], optional
        Distortion-specific parameters

    Returns
    -------
    coords_distorted : np.ndarray
        Transformed coordinates
    meta : Dict[str, Any]
        Transformation metadata
    """
    params = params or {}

    if kind == "none":
        return coords.copy(), {"distortion": "none", "strength": 0.0}

    elif kind == "rotate":
        return _rotate(coords, strength, params)

    elif kind == "aniso_scale":
        return _aniso_scale(coords, strength, params)

    elif kind == "swirl":
        return _swirl(coords, strength, params)

    elif kind == "radial_warp":
        return _radial_warp(coords, strength, params)

    elif kind == "jitter":
        return _jitter(coords, strength, rng, params)

    elif kind == "subsample":
        return _subsample(coords, strength, rng, params)

    else:
        raise ValueError(f"Unknown distortion kind: {kind}")


def _rotate(coords: np.ndarray, strength: float, params: Dict) -> Tuple[np.ndarray, Dict]:
    """Rotate coordinates by angle (strength in degrees)."""
    theta = np.radians(strength)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    coords_rot = coords @ R.T
    meta = {"distortion": "rotate", "angle_deg": strength}
    return coords_rot, meta


def _aniso_scale(coords: np.ndarray, strength: float, params: Dict) -> Tuple[np.ndarray, Dict]:
    """Anisotropic scaling (stretch one axis)."""
    axis = params.get("axis", 0)
    coords_scaled = coords.copy()
    coords_scaled[:, axis] *= strength
    meta = {"distortion": "aniso_scale", "strength": strength, "axis": axis}
    return coords_scaled, meta


def _swirl(coords: np.ndarray, strength: float, params: Dict) -> Tuple[np.ndarray, Dict]:
    """Swirl: rotation angle proportional to radius."""
    r = np.linalg.norm(coords, axis=1, keepdims=True)
    theta = strength * r.flatten()
    c, s = np.cos(theta), np.sin(theta)
    coords_swirl = np.zeros_like(coords)
    coords_swirl[:, 0] = coords[:, 0] * c - coords[:, 1] * s
    coords_swirl[:, 1] = coords[:, 0] * s + coords[:, 1] * c
    meta = {"distortion": "swirl", "strength": strength}
    return coords_swirl, meta


def _radial_warp(coords: np.ndarray, strength: float, params: Dict) -> Tuple[np.ndarray, Dict]:
    """Radial warping: r' = r^strength."""
    r = np.linalg.norm(coords, axis=1)
    theta = np.arctan2(coords[:, 1], coords[:, 0])
    r_new = np.power(r + 1e-9, strength)
    coords_warp = np.column_stack([r_new * np.cos(theta), r_new * np.sin(theta)])
    meta = {"distortion": "radial_warp", "exponent": strength}
    return coords_warp, meta


def _jitter(
    coords: np.ndarray, strength: float, rng: Generator, params: Dict
) -> Tuple[np.ndarray, Dict]:
    """Add Gaussian noise with sigma=strength."""
    noise = rng.normal(0, strength, coords.shape)
    coords_jitter = coords + noise
    meta = {"distortion": "jitter", "sigma": strength}
    return coords_jitter, meta


def _subsample(
    coords: np.ndarray, strength: float, rng: Generator, params: Dict
) -> Tuple[np.ndarray, Dict]:
    """Subsample cells (strength = fraction to keep, 0 < strength <= 1)."""
    if strength >= 1.0:
        return coords.copy(), {"distortion": "subsample", "frac": 1.0, "n": len(coords)}

    n_keep = max(int(len(coords) * strength), 10)
    idx = rng.choice(len(coords), n_keep, replace=False)
    coords_sub = coords[idx]
    meta = {"distortion": "subsample", "frac": strength, "n": n_keep, "indices": idx}
    return coords_sub, meta
