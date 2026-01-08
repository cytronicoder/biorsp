"""
Expression simulation: library sizes, signal fields, and count generation.

Implements various spatial patterns and confounded null models.
"""

from typing import Any, Dict, Tuple

import numpy as np
from numpy.random import Generator
from scipy.stats import nbinom

from .density import knn_density
from .geometry import compute_polar


def simulate_library_size(
    n: int, rng: Generator, model: str = "lognormal", params: Dict[str, Any] = None
) -> np.ndarray:
    """
    Generate per-cell library sizes (total UMI counts).

    Parameters
    ----------
    n : int
        Number of cells
    rng : Generator
        Random number generator
    model : str, optional
        Distribution model: 'lognormal', 'uniform', 'gamma'
    params : Dict, optional
        Model parameters

    Returns
    -------
    libsize : np.ndarray
        Library size per cell (n,)
    """
    params = params or {}

    if model == "lognormal":
        mean_lib = params.get("mean", 1000)
        sigma = params.get("sigma", 0.5)
        return rng.lognormal(np.log(mean_lib), sigma, n)

    elif model == "uniform":
        low = params.get("low", 500)
        high = params.get("high", 2000)
        return rng.uniform(low, high, n)

    elif model == "gamma":
        shape = params.get("shape", 2.0)
        scale = params.get("scale", 500)
        return rng.gamma(shape, scale, n)

    else:
        raise ValueError(f"Unknown libsize model: {model}")


def generate_signal_field(
    coords: np.ndarray, pattern: str, params: Dict[str, Any] = None
) -> np.ndarray:
    """
    Generate spatial signal field in [0, 1].

    Represents relative expression probability or intensity.

    Parameters
    ----------
    coords : np.ndarray
        Cell coordinates (n, 2)
    pattern : str
        Pattern type: uniform, core, rim, wedge, wedge_core, wedge_rim,
        two_wedges, halfplane_gradient
    params : Dict, optional
        Pattern-specific parameters

    Returns
    -------
    field : np.ndarray
        Signal field values in [0, 1] (n,)
    """
    params = params or {}
    vantage, r, theta = compute_polar(coords, center="median")
    r_norm = (r - r.min()) / (r.max() - r.min() + 1e-9)

    if pattern == "uniform":
        base = params.get("base", 0.5)
        return np.full(len(coords), base)

    elif pattern == "sparse":
        base = params.get("base", 0.05)
        return np.full(len(coords), base)

    elif pattern == "core":

        steepness = params.get("steepness", 5.0)
        return 1.0 / (1.0 + np.exp(steepness * (r_norm - 0.5)))

    elif pattern == "rim":

        steepness = params.get("steepness", 5.0)
        return 1.0 / (1.0 + np.exp(-steepness * (r_norm - 0.5)))

    elif pattern == "wedge":

        angle_center = params.get("angle_center", 0.0)
        width_rad = params.get("width_rad", np.pi / 4)
        diff = np.abs(np.arctan2(np.sin(theta - angle_center), np.cos(theta - angle_center)))
        mask = diff < width_rad
        field = np.full(len(coords), 0.05)
        field[mask] = 0.95
        return field

    elif pattern == "wedge_core":

        field_wedge = generate_signal_field(coords, "wedge", params)
        field_core = generate_signal_field(coords, "core", params)
        return field_wedge * field_core

    elif pattern == "wedge_rim":

        field_wedge = generate_signal_field(coords, "wedge", params)
        field_rim = generate_signal_field(coords, "rim", params)
        return field_wedge * field_rim

    elif pattern == "two_wedges":

        p1 = generate_signal_field(coords, "wedge", {**params, "angle_center": 0.0})
        p2 = generate_signal_field(coords, "wedge", {**params, "angle_center": np.pi})
        return np.maximum(p1, p2)

    elif pattern == "halfplane_gradient":

        phi = params.get("phi", 0.0)
        projection = coords[:, 0] * np.cos(phi) + coords[:, 1] * np.sin(phi)
        proj_norm = (projection - projection.min()) / (projection.max() - projection.min() + 1e-9)
        return proj_norm

    else:
        raise ValueError(f"Unknown pattern: {pattern}")


def generate_expression_from_field(
    field: np.ndarray,
    libsize: np.ndarray,
    rng: Generator,
    expr_model: str = "nb",
    params: Dict[str, Any] = None,
) -> np.ndarray:
    """
    Generate expression counts from signal field and library sizes.

    Parameters
    ----------
    field : np.ndarray
        Signal field in [0, 1] (n,)
    libsize : np.ndarray
        Library size per cell (n,)
    rng : Generator
        Random number generator
    expr_model : str, optional
        Expression model: 'nb', 'poisson', 'bernoulli'
    params : Dict, optional
        Model parameters

    Returns
    -------
    counts : np.ndarray
        Expression counts (n,)
    """
    params = params or {}

    if expr_model == "bernoulli":

        p = np.clip(field, 0, 1)
        return rng.binomial(1, p).astype(float)

    elif expr_model == "poisson":

        abundance = params.get("abundance", 1e-3)
        mu = libsize * field * abundance
        return rng.poisson(mu)

    elif expr_model == "nb":

        abundance = params.get("abundance", 1e-3)
        phi = params.get("phi", 10.0)

        mu = libsize * field * abundance
        var = mu + (mu**2) / phi

        p_nb = np.clip(mu / (var + 1e-9), 0, 1)
        n_nb = np.clip(mu**2 / (var - mu + 1e-9), 1e-3, 1e6)

        counts = np.array([nbinom.rvs(n_nb[i], p_nb[i], random_state=rng) for i in range(len(mu))])
        return counts

    else:
        raise ValueError(f"Unknown expression model: {expr_model}")


def generate_confounded_null(
    coords: np.ndarray,
    libsize: np.ndarray,
    rng: Generator,
    null_type: str,
    params: Dict[str, Any] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Generate confounded null expression (no true spatial pattern).

    Parameters
    ----------
    coords : np.ndarray
        Coordinates (n, 2)
    libsize : np.ndarray
        Library sizes (n,)
    rng : Generator
        Random number generator
    null_type : str
        Null model: 'iid', 'depth_confounded', 'density_confounded', 'mask_stress'
    params : Dict, optional
        Model parameters

    Returns
    -------
    counts : np.ndarray
        Expression counts (n,)
    meta : Dict
        Metadata about the null
    """
    params = params or {}

    if null_type == "iid":

        base_prob = params.get("base_prob", 0.1)
        field = np.full(len(coords), base_prob)
        counts = generate_expression_from_field(field, libsize, rng, expr_model="nb", params=params)
        meta = {"null_type": "iid", "base_prob": base_prob}
        return counts, meta

    elif null_type == "depth_confounded":

        depth_effect = params.get("depth_effect", 0.5)
        libsize_norm = (libsize - libsize.min()) / (libsize.max() - libsize.min() + 1e-9)
        field = 0.1 + depth_effect * libsize_norm
        counts = generate_expression_from_field(field, libsize, rng, expr_model="nb", params=params)
        meta = {"null_type": "depth_confounded", "depth_effect": depth_effect}
        return counts, meta

    elif null_type == "density_confounded":

        density = knn_density(coords, k=20)
        density_effect = params.get("density_effect", 0.5)
        field = 0.1 + density_effect * (density / density.max())
        counts = generate_expression_from_field(field, libsize, rng, expr_model="nb", params=params)
        meta = {"null_type": "density_confounded", "density_effect": density_effect}
        return counts, meta

    elif null_type == "mask_stress":

        base_prob = params.get("base_prob", 0.01)
        field = np.full(len(coords), base_prob)
        counts = generate_expression_from_field(field, libsize, rng, expr_model="nb", params=params)
        meta = {"null_type": "mask_stress", "base_prob": base_prob}
        return counts, meta

    else:
        raise ValueError(f"Unknown null type: {null_type}")
