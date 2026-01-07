"""
Unified Simulation Engine for BioRSP Methods Benchmarking.

This module provides primitives for:
1. Generating synthetic spatial manifolds (shapes).
2. Creating ground-truth expression patterns (archetypes).
3. Simulating count data with realistic noise and confounders.
4. Applying distortions to spatial coordinates.
5. Scoring standardized outputs using BioRSP v3.
"""

from typing import Dict, Optional, Tuple

import numpy as np
from scipy.stats import nbinom

from biorsp import (
    BioRSPConfig,
    assess_adequacy,
    compute_p_value,
    compute_rsp_radar,
    compute_scalar_summaries,
)
from biorsp.preprocess.geometry import compute_vantage, polar_coordinates

# Standardized Defaults for Methods Paper
DEFAULT_DELTA = 60.0
DEFAULT_B = 72
SECONDARY_DELTA = 180.0  # "Half-plane power mode"


def generate_coords(
    shape: str, n_points: int, seed: int, params: Optional[Dict] = None
) -> np.ndarray:
    """
    Generate 2D spatial coordinates for a given manifold shape.

    Shapes:
    - disk: Uniform disk
    - ellipse: Stretched disk
    - crescent: Non-convex arc
    - annulus: Ring
    - peanut: Two merging lobes
    - disconnected: Two separate blobs
    """
    rng = np.random.default_rng(seed)
    params = params or {}

    if shape == "disk":
        r = np.sqrt(rng.uniform(0, 1, n_points))
        theta = rng.uniform(0, 2 * np.pi, n_points)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return np.column_stack([x, y])

    elif shape == "ellipse":
        major = params.get("major", 1.5)
        minor = params.get("minor", 0.8)
        coords = generate_coords("disk", n_points, seed)
        coords[:, 0] *= major
        coords[:, 1] *= minor
        return coords

    elif shape == "crescent":
        # Rejection sampling for a crescent moon
        # Outer circle R=1, Inner circle r=0.7 offset by d=0.4
        coords = []
        count = 0
        local_rng = np.random.default_rng(seed)
        while count < n_points:
            batch = local_rng.uniform(-1, 1, (n_points, 2))
            # In outer circle
            mask_outer = (batch[:, 0] ** 2 + batch[:, 1] ** 2) <= 1.0
            # Not in inner circle
            r_inner = params.get("r_inner", 0.7)
            d_shift = params.get("d_shift", 0.4)
            mask_inner = ((batch[:, 0] - d_shift) ** 2 + batch[:, 1] ** 2) <= r_inner**2

            valid = batch[mask_outer & ~mask_inner]
            coords.append(valid)
            count += len(valid)

        return np.vstack(coords)[:n_points]

    elif shape == "annulus":
        r_in = params.get("r_in", 0.5)
        r_out = params.get("r_out", 1.0)
        r = np.sqrt(rng.uniform(r_in**2, r_out**2, n_points))
        theta = rng.uniform(0, 2 * np.pi, n_points)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return np.column_stack([x, y])

    elif shape == "peanut":
        # Two overlapping gaussians or two disks
        n1 = n_points // 2
        n2 = n_points - n1

        # Lobe 1
        r1 = np.sqrt(rng.uniform(0, 1, n1)) * 0.8
        t1 = rng.uniform(0, 2 * np.pi, n1)
        x1 = r1 * np.cos(t1) - 0.5
        y1 = r1 * np.sin(t1)

        # Lobe 2
        r2 = np.sqrt(rng.uniform(0, 1, n2)) * 0.8
        t2 = rng.uniform(0, 2 * np.pi, n2)
        x2 = r2 * np.cos(t2) + 0.5
        y2 = r2 * np.sin(t2)

        return np.vstack([np.column_stack([x1, y1]), np.column_stack([x2, y2])])

    elif shape == "disconnected":
        n1 = n_points // 2
        n2 = n_points - n1

        # Blob 1
        x1 = rng.normal(-1.5, 0.4, n1)
        y1 = rng.normal(0, 0.4, n1)

        # Blob 2
        x2 = rng.normal(1.5, 0.4, n2)
        y2 = rng.normal(0, 0.4, n2)

        return np.vstack([np.column_stack([x1, y1]), np.column_stack([x2, y2])])

    else:
        raise ValueError(f"Unknown shape: {shape}")


def apply_distortion(coords: np.ndarray, mode: str, strength: float, seed: int) -> np.ndarray:
    """
    Apply spatial distortions to coordinates.
    """
    rng = np.random.default_rng(seed)
    X = coords.copy()

    if mode == "none":
        return X

    elif mode == "rotation":
        theta = np.radians(strength)  # strength in degrees
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        return X @ R.T

    elif mode == "stretch":
        # Anisotropic scaling
        # strength > 1 stretches X, < 1 computes
        X[:, 0] *= strength
        return X

    elif mode == "swirl":
        # Swirl: rotation depends on radius
        r = np.linalg.norm(X, axis=1)
        theta = strength * r
        X_new = np.zeros_like(X)
        X_new[:, 0] = X[:, 0] * np.cos(theta) - X[:, 1] * np.sin(theta)
        X_new[:, 1] = X[:, 0] * np.sin(theta) + X[:, 1] * np.cos(theta)
        return X_new

    elif mode == "radial_warp":
        # Expand or contract radially
        r = np.linalg.norm(X, axis=1)
        theta = np.arctan2(X[:, 1], X[:, 0])
        r_new = np.power(r, strength)  # strength != 1 warps
        X_new = np.zeros_like(X)
        X_new[:, 0] = r_new * np.cos(theta)
        X_new[:, 1] = r_new * np.sin(theta)
        return X_new

    elif mode == "jitter":
        noise = rng.normal(0, strength, X.shape)
        return X + noise

    elif mode == "subsample":
        # Handled in sampling, but can do post-hoc here
        # strength = fraction to keep
        if strength >= 1.0:
            return X
        n_keep = int(len(X) * strength)
        idx = rng.choice(len(X), n_keep, replace=False)
        return X[idx]

    else:
        raise ValueError(f"Unknown distortion: {mode}")


def generate_true_probability(
    coords: np.ndarray, archetype: str, params: Optional[Dict] = None
) -> np.ndarray:
    """
    Define ground-truth probability P(expression | location).
    This represents the underlying biological signal.
    """
    params = params or {}
    v = compute_vantage(coords)
    rel = coords - v
    r = np.linalg.norm(rel, axis=1)
    theta = np.arctan2(rel[:, 1], rel[:, 0])  # [-pi, pi]

    # Normalize r to [0, 1] roughly for defining patterns
    r_norm = (r - r.min()) / (r.max() - r.min() + 1e-6)

    if archetype == "uniform":
        return np.ones(len(coords)) * params.get("p_base", 0.5)

    elif archetype == "sparse":
        return np.ones(len(coords)) * params.get("p_base", 0.05)

    elif archetype == "rim":
        # Prob increases with r
        steepness = params.get("steepness", 5)
        return 1 / (1 + np.exp(-steepness * (r_norm - 0.5)))

    elif archetype == "core":
        # Prob decreases with r
        steepness = params.get("steepness", 5)
        return 1 / (1 + np.exp(steepness * (r_norm - 0.5)))

    elif archetype == "wedge":
        # Prob localized in theta
        center = params.get("angle_center", 0)
        width = params.get("width_rad", np.pi / 4)

        # Angular distance
        diff = np.abs(np.arctan2(np.sin(theta - center), np.cos(theta - center)))

        # Gaussian decay or sharp window? Let's do sharp with soft edge
        mask = diff < width
        prob = np.zeros(len(coords)) + 0.05
        prob[mask] = 0.8
        return prob

    elif archetype == "wedge_rim":
        # AND logic
        p_wedge = generate_true_probability(coords, "wedge", params)
        p_rim = generate_true_probability(coords, "rim", params)
        return p_wedge * p_rim

    elif archetype == "wedge_core":
        p_wedge = generate_true_probability(coords, "wedge", params)
        p_core = generate_true_probability(coords, "core", params)
        return p_wedge * p_core

    elif archetype == "two_wedges":
        p1 = generate_true_probability(coords, "wedge", {**params, "angle_center": 0})
        p2 = generate_true_probability(coords, "wedge", {**params, "angle_center": np.pi})
        return np.maximum(p1, p2)

    else:
        raise ValueError(f"Unknown archetype: {archetype}")


def generate_expression(
    coords: np.ndarray, prob_map: np.ndarray, mode: str, seed: int, params: Optional[Dict] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate expression counts and library sizes.

    Returns:
       counts: Gene expression counts
       lib_size: Total counts per cell
       is_foreground: Boolean ground truth for 'expressed' (for validation)
    """
    rng = np.random.default_rng(seed)
    n = len(coords)
    params = params or {}

    # Base library size (log-normal)
    mean_lib = params.get("mean_lib", 1000)
    std_lib = params.get("std_lib", 0.5)
    libs = rng.lognormal(np.log(mean_lib), std_lib, n)

    # Confounders
    confounder = params.get("confounder", "none")
    if confounder == "depth_radial":
        # Library size correlates with radius
        v = compute_vantage(coords)
        r = np.linalg.norm(coords - v, axis=1)
        # Normalize and scale libs
        r_scale = (r / r.max()) + 0.5  # 0.5 to 1.5
        libs = libs * r_scale
    elif confounder == "depth_angular":
        theta = np.arctan2(coords[:, 1], coords[:, 0])
        scale = 1 + 0.5 * np.cos(theta)
        libs = libs * scale

    # Model
    if mode == "bernoulli":
        # Simple binary labels, ignore library size
        # prob_map is P(y=1)
        y = rng.binomial(1, prob_map)
        return y, libs, y  # In bernoulli, counts are the labels

    elif mode == "nb":
        # Negative Binomial
        # mu = lib_size * prob_map * scale_factor
        # prob_map serves as 'relative abundance'

        abundance_scale = params.get("abundance_scale", 1e-3)

        # Density confounding: higher "prob" in dense areas?
        if confounder == "density":
            # Compute local density
            from sklearn.neighbors import NearestNeighbors

            nn = NearestNeighbors(n_neighbors=20).fit(coords)
            dists, _ = nn.kneighbors(coords)
            density = 1 / (np.mean(dists, axis=1) + 1e-6)
            density_norm = density / density.mean()
            prob_map = prob_map * density_norm

        mu = libs * prob_map * abundance_scale

        # Dispersion alpha (var = mu + alpha * mu^2)
        # nbinom in scipy uses n, p.
        # Mean = n(1-p)/p, Var = n(1-p)/p^2.
        # Alternative parameterization: phi (inverse dispersion).
        phi = params.get("phi", 10.0)  # 1/alpha

        # Convert to n, p
        # Var = mu + mu^2 / phi
        var = mu + (mu**2) / phi
        p_nb = mu / var
        n_nb = mu**2 / (var - mu + 1e-9)

        counts = nbinom.rvs(n_nb, p_nb, random_state=rng)

        # Define 'foreground' for validation ?
        # Usually we just return counts and let BioRSP detect.
        # But for ground truth, we might flag 'true positive' components.
        # Here we just return counts.
        return counts, libs, prob_map  # Return prob_map as 'ground truth' potential

    else:
        raise ValueError(f"Unknown expression mode: {mode}")


def score_with_biorsp(
    coords: np.ndarray,
    counts: np.ndarray,
    libs: np.ndarray,
    gene_name: str,
    config: BioRSPConfig,
    seed: int,
) -> Dict:
    """
    Run single-gene BioRSP v3 scoring.
    """

    # 1. Define Foreground
    # For 'bernoulli', counts are binary.
    # For 'nb', we likely want BioRSP to define foreground via quantile or threshold.
    # Here we assume standard workflow: quantile on normalized data

    rng = np.random.default_rng(seed)

    # Pre-normalization (e.g. log1pCP10k) usually happens before,
    # but BioRSP works on 'expression' vector x.
    # Let's do simple CP10k
    x_norm = counts / (libs + 1e-6) * 10000 if np.max(counts) > 1 else counts

    # Vantage
    v = compute_vantage(coords)
    r, theta = polar_coordinates(coords, v)

    # Explicitly define foreground using biorsp internals or just mimic main.py
    # Here we'll mimic main.py logic for consistency
    from biorsp.preprocess.foreground import define_foreground

    y, fg_info = define_foreground(
        x_norm, mode=config.foreground_mode, q=config.foreground_quantile, rng=rng
    )

    if y is None:
        # Underpowered
        return {
            "gene": gene_name,
            "abstain": True,
            "reason": "low_fg",
            "p_value": np.nan,
            "S": np.nan,
            "C": fg_info.get("target_frac", 0),
            "r_mean_bg_sign": 0,
        }

    # Adequacy
    adequacy = assess_adequacy(r, theta, y, config=config)
    if not adequacy.is_adequate:
        return {
            "gene": gene_name,
            "abstain": True,
            "reason": adequacy.reason,
            "p_value": np.nan,
            "S": np.nan,
            "C": adequacy.adequacy_fraction,
            "r_mean_bg_sign": 0,
        }

    # Radar
    radar = compute_rsp_radar(
        r,
        theta,
        y,
        config=config,
        sector_indices=adequacy.sector_indices,
        frozen_mask=adequacy.sector_mask,
    )

    # Inference
    inference = compute_p_value(
        r, theta, y, config=config, umi_counts=libs, adequacy=adequacy, rng=rng, show_progress=False
    )

    # Summaries
    summ = compute_scalar_summaries(radar)

    return {
        "gene": gene_name,
        "abstain": False,
        "reason": "ok",
        "p_value": inference.p_value,
        "S": summ.anisotropy,  # v3 Spatial Score
        "C": adequacy.adequacy_fraction,  # v3 Coverage
        "r_mean_bg_sign": np.sign(summ.r_mean),
        "peak_angle": summ.peak_extremal_angle,
        "peak_magnitude": summ.peak_extremal,
        "coverage_fg": np.sum(radar.counts_fg > 0) / len(radar.counts_fg),
        "coverage_bg": np.sum(radar.counts_bg > 0) / len(radar.counts_bg),
        "polarity": summ.polarity,
    }


def get_base_config_v3() -> BioRSPConfig:
    return BioRSPConfig(
        B=DEFAULT_B,
        delta_deg=DEFAULT_DELTA,
        n_permutations=250,  # Sufficient for methods sims usually
        qc_mode="principled",
    )
