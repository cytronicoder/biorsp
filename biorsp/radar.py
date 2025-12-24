"""
Radar estimand module for BioRSP.

Implements the radar radius function R_g(theta):
- Sliding window analysis
- Wasserstein-1 distance vs Uniform
- Signed magnitude (Concentration vs Dispersion)
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import wasserstein_distance

from .geometry import angle_grid


@dataclass
class RadarResult:
    """
    Result of radar computation.

    Attributes:
        rsp: (B,) array of signed RSP values.
        counts: (B,) array of foreground counts per sector (window).
        centers: (B,) array of sector center angles.
    """

    rsp: np.ndarray
    counts: np.ndarray
    centers: np.ndarray


def compute_rsp_radar(
    theta_fg: np.ndarray, B: int = 360, delta_deg: float = 20.0
) -> RadarResult:
    """
    Compute the signed RSP radar function.

    For each sector b with center phi_b:
    1. Select foreground cells in window [phi_b - delta/2, phi_b + delta/2].
    2. Compute Wasserstein-1 distance (W1) between empirical angles and Uniform.
    3. Determine sign based on dispersion (MAD) vs Uniform MAD.
    4. R_b = sign * W1.

    Args:
        theta_fg: (N_fg,) array of foreground angles in [0, 2pi).
        B: Number of grid points.
        delta_deg: Window width in degrees.

    Returns:
        RadarResult object.
    """
    # 1. Define grid
    centers = angle_grid(B)
    delta_rad = np.deg2rad(delta_deg)
    half_width = delta_rad / 2.0

    rsp_values = np.zeros(B)
    counts = np.zeros(B, dtype=int)

    # Pre-compute Uniform MAD for sign reference
    # For U ~ Uniform[-d/2, d/2], MAD = d/4
    mad_uniform = delta_rad / 4.0

    # Reference uniform distribution for W1 (using a dense grid approximation)
    # We use a fixed set of points to represent the ideal uniform distribution
    n_ref = 100
    uniform_ref = np.linspace(-half_width, half_width, n_ref)

    for b in range(B):
        phi = centers[b]

        # 2. Relative angles centered at phi
        # Wrap to [-pi, pi)
        rel_theta = theta_fg - phi
        rel_theta = (rel_theta + np.pi) % (2 * np.pi) - np.pi

        # 3. Select points in window [-delta/2, delta/2]
        in_window = np.abs(rel_theta) <= half_width
        samples = rel_theta[in_window]
        n_samples = len(samples)
        counts[b] = n_samples

        if n_samples == 0:
            rsp_values[b] = 0.0
            continue

        # 4. Compute W1 distance
        # We compare the samples (centered at 0) to the uniform reference (centered at 0)
        w1 = wasserstein_distance(samples, uniform_ref)

        # 5. Compute sign using median absolute deviation (MAD) for robustness
        mad_sample = np.median(np.abs(samples))
        sign = 1.0 if mad_sample < mad_uniform else -1.0

        rsp_values[b] = sign * w1

    return RadarResult(rsp=rsp_values, counts=counts, centers=centers)


__all__ = ["RadarResult", "compute_rsp_radar"]
