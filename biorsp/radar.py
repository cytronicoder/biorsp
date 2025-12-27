"""
Radar estimand module for BioRSP.

Implements the radar radius function R_g(theta):
- Sliding window analysis
- Radial distribution comparison (Foreground vs Background)
- Signed, IQR-normalized Wasserstein-1 distance
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import iqr, wasserstein_distance

from .constants import EPS, N_BG_MIN_DEFAULT, N_FG_MIN_DEFAULT
from .geometry import angle_grid


@dataclass
class RadarResult:
    """
    Result of radar computation.

    Attributes:
        rsp: (B,) array of signed RSP values. NaNs indicate underpowered sectors.
        counts_fg: (B,) array of foreground counts per sector.
        counts_bg: (B,) array of background counts per sector.
        centers: (B,) array of sector center angles.
        iqr_floor: Lower bound used for IQR normalization.
        iqr_floor_hits: (B,) boolean array indicating when floor was applied.
    """

    rsp: np.ndarray
    counts_fg: np.ndarray
    counts_bg: np.ndarray
    centers: np.ndarray
    iqr_floor: float
    iqr_floor_hits: np.ndarray


def compute_rsp_radar(
    r: np.ndarray,
    theta: np.ndarray,
    y: np.ndarray,
    B: int = 360,
    delta_deg: float = 20.0,
    min_fg_sector: int = N_FG_MIN_DEFAULT,
    min_bg_sector: int = N_BG_MIN_DEFAULT,
) -> RadarResult:
    """
    Compute the signed RSP radar function comparing radial distributions.

    For each sector b with center phi_b:
    1. Select foreground and background cells in window [phi_b - delta/2, phi_b + delta/2].
    2. Check adequacy (min counts). If not adequate, R_b = NaN.
    3. Compute Wasserstein-1 distance (W1) between foreground radii and background radii.
    4. Normalize by IQR of background radii in the sector with a global floor.
    5. Determine sign: negative if foreground is radially distal (larger radii), positive if proximal.
       Sign = sign(median(r_bg) - median(r_fg)).
       (Note: This convention makes 'rim' patterns negative, consistent with P_g = min R_g).

    Args:
        r: (N,) array of radial distances.
        theta: (N,) array of angles in [-pi, pi).
        y: (N,) binary array (1 for foreground, 0 for background).
        B: Number of grid points.
        delta_deg: Window width in degrees.
        min_fg_sector: Minimum foreground counts to compute RSP.
        min_bg_sector: Minimum background counts to compute RSP.

    Returns:
        RadarResult object.
    """
    if len(r) != len(theta) or len(theta) != len(y):
        raise ValueError("r, theta, and y must have the same length.")
    if B <= 0:
        raise ValueError("B must be positive.")
    if delta_deg <= 0:
        raise ValueError("delta_deg must be > 0.")

    # 1. Define grid
    centers = angle_grid(B)
    delta_rad = np.deg2rad(delta_deg)
    half_width = delta_rad / 2.0

    rsp_values = np.full(B, np.nan)
    counts_fg = np.zeros(B, dtype=int)
    counts_bg = np.zeros(B, dtype=int)

    # Separate foreground and background
    y_fg = y == 1
    y_bg = y == 0
    r_bg_all = r[y_bg]

    # Global fallback for IQR if sector is degenerate
    global_iqr = iqr(r_bg_all) if len(r_bg_all) > 0 else np.nan
    if not np.isfinite(global_iqr) or global_iqr <= 0:
        global_iqr = 1.0  # Fallback if even global is degenerate
    iqr_floor = max(0.1 * global_iqr, EPS)
    iqr_floor_hits = np.zeros(B, dtype=bool)

    for b in range(B):
        phi = centers[b]
        rel_theta = theta - phi
        rel_theta = (rel_theta + np.pi) % (2 * np.pi) - np.pi
        mask_all = np.abs(rel_theta) <= half_width

        # Foreground in window
        mask_fg = mask_all & y_fg
        r_fg_sector = r[mask_fg]
        n_fg = len(r_fg_sector)
        counts_fg[b] = n_fg

        # Background in window
        mask_bg = mask_all & y_bg
        r_bg_sector = r[mask_bg]
        n_bg = len(r_bg_sector)
        counts_bg[b] = n_bg

        # 2. Check adequacy
        if n_fg < min_fg_sector or n_bg < min_bg_sector:
            continue

        # 3. Compute W1 distance
        w1 = wasserstein_distance(r_fg_sector, r_bg_sector)

        # 4. Normalize by IQR of background
        iqr_bg = iqr(r_bg_sector)
        # Stabilize IQR with a global floor
        denom = max(iqr_bg, iqr_floor)
        if denom > iqr_bg:
            iqr_floor_hits[b] = True

        normalized_w1 = w1 / denom

        # 5. Determine sign
        # Sign = sign(median(r_bg) - median(r_fg))
        # If fg is at larger radii (rim), median(r_fg) > median(r_bg) -> negative sign.
        diff_median = np.median(r_bg_sector) - np.median(r_fg_sector)
        sign = 1.0 if diff_median >= 0 else -1.0

        rsp_values[b] = sign * normalized_w1

    return RadarResult(
        rsp=rsp_values,
        counts_fg=counts_fg,
        counts_bg=counts_bg,
        centers=centers,
        iqr_floor=iqr_floor,
        iqr_floor_hits=iqr_floor_hits,
    )


__all__ = ["RadarResult", "compute_rsp_radar"]
