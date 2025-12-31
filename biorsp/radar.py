"""
Radar estimand module for BioRSP.

Implements the radar radius function R_g(theta):
- Sliding window analysis
- Radial distribution comparison (Foreground vs Background)
- Signed Wasserstein-1 distance on background-CDF-normalized radii
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from .adequacy import AdequacyReport

import numpy as np
from scipy.stats import wasserstein_distance

from .constants import B_DEFAULT, DELTA_DEG_DEFAULT, N_BG_MIN_DEFAULT, N_FG_MIN_DEFAULT
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
        iqr_floor: Deprecated (retained for backward compatibility).
        iqr_floor_hits: Deprecated (retained for backward compatibility).
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
    B: int = B_DEFAULT,
    delta_deg: float = DELTA_DEG_DEFAULT,
    min_fg_sector: int = N_FG_MIN_DEFAULT,
    min_bg_sector: int = N_BG_MIN_DEFAULT,
    sector_indices: Optional[List[np.ndarray]] = None,
    adequacy: Optional["AdequacyReport"] = None,
) -> RadarResult:
    """
    Compute the signed RSP radar function comparing radial distributions.

    For each sector b with center phi_b:
    1. Select foreground and background cells in window [phi_b - delta/2, phi_b + delta/2].
    2. Check adequacy (min counts). If not adequate, R_b = NaN.
    3. Normalize radii by the background empirical CDF in the sector.
    4. Compute Wasserstein-1 distance (W1) between foreground and background CDF-normalized radii.
    5. Determine sign: negative if foreground is radially distal (larger radii), positive if proximal.
       Sign = sign(median(u_bg) - median(u_fg)).
       (Note: This convention makes 'rim' patterns negative, consistent with P_g = min R_g).

    Re-use API:
        - You may pass precomputed `sector_indices` (list of arrays) to avoid re-scanning angles.
        - Alternatively, pass an `adequacy` object returned by `gene_adequacy()`; `adequacy.sector_indices`
          will be used. If both are provided, `sector_indices` takes precedence.

    Args:
        r: (N,) array of radial distances.
        theta: (N,) array of angles in [-pi, pi).
        y: (N,) binary array (1 for foreground, 0 for background).
        B: Number of grid points.
        delta_deg: Window width in degrees.
        min_fg_sector: Minimum foreground counts to compute RSP.
        min_bg_sector: Minimum background counts to compute RSP.
        sector_indices: Optional precomputed per-sector index lists (list of arrays).
        adequacy: Optional `AdequacyReport` object to reuse precomputed indices.

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

    # Normalize y to boolean to avoid surprises from floats/ints
    y = np.asarray(y).astype(bool)
    y_fg = y
    y_bg = ~y
    iqr_floor = np.nan
    iqr_floor_hits = np.zeros(B, dtype=bool)

    # Decide whether to use precomputed indices. Precedence: explicit `sector_indices` -> `adequacy` -> compute
    if sector_indices is None and adequacy is not None:
        sector_indices = adequacy.sector_indices

    if sector_indices is None:
        # Prepare sorted angles for two-pointer sliding window (O(N + B))
        two_pi = 2 * np.pi
        theta_mod = theta % two_pi
        order = np.argsort(theta_mod)
        theta_sorted = theta_mod[order]
        theta2 = np.concatenate([theta_sorted, theta_sorted + two_pi])
        idx2 = np.concatenate([order, order])

        left = 0
        right = 0
        n2 = len(theta2)

        # Process centers in ascending angular order to allow monotonic two-pointer movement
        centers_mod = (centers + two_pi) % two_pi
        # Map centers that would wrap to the interval (2pi, 3pi) so phi_use is monotonic
        centers_use = np.where((centers_mod - half_width) < 0, centers_mod + two_pi, centers_mod)
        centers_order = np.argsort(centers_use)

        for b_idx in centers_order:
            phi = centers[b_idx]
            phi_mod = (phi + two_pi) % two_pi
            phi_use = phi_mod + two_pi if (phi_mod - half_width) < 0 else phi_mod

            start = phi_use - half_width
            end = phi_use + half_width

            while left < n2 and theta2[left] < start:
                left += 1
            if right < left:
                right = left
            while right < n2 and theta2[right] <= end:
                right += 1

            if right <= left:
                counts_fg[b_idx] = 0
                counts_bg[b_idx] = 0
                continue

            window_idx = idx2[left:right]
            unique_idx = np.unique(window_idx)

            # Foreground in window
            mask_fg_idx = unique_idx[y_fg[unique_idx]]
            r_fg_sector = r[mask_fg_idx]
            n_fg = len(r_fg_sector)
            counts_fg[b_idx] = n_fg

            # Background in window
            bg_idx = unique_idx[~y_fg[unique_idx]]
            r_bg_sector = r[bg_idx]
            n_bg = len(r_bg_sector)
            counts_bg[b_idx] = n_bg

            # 2. Check adequacy
            if n_fg < min_fg_sector or n_bg < min_bg_sector:
                continue

            # 3. Compute W1 distance on background-CDF-normalized radii
            r_bg_sorted = np.sort(r_bg_sector)
            n_bg = len(r_bg_sorted)
            u_fg = np.searchsorted(r_bg_sorted, r_fg_sector, side="right") / n_bg
            u_bg = np.searchsorted(r_bg_sorted, r_bg_sector, side="right") / n_bg
            w1 = wasserstein_distance(u_fg, u_bg)

            # 4. Determine sign with tie-breaker using mean
            diff_median = np.median(u_bg) - np.median(u_fg)
            if diff_median > 0:
                sign = 1.0
            elif diff_median < 0:
                sign = -1.0
            else:
                # Tie-breaker: use mean difference
                diff_mean = np.mean(u_bg) - np.mean(u_fg)
                sign = 1.0 if diff_mean >= 0 else -1.0

            rsp_values[b_idx] = sign * w1
    else:
        # Use precomputed sector indices from adequacy to avoid re-scanning angles
        for b_idx in range(B):
            unique_idx = np.asarray(sector_indices[b_idx], dtype=int)
            if unique_idx.size == 0:
                counts_fg[b_idx] = 0
                counts_bg[b_idx] = 0
                continue

            mask_fg_idx = unique_idx[y_fg[unique_idx]]
            r_fg_sector = r[mask_fg_idx]
            n_fg = len(r_fg_sector)
            counts_fg[b_idx] = n_fg

            bg_idx = unique_idx[~y_fg[unique_idx]]
            r_bg_sector = r[bg_idx]
            n_bg = len(r_bg_sector)
            counts_bg[b_idx] = n_bg

            if n_fg < min_fg_sector or n_bg < min_bg_sector:
                continue

            r_bg_sorted = np.sort(r_bg_sector)
            n_bg = len(r_bg_sorted)
            u_fg = np.searchsorted(r_bg_sorted, r_fg_sector, side="right") / n_bg
            u_bg = np.searchsorted(r_bg_sorted, r_bg_sector, side="right") / n_bg
            w1 = wasserstein_distance(u_fg, u_bg)
            diff_median = np.median(u_bg) - np.median(u_fg)
            if diff_median > 0:
                sign = 1.0
            elif diff_median < 0:
                sign = -1.0
            else:
                # Tie-breaker: use mean difference
                diff_mean = np.mean(u_bg) - np.mean(u_fg)
                sign = 1.0 if diff_mean >= 0 else -1.0
            rsp_values[b_idx] = sign * w1

    return RadarResult(
        rsp=rsp_values,
        counts_fg=counts_fg,
        counts_bg=counts_bg,
        centers=centers,
        iqr_floor=iqr_floor,
        iqr_floor_hits=iqr_floor_hits,
    )

def empty_radar(
    B: int,
    counts_fg: Optional[np.ndarray] = None,
    counts_bg: Optional[np.ndarray] = None,
) -> RadarResult:
    """Create an empty radar result with NaN RSP values."""
    centers = angle_grid(B)
    if counts_fg is None:
        counts_fg = np.zeros(B, dtype=int)
    if counts_bg is None:
        counts_bg = np.zeros(B, dtype=int)
    return RadarResult(
        rsp=np.full(B, np.nan),
        counts_fg=counts_fg,
        counts_bg=counts_bg,
        centers=centers,
        iqr_floor=np.nan,
        iqr_floor_hits=np.zeros(B, dtype=bool),
    )


__all__ = ["RadarResult", "compute_rsp_radar", "empty_radar"]
