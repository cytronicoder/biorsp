"""
Adequacy assessment module for BioRSP.

Implements sector coverage checks:
- Counting foreground/background cells in sliding angular windows
- Checking minimum cell count per sector
- Determining overall gene adequacy
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .constants import (
    ADEQUACY_FRACTION_DEFAULT,
    N_BG_MIN_DEFAULT,
    N_FG_MIN_DEFAULT,
    N_FG_TOT_MIN_DEFAULT,
    REASON_GENE_UNDERPOWERED,
    REASON_OK,
    REASON_SECTOR_BG_TOO_SMALL,
    REASON_SECTOR_FG_TOO_SMALL,
    REASON_SECTOR_MIXED_TOO_SMALL,
)
from .geometry import angle_grid


@dataclass
class AdequacyReport:
    """
    Report on gene adequacy for RSP computation.

    Attributes:
        is_adequate: Boolean indicating if gene passed all checks.
        reason: String code explaining failure (or OK).
        counts_fg: (B,) array of foreground cell counts per sector.
        counts_bg: (B,) array of background cell counts per sector.
        sector_mask: (B,) boolean array, True if sector meets fg/bg thresholds.
        n_foreground: Total number of foreground cells.
        n_background: Total number of background cells.
        adequacy_fraction: Fraction of sectors meeting adequacy thresholds.
        sector_indices: Optional list of arrays, each with the (unique) cell indices inside the sector window.
    """

    is_adequate: bool
    reason: str
    counts_fg: np.ndarray
    counts_bg: np.ndarray
    sector_mask: np.ndarray
    n_foreground: int
    n_background: int
    adequacy_fraction: float
    sector_indices: Optional[List[np.ndarray]] = None

def sector_counts(theta: np.ndarray, n_sectors: int) -> np.ndarray:
    """
    Count foreground cells in each of `n_sectors` angular sectors.

    Sectors are defined by [2*pi*b/B, 2*pi*(b+1)/B) for b=0..n_sectors-1.

    Args:
        theta: (N_fg,) array of angles in radians (any range).
        n_sectors: Number of sectors (B).

    Returns:
        counts: (n_sectors,) integer array of counts.
    """
    # Normalize angles to [0, 2pi) and compute bin indices via floor(theta * B / 2pi)
    theta_mod = theta % (2 * np.pi)
    bin_indices = np.floor(theta_mod * n_sectors / (2 * np.pi)).astype(int)

    # Count occurrences per sector
    counts = np.bincount(bin_indices, minlength=n_sectors)

    return counts


def sector_adequacy_mask(counts: np.ndarray, min_count: int = 1) -> np.ndarray:
    """
    Identify adequate sectors.

    Args:
        counts: (B,) array of sector counts.
        min_count: Minimum required cells per sector (default 1).

    Returns:
        mask: (B,) boolean array, True where counts >= min_count.
    """
    return counts >= min_count


def gene_adequacy(
    y: np.ndarray,
    theta: np.ndarray,
    n_sectors: int,
    delta_deg: float,
    min_fg_sector: int = N_FG_MIN_DEFAULT,
    min_bg_sector: int = N_BG_MIN_DEFAULT,
    min_fg_total: int = N_FG_TOT_MIN_DEFAULT,
    min_adequacy_fraction: float = ADEQUACY_FRACTION_DEFAULT,
) -> AdequacyReport:
    """
    Assess gene adequacy for RSP.

    Checks performed:
    - Presence of any foreground cells.
    - Each sector has at least `min_fg_sector` foreground and `min_bg_sector` background cells.

    Args:
        y: (N,) boolean foreground indicator.
        theta: (N,) array of angles for ALL cells (radians, used for foreground cells only).
        n_sectors: Number of sectors.
        delta_deg: Sector width in degrees.
        min_fg_sector: Minimum foreground cells per sector.
        min_bg_sector: Minimum background cells per sector.
        min_fg_total: Minimum total foreground cells required.
        min_adequacy_fraction: Minimum fraction of adequate sectors.

    Returns:
        AdequacyReport object.
    """
    if len(theta) != len(y):
        raise ValueError("theta and y must have the same length.")
    if n_sectors <= 0:
        raise ValueError("n_sectors must be positive.")
    if delta_deg <= 0:
        raise ValueError("delta_deg must be > 0.")

    # Normalize y to boolean to avoid surprises from floats or non-binary inputs
    y = np.asarray(y).astype(bool)

    n_fg = int(np.sum(y))
    n_bg = len(y) - n_fg

    if n_fg < min_fg_total:
        return AdequacyReport(
            is_adequate=False,
            reason=REASON_GENE_UNDERPOWERED,
            counts_fg=np.zeros(n_sectors, dtype=int),
            counts_bg=np.zeros(n_sectors, dtype=int),
            sector_mask=np.zeros(n_sectors, dtype=bool),
            n_foreground=n_fg,
            n_background=n_bg,
            adequacy_fraction=0.0,
            sector_indices=None,
        )

    centers = angle_grid(n_sectors)
    delta_rad = np.deg2rad(delta_deg)
    half_width = delta_rad / 2.0

    # Prepare sorted angles for two-pointer sliding window (O(N + B))
    two_pi = 2 * np.pi
    theta_mod = (theta % two_pi)
    order = np.argsort(theta_mod)
    theta_sorted = theta_mod[order]
    # duplicated arrays to avoid wrap handling
    theta2 = np.concatenate([theta_sorted, theta_sorted + two_pi])
    idx2 = np.concatenate([order, order])

    counts_fg = np.zeros(n_sectors, dtype=int)
    counts_bg = np.zeros(n_sectors, dtype=int)
    sector_indices = [np.array([], dtype=int) for _ in range(n_sectors)]

    y_bool = np.asarray(y).astype(bool)

    # two-pointer window over theta2
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
            sector_indices[b_idx] = np.array([], dtype=int)
        else:
            window_idx = idx2[left:right]
            unique_idx = np.unique(window_idx)
            sector_indices[b_idx] = unique_idx
            counts_fg[b_idx] = int(np.sum(y_bool[unique_idx]))
            counts_bg[b_idx] = int(len(unique_idx) - counts_fg[b_idx])

    fg_mask = sector_adequacy_mask(counts_fg, min_fg_sector)
    bg_mask = sector_adequacy_mask(counts_bg, min_bg_sector)
    mask = fg_mask & bg_mask
    adequacy_fraction = float(np.mean(mask)) if n_sectors > 0 else 0.0

    is_adequate = adequacy_fraction >= min_adequacy_fraction
    if is_adequate:
        reason = REASON_OK
    elif np.all(~fg_mask):
        reason = REASON_SECTOR_FG_TOO_SMALL
    elif np.all(~bg_mask):
        reason = REASON_SECTOR_BG_TOO_SMALL
    else:
        reason = REASON_SECTOR_MIXED_TOO_SMALL

    return AdequacyReport(
        is_adequate=is_adequate,
        reason=reason,
        counts_fg=counts_fg,
        counts_bg=counts_bg,
        sector_mask=mask,
        n_foreground=n_fg,
        n_background=n_bg,
        adequacy_fraction=adequacy_fraction,
        sector_indices=sector_indices,
    )


__all__ = ["AdequacyReport", "sector_counts", "sector_adequacy_mask", "gene_adequacy"]
