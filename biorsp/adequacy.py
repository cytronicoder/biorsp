"""
Adequacy assessment module for BioRSP.

Implements sector coverage checks:
- Counting foreground/background cells in sliding angular windows
- Checking minimum cell count per sector
- Determining overall gene adequacy
"""

from dataclasses import dataclass

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
    """

    is_adequate: bool
    reason: str
    counts_fg: np.ndarray
    counts_bg: np.ndarray
    sector_mask: np.ndarray
    n_foreground: int
    n_background: int
    adequacy_fraction: float


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
        )

    centers = angle_grid(n_sectors)
    delta_rad = np.deg2rad(delta_deg)
    half_width = delta_rad / 2.0

    counts_fg = np.zeros(n_sectors, dtype=int)
    counts_bg = np.zeros(n_sectors, dtype=int)

    y_fg = y == 1
    y_bg = ~y_fg

    for b, phi in enumerate(centers):
        rel_theta = theta - phi
        rel_theta = (rel_theta + np.pi) % (2 * np.pi) - np.pi
        mask_all = np.abs(rel_theta) <= half_width
        counts_fg[b] = int(np.sum(mask_all & y_fg))
        counts_bg[b] = int(np.sum(mask_all & y_bg))

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
    )


__all__ = ["AdequacyReport", "sector_counts", "sector_adequacy_mask", "gene_adequacy"]
