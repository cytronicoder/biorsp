"""
Adequacy assessment module for BioRSP.

Implements sector coverage checks:
- Binning foreground cells into B sectors
- Checking minimum cell count per sector
- Determining overall gene adequacy
"""

from dataclasses import dataclass

import numpy as np

from .constants import (REASON_GENE_UNDERPOWERED, REASON_OK,
                        REASON_SECTOR_FG_TOO_SMALL)


@dataclass
class AdequacyReport:
    """
    Report on gene adequacy for RSP computation.

    Attributes:
        is_adequate: Boolean indicating if gene passed all checks.
        reason: String code explaining failure (or OK).
        sector_counts: (B,) array of foreground cell counts per sector.
        sector_mask: (B,) boolean array, True if sector has >= min_count cells.
        n_foreground: Total number of foreground cells.
    """

    is_adequate: bool
    reason: str
    sector_counts: np.ndarray
    sector_mask: np.ndarray
    n_foreground: int


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
    y: np.ndarray, theta: np.ndarray, n_sectors: int, min_count: int = 1
) -> AdequacyReport:
    """
    Assess gene adequacy for RSP.

    Checks performed:
    - Presence of any foreground cells.
    - Each of the `n_sectors` angular sectors has at least `min_count` foreground cells.

    Args:
        y: (N,) boolean foreground indicator.
        theta: (N,) array of angles for ALL cells (radians, used for foreground cells only).
        n_sectors: Number of sectors.
        min_count: Minimum cells per sector.

    Returns:
        AdequacyReport object.
    """
    theta_fg = theta[y]
    n_fg = len(theta_fg)

    if n_fg == 0:
        return AdequacyReport(
            is_adequate=False,
            reason=REASON_GENE_UNDERPOWERED,
            sector_counts=np.zeros(n_sectors, dtype=int),
            sector_mask=np.zeros(n_sectors, dtype=bool),
            n_foreground=0,
        )

    counts = sector_counts(theta_fg, n_sectors)
    mask = sector_adequacy_mask(counts, min_count)
    all_sectors_ok = np.all(mask)

    reason = REASON_OK if all_sectors_ok else REASON_SECTOR_FG_TOO_SMALL

    return AdequacyReport(
        is_adequate=all_sectors_ok,
        reason=reason,
        sector_counts=counts,
        sector_mask=mask,
        n_foreground=n_fg,
    )


__all__ = ["AdequacyReport", "sector_counts", "sector_adequacy_mask", "gene_adequacy"]
