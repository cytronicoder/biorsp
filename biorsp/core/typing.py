"""
Core data models for BioRSP.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from biorsp.utils.config import BioRSPConfig


@dataclass
class RadarResult:
    """
    Result of the R(theta) radar function computation.

    Attributes:
        rsp: (B,) array of signed RSP values. NaNs indicate sectors with insufficient background support.
        counts_fg: (B,) array of foreground mass per sector.
        counts_bg: (B,) array of background mass per sector.
        centers: (B,) array of sector center angles in radians.
        iqr_floor: The stability floor used for IQR normalization.
        iqr_floor_hits: (B,) boolean array indicating where the floor was applied.
        normalization_stats: Metadata from radial normalization.
        n_fg_per_sector: (B,) array of literal foreground cell counts.
        n_bg_per_sector: (B,) array of literal background cell counts.
        denom_scale_per_sector: (B,) array of robust scale denominators used.
        bg_supported_mask: (B,) boolean array indicating background-supported sectors.
    """

    rsp: np.ndarray
    counts_fg: np.ndarray
    counts_bg: np.ndarray
    centers: np.ndarray
    iqr_floor: float
    iqr_floor_hits: np.ndarray
    sector_weights: Optional[np.ndarray] = None
    normalization_stats: Dict = field(default_factory=dict)
    n_fg_per_sector: Optional[np.ndarray] = None
    n_bg_per_sector: Optional[np.ndarray] = None
    denom_scale_per_sector: Optional[np.ndarray] = None
    bg_supported_mask: Optional[np.ndarray] = None


@dataclass
class AdequacyReport:
    """
    Report on gene adequacy for BioRSP computation.

    Attributes:
        is_adequate: Whether the gene meets all adequacy criteria.
        reason: Explanation for failure (or "ok").
        counts_fg: (B,) array of foreground mass per sector.
        counts_bg: (B,) array of background mass per sector.
        sector_mask: (B,) boolean mask of adequate sectors.
        n_foreground: Total foreground mass.
        n_background: Total background mass.
        adequacy_fraction: Fraction of sectors that are adequate.
        sector_indices: List of arrays containing cell indices for each sector window.
    """

    is_adequate: bool
    reason: str
    counts_fg: np.ndarray
    counts_bg: np.ndarray
    sector_mask: np.ndarray
    n_foreground: float
    n_background: float
    adequacy_fraction: float
    sector_indices: Optional[List[np.ndarray]] = None
    sector_reasons: Optional[List[str]] = None


@dataclass
class InferenceResult:
    """
    Result of statistical inference (permutation test).

    Attributes:
        p_value: Finite-permutation corrected p-value.
        observed_stat: Observed RMS anisotropy.
        null_stats: (n_perm,) array of null statistics.
        valid_mask: (B,) boolean mask of sectors used for anisotropy.
        q_value: FDR-corrected p-value (if computed).
    """

    p_value: float
    observed_stat: float
    null_stats: np.ndarray
    valid_mask: np.ndarray
    seeds: Optional[np.ndarray] = None
    q_value: Optional[float] = None
    perm_mode: Optional[str] = None
    K_eff: Optional[int] = None
    rejection_count: int = 0
    empty_sector_count: Optional[int] = None


__all__ = [
    "BioRSPConfig",
    "RadarResult",
    "AdequacyReport",
    "InferenceResult",
]
