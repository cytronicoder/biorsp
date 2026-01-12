"""Core data models for BioRSP.

RadarResult contract:
- rsp: RAW unweighted signed statistic profile R(θ)
- sector_weights: for aggregation only, applied in scoring/summaries ONCE
- geom_supported_mask: sector has sufficient total density and valid scale
- contrast_supported_mask: sector has sufficient FG and BG for contrast
- forced_zero_mask: geometry-supported but n_fg==0 and empty_fg_policy="zero"
- invalid_reason: per-sector reason strings for diagnostics

See docs/theory.md for detailed semantics.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from biorsp.utils.config import BioRSPConfig


@dataclass
class RadarResult:
    """Result of the R(θ) radar function computation.

    RadarResult contract:
    - rsp is RAW (unweighted). Weighting is applied ONCE in scoring/summaries.
    - geom_supported_mask indicates geometry validity (total support + scale).
    - contrast_supported_mask indicates FG/BG contrast validity.
    - forced_zero_mask indicates sectors where rsp=0 due to empty_fg_policy="zero".

    Attributes:
        rsp: (B,) array of RAW signed RSP values. NaNs indicate unsupported sectors.
        counts_fg: (B,) array of foreground mass per sector.
        counts_bg: (B,) array of background mass per sector.
        counts_total: (B,) array of total cell count per sector.
        centers: (B,) array of sector center angles in radians (math convention).
        iqr_floor: Stability floor used for IQR normalization.
        iqr_floor_hits: (B,) boolean array indicating floor was applied.
        sector_weights: (B,) weights for aggregation (never baked into rsp).
        normalization_stats: Metadata from radial normalization.
        denom_scale: (B,) robust scale per sector (IQR/MAD).
        geom_supported_mask: (B,) bool - sector has sufficient total density + valid scale.
        contrast_supported_mask: (B,) bool - sector has sufficient FG+BG for contrast.
        forced_zero_mask: (B,) bool - geom-supported but n_fg==0 with zero policy.
        invalid_reason: (B,) list of reason strings for each sector.

    """

    rsp: np.ndarray
    counts_fg: np.ndarray
    counts_bg: np.ndarray
    centers: np.ndarray
    iqr_floor: float
    iqr_floor_hits: np.ndarray
    counts_total: Optional[np.ndarray] = None
    sector_weights: Optional[np.ndarray] = None
    normalization_stats: Dict = field(default_factory=dict)
    denom_scale: Optional[np.ndarray] = None
    geom_supported_mask: Optional[np.ndarray] = None
    contrast_supported_mask: Optional[np.ndarray] = None
    forced_zero_mask: Optional[np.ndarray] = None
    invalid_reason: Optional[List[str]] = None

    def __post_init__(self):
        """Derive counts_total and sector_weights if not provided."""
        B = len(self.rsp)
        if self.counts_total is None:
            self.counts_total = self.counts_fg + self.counts_bg
        if self.sector_weights is None:
            self.sector_weights = np.ones(B)


@dataclass
class AdequacyReport:
    """Report on gene adequacy for BioRSP computation.

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
    """Result of statistical inference (permutation test).

    Attributes:
        p_value: Finite-permutation corrected p-value.
        observed_stat: Observed spatial organization score S_g.
        null_stats: (n_perm,) array of null statistics.
        valid_mask: (B,) boolean mask of sectors used for S_g computation.
        q_value: FDR-corrected p-value (if computed).
        warnings: List of warning strings.

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
    warnings: Optional[List[str]] = None


__all__ = [
    "BioRSPConfig",
    "RadarResult",
    "AdequacyReport",
    "InferenceResult",
]
