"""
Configuration module for BioRSP.

Immutable configuration object that maps 1:1 to Methods parameters.
    It is stored in the run manifest.
"""

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np

from .constants import (
    B_DEFAULT,
    DELTA_DEG_DEFAULT,
    K_EXPLORATORY_DEFAULT,
    N_BG_MIN_DEFAULT,
    N_FG_MIN_DEFAULT,
    N_FG_TOT_MIN_DEFAULT,
    SMOOTH_DEG_DEFAULT,
    UMI_BINS_DEFAULT,
)


@dataclass(frozen=True)
class BioRSPConfig:
    """
    Configuration for BioRSP analysis.

    Maps directly to Methods parameters.
    """

    n_angles: int = B_DEFAULT
    sector_width_deg: float = DELTA_DEG_DEFAULT
    min_fg_sector: int = N_FG_MIN_DEFAULT
    min_bg_sector: int = N_BG_MIN_DEFAULT
    min_fg_total: int = N_FG_TOT_MIN_DEFAULT
    umi_bins: int = UMI_BINS_DEFAULT
    n_permutations: int = K_EXPLORATORY_DEFAULT
    smoothing_deg: float = SMOOTH_DEG_DEFAULT
    vantage: Literal["geometric_median", "mean", "user"] = "geometric_median"
    geom_median_tol: float = 1e-5
    geom_median_max_iter: int = 100
    seed: int = 0
    donor_stratify: bool = False
    min_stratum_size: int = 50
    foreground_quantile: float = 0.90

    @property
    def sector_width_rad(self) -> float:
        """Sector width in radians."""
        return np.deg2rad(self.sector_width_deg)

    def to_dict(self) -> dict:
        """Convert to dictionary for manifest writing."""
        return asdict(self)


__all__ = ["BioRSPConfig"]
