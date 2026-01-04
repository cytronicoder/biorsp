"""
Configuration module for BioRSP.

Immutable configuration object that maps 1:1 to Methods parameters.
    It is stored in the run manifest.
"""

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np

from .constants import (
    ADEQUACY_FRACTION_DEFAULT,
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

    B: int = B_DEFAULT
    delta_deg: float = DELTA_DEG_DEFAULT
    min_fg_sector: float = N_FG_MIN_DEFAULT
    min_bg_sector: float = N_BG_MIN_DEFAULT
    min_fg_total: float = N_FG_TOT_MIN_DEFAULT
    min_adequacy_fraction: float = ADEQUACY_FRACTION_DEFAULT
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
    iqr_floor_pct: float = 0.1

    @property
    def n_angles(self) -> int:
        """Backward compatible alias for B."""
        return self.B

    @property
    def sector_width_deg(self) -> float:
        """Backward compatible alias for delta_deg."""
        return self.delta_deg

    @property
    def sector_width_rad(self) -> float:
        """Sector width in radians."""
        return np.deg2rad(self.sector_width_deg)

    def to_dict(self) -> dict:
        """Convert to dictionary for manifest writing."""
        return asdict(self)


__all__ = ["BioRSPConfig"]
