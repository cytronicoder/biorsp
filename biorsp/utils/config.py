"""
Configuration module for BioRSP.

Immutable configuration object that maps 1:1 to Methods parameters.
    It is stored in the run manifest.
"""

from dataclasses import asdict, dataclass
from typing import Literal, Optional

import numpy as np

from biorsp.utils.constants import (
    ADEQUACY_FRACTION_DEFAULT,
    B_DEFAULT,
    DELTA_DEG_DEFAULT,
    K_EXPLORATORY_DEFAULT,
    MIN_BG_EFF_DEFAULT,
    MIN_COVERAGE_DEFAULT,
    MIN_FG_EFF_DEFAULT,
    MIN_STRATUM_SIZE_DEFAULT,
    MIN_TOTAL_MF_DEFAULT,
    MIN_VALID_SECTORS_DEFAULT,
    N_BG_MIN_DEFAULT,
    N_FG_MIN_DEFAULT,
    N_FG_TOT_MIN_DEFAULT,
    N_R_BINS_DEFAULT,
    N_THETA_BINS_DEFAULT,
    SMOOTH_DEG_DEFAULT,
    UMI_BINS_DEFAULT,
)


@dataclass(frozen=True)
class BioRSPConfig:
    """
    Configuration for BioRSP analysis.

    Centralizes all parameters for geometry, foreground definition,
    adequacy thresholds, permutation tests, and normalization.
    """

    # Geometry
    B: int = B_DEFAULT
    delta_deg: float = DELTA_DEG_DEFAULT
    vantage: Literal["geometric_median", "mean", "user"] = "geometric_median"
    geom_median_tol: float = 1e-5
    geom_median_max_iter: int = 100

    # Foreground definition
    foreground_mode: Literal["quantile", "absolute", "auto", "weights"] = "quantile"
    foreground_quantile: float = 0.90
    foreground_threshold: Optional[float] = None

    # Adequacy thresholds
    qc_mode: Literal["legacy", "principled"] = "principled"
    min_fg_sector: float = N_FG_MIN_DEFAULT
    min_bg_sector: float = N_BG_MIN_DEFAULT
    min_fg_total: float = N_FG_TOT_MIN_DEFAULT
    min_adequacy_fraction: float = ADEQUACY_FRACTION_DEFAULT

    # Principled QC thresholds
    min_fg_eff: float = MIN_FG_EFF_DEFAULT
    min_bg_eff: float = MIN_BG_EFF_DEFAULT
    min_total_mF: float = MIN_TOTAL_MF_DEFAULT
    min_coverage: float = MIN_COVERAGE_DEFAULT
    min_valid_sectors: int = MIN_VALID_SECTORS_DEFAULT

    # Permutation parameters
    perm_mode: Literal["radial", "joint", "rt_umi", "none"] = "radial"
    n_permutations: int = K_EXPLORATORY_DEFAULT
    umi_bins: int = UMI_BINS_DEFAULT
    n_r_bins: int = N_R_BINS_DEFAULT
    n_theta_bins: int = N_THETA_BINS_DEFAULT
    seed: int = 0
    donor_stratify: bool = False
    min_stratum_size: int = MIN_STRATUM_SIZE_DEFAULT

    # Normalization and Scaling
    scale_mode: Literal["pooled_iqr", "bg_iqr", "fg_iqr", "pooled_mad"] = "pooled_iqr"
    min_scale: float = 1e-3
    iqr_floor_pct: float = 0.1
    smoothing_deg: float = SMOOTH_DEG_DEFAULT
    sign_tol: float = 0.0

    # Sector weighting
    sector_weight_mode: Literal["none", "sqrt_frac", "effective_min", "logistic_support"] = "none"
    sector_weight_k: float = 5.0

    # Output options
    save_profiles: bool = True
    save_plots: bool = True

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
