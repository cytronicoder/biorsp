"""Configuration module for BioRSP.

Immutable configuration object that maps 1:1 to Methods parameters.
    It is stored in the run manifest.
"""

from dataclasses import asdict, dataclass
from typing import Literal, Optional

import numpy as np

from biorsp.utils.constants import (
    ADEQUACY_FRACTION_DEFAULT,
    B_DEFAULT,
    CENTER_DENSITY_PERCENTILE_DEFAULT,
    CENTER_KNN_K_DEFAULT,
    DELTA_DEG_DEFAULT,
    K_EXPLORATORY_DEFAULT,
    MIN_BG_EFF_DEFAULT,
    MIN_COVERAGE_DEFAULT,
    MIN_FG_EFF_DEFAULT,
    MIN_STRATUM_SIZE_DEFAULT,
    MIN_TOTAL_MF_DEFAULT,
    MIN_TOTAL_PER_SECTOR_DEFAULT,
    MIN_UNIQUE_FG_DEFAULT,
    MIN_VALID_SECTORS_DEFAULT,
    N_BG_MIN_DEFAULT,
    N_FG_MIN_DEFAULT,
    N_FG_TOT_MIN_DEFAULT,
    N_R_BINS_DEFAULT,
    N_THETA_BINS_DEFAULT,
    REJECTION_MAX_RETRIES_DEFAULT,
    SMOOTH_DEG_DEFAULT,
    UMI_BINS_DEFAULT,
)


@dataclass(frozen=True)
class BioRSPConfig:
    """Configuration for BioRSP analysis.

    Centralizes all parameters for geometry, foreground definition,
    adequacy thresholds, permutation tests, and normalization.
    """

    B: int = B_DEFAULT
    delta_deg: float = DELTA_DEG_DEFAULT
    vantage: Literal["geometric_median", "mean", "user"] = "geometric_median"
    geom_median_tol: float = 1e-5
    geom_median_max_iter: int = 100
    center_knn_k: int = CENTER_KNN_K_DEFAULT
    center_density_percentile: float = CENTER_DENSITY_PERCENTILE_DEFAULT

    expr_threshold_mode: Literal["detect", "fixed", "nonzero_quantile"] = "detect"
    expr_threshold_value: Optional[float] = None
    nonzero_quantile: float = 0.25
    coverage_use_raw: bool = False

    foreground_mode: Literal["quantile", "absolute", "auto", "weights"] = "quantile"
    foreground_quantile: float = 0.90
    foreground_threshold: Optional[float] = None
    min_unique_foreground_values: int = MIN_UNIQUE_FG_DEFAULT

    min_fg_sector: float = N_FG_MIN_DEFAULT
    min_bg_sector: float = N_BG_MIN_DEFAULT
    min_fg_total: float = N_FG_TOT_MIN_DEFAULT
    min_adequacy_fraction: float = ADEQUACY_FRACTION_DEFAULT

    min_fg_eff: float = MIN_FG_EFF_DEFAULT
    min_bg_eff: float = MIN_BG_EFF_DEFAULT
    min_total_mF: float = MIN_TOTAL_MF_DEFAULT
    min_coverage: float = MIN_COVERAGE_DEFAULT
    min_valid_sectors: int = MIN_VALID_SECTORS_DEFAULT
    min_total_per_sector: int = MIN_TOTAL_PER_SECTOR_DEFAULT

    perm_mode: Literal["radial", "joint", "rt_umi", "none"] = "radial"
    perm_mode_scoring: Literal["global", "within_cluster", "knn_block"] = "global"
    perm_cluster_key: Optional[str] = None
    knn_k: int = 15
    knn_block_size: int = 50
    n_permutations: int = K_EXPLORATORY_DEFAULT
    rejection_max_retries: int = REJECTION_MAX_RETRIES_DEFAULT
    umi_bins: int = UMI_BINS_DEFAULT
    n_r_bins: int = N_R_BINS_DEFAULT
    n_theta_bins: int = N_THETA_BINS_DEFAULT
    seed: int = 0
    donor_stratify: bool = False
    stratify_key: Optional[str] = None
    n_strata: int = 10
    min_stratum_size: int = MIN_STRATUM_SIZE_DEFAULT

    scale_mode: Literal["u_space", "pooled_iqr", "bg_iqr", "fg_iqr", "pooled_mad"] = "u_space"
    min_scale: float = 1e-3
    iqr_floor_pct: float = 0.1
    smoothing_deg: float = SMOOTH_DEG_DEFAULT
    sign_tol: float = 0.0
    effect_floor: float = 0.05

    sector_weight_mode: Literal["none", "sqrt_frac", "effective_min", "logistic_support"] = "none"
    sector_weight_k: float = 5.0

    min_shared_mask_fraction: float = 0.6
    empty_fg_policy: Literal["nan", "zero"] = "zero"

    radius_norm: Literal["max", "quantile", "std"] = "quantile"
    radius_q: float = 0.99
    radial_rule: Literal["equal", "quantile"] = "quantile"
    fixed_vantage: Optional[np.ndarray] = None

    save_profiles: bool = True
    save_plots: bool = True

    @property
    def sector_width_rad(self) -> float:
        """Sector width in radians."""
        return np.deg2rad(self.delta_deg)

    def to_dict(self) -> dict:
        """Convert to dictionary for manifest writing."""
        return asdict(self)


__all__ = ["BioRSPConfig"]
