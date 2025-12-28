"""
Robustness diagnostics module for BioRSP.

Implements stability checks via subsampling:
- Subsample cells (e.g. 80%)
- Recompute RSP profile
- Correlate with full profile
- Compute CV of scalar summaries
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import pearsonr

from .foreground import binary_foreground
from .radar import compute_rsp_radar
from .summaries import compute_scalar_summaries
from .constants import N_BG_MIN_DEFAULT, N_FG_MIN_DEFAULT


@dataclass
class RobustnessResult:
    """
    Results of robustness analysis.

    Attributes:
        mean_correlation: Mean Pearson correlation of subsampled RSP profiles with full profile.
        cv_anisotropy: Coefficient of variation of anisotropy across subsamples.
        n_subsamples: Number of subsamples performed.
    """

    mean_correlation: float
    cv_anisotropy: float
    n_subsamples: int


def compute_robustness_score(
    x: np.ndarray,
    r: np.ndarray,
    theta: np.ndarray,
    B: int = 360,
    delta_deg: float = 20.0,
    n_subsample: int = 20,
    subsample_frac: float = 0.8,
    seed: int = 42,
    min_fg_sector: int = N_FG_MIN_DEFAULT,
    min_bg_sector: int = N_BG_MIN_DEFAULT,
) -> RobustnessResult:
    """
    Compute robustness metrics via subsampling.

    Args:
        x: (N,) expression values.
        r: (N,) radial distances.
        theta: (N,) angles.
        B: Number of sectors.
        delta_deg: Sector width.
        n_subsample: Number of iterations.
        subsample_frac: Fraction of cells to keep per iteration.
        seed: Random seed.

    Returns:
        RobustnessResult object.
    """
    rng = np.random.default_rng(seed)
    n_cells = len(x)
    n_keep = int(n_cells * subsample_frac)

    # 1. Compute full profile
    y_full, _, _ = binary_foreground(x)
    # If full data is inadequate, robustness estimates may be unreliable
    radar_full = compute_rsp_radar(
        r, theta, y_full, B, delta_deg, min_fg_sector, min_bg_sector
    )
    rsp_full = radar_full.rsp

    correlations = []
    anisotropies = []

    for _ in range(n_subsample):
        # Subsample indices
        indices = rng.choice(n_cells, size=n_keep, replace=False)

        x_sub = x[indices]
        r_sub = r[indices]
        theta_sub = theta[indices]

        # Recompute foreground on subsample
        y_sub, _, _ = binary_foreground(x_sub)
        # Compute RSP
        radar_sub = compute_rsp_radar(
            r_sub, theta_sub, y_sub, B, delta_deg, min_fg_sector, min_bg_sector
        )
        rsp_sub = radar_sub.rsp

        mask = np.isfinite(rsp_sub) & np.isfinite(rsp_full)
        if np.sum(mask) < 2:
            corr = np.nan
        elif np.std(rsp_sub[mask]) == 0 or np.std(rsp_full[mask]) == 0:
            corr = 0.0
        else:
            corr, _ = pearsonr(rsp_sub[mask], rsp_full[mask])

        correlations.append(corr)

        # Compute anisotropy
        summ = compute_scalar_summaries(radar_sub)
        anisotropies.append(summ.rms_anisotropy)

    mean_corr = float(np.nanmean(correlations))

    mean_ani = float(np.nanmean(anisotropies))
    std_ani = float(np.nanstd(anisotropies))

    if not np.isfinite(mean_ani) or mean_ani <= 0:
        cv_ani = np.nan
    else:
        cv_ani = std_ani / mean_ani

    return RobustnessResult(
        mean_correlation=float(mean_corr),
        cv_anisotropy=float(cv_ani),
        n_subsamples=n_subsample,
    )


__all__ = ["RobustnessResult", "compute_robustness_score"]
