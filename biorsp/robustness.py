"""
Robustness diagnostics module for BioRSP.

Implements stability checks via subsampling:
- Subsample cells (e.g. 80%)
- Recompute RSP profile
- Correlate with full profile
- Compute CV of scalar summaries
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.stats import pearsonr

from .config import BioRSPConfig
from .core import compute_rsp_radar
from .preprocessing import define_foreground
from .summaries import compute_scalar_summaries


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
    config: BioRSPConfig = BioRSPConfig(),
    n_subsample: int = 20,
    subsample_frac: float = 0.8,
    seed: int = 42,
    fg_mode: str = "quantile",
    abs_threshold: Optional[float] = None,
) -> RobustnessResult:
    """
    Compute robustness metrics via subsampling.

    Parameters
    ----------
    x : np.ndarray
        (N,) expression values.
    r : np.ndarray
        (N,) radial distances.
    theta : np.ndarray
        (N,) angles.
    config : BioRSPConfig, optional
        Configuration object, by default BioRSPConfig().
    n_subsample : int, optional
        Number of iterations, by default 20.
    subsample_frac : float, optional
        Fraction of cells to keep per iteration, by default 0.8.
    seed : int, optional
        Random seed, by default 42.
    fg_mode : str, optional
        Foreground selection mode, by default "quantile".
    abs_threshold : float, optional
        Absolute threshold for foreground, by default None.

    Returns
    -------
    RobustnessResult
        The result of the robustness analysis.
    """
    rng = np.random.default_rng(seed)
    n_cells = len(x)
    n_keep = int(n_cells * subsample_frac)

    # 1. Compute full profile
    y_full, _ = define_foreground(
        x, mode=fg_mode, q=config.foreground_quantile, abs_threshold=abs_threshold, rng=rng
    )
    if y_full is None:
        return RobustnessResult(mean_correlation=np.nan, cv_anisotropy=np.nan, n_subsamples=0)

    # If full data is inadequate, robustness estimates may be unreliable
    radar_full = compute_rsp_radar(r, theta, y_full, config=config)
    rsp_full = radar_full.rsp

    correlations = []
    anisotropies = []

    for i in range(n_subsample):
        # Subsample indices
        indices = rng.choice(n_cells, size=n_keep, replace=False)

        x_sub = x[indices]
        r_sub = r[indices]
        theta_sub = theta[indices]

        # Recompute foreground on subsample
        # Use the same rng to ensure deterministic but varied tie-breaking
        y_sub, _ = define_foreground(
            x_sub, mode=fg_mode, q=config.foreground_quantile, abs_threshold=abs_threshold, rng=rng
        )
        if y_sub is None:
            continue

        # Compute RSP
        radar_sub = compute_rsp_radar(r_sub, theta_sub, y_sub, config=config)
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
        anisotropies.append(summ.anisotropy)

    mean_corr = float(np.nanmean(correlations))

    mean_ani = float(np.nanmean(anisotropies))
    std_ani = float(np.nanstd(anisotropies))

    if not np.isfinite(mean_ani):
        cv_ani = np.nan
    elif mean_ani == 0:
        cv_ani = 0.0
    else:
        cv_ani = std_ani / mean_ani

    return RobustnessResult(
        mean_correlation=float(mean_corr),
        cv_anisotropy=float(cv_ani),
        n_subsamples=n_subsample,
    )


__all__ = ["RobustnessResult", "compute_robustness_score"]
