"""Robustness diagnostics module for BioRSP.

Implements stability checks via subsampling:
- Subsample cells (e.g. 80%)
- Recompute RSP profile
- Correlate with full profile
- Compute CV of scalar summaries
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.stats import pearsonr, spearmanr

from biorsp.core.engine import compute_rsp_radar
from biorsp.core.summaries import compute_scalar_summaries
from biorsp.preprocess.foreground import define_foreground
from biorsp.utils.config import BioRSPConfig


@dataclass
class RobustnessResult:
    """Results of robustness analysis.

    Attributes:
        mean_correlation: Mean Pearson correlation of subsampled profiles with the full profile.
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
    config: Optional[BioRSPConfig] = None,
    n_subsample: int = 20,
    subsample_frac: float = 0.8,
    seed: int = 42,
    fg_mode: str = "quantile",
    abs_threshold: Optional[float] = None,
    corr_method: str = "pearson",
) -> RobustnessResult:
    """Compute robustness metrics via subsampling.

    Args:
        x: Expression values of shape (N,).
        r: Radial distances of shape (N,).
        theta: Angles of shape (N,).
        config: Optional BioRSP configuration. Defaults are used if None.
        n_subsample: Number of subsampling iterations.
        subsample_frac: Fraction of cells to keep per iteration.
        seed: Random seed.
        fg_mode: Foreground selection mode.
        abs_threshold: Optional absolute threshold for foreground selection.

    Returns:
        RobustnessResult with mean correlation and anisotropy variability.
    """
    if config is None:
        config = BioRSPConfig()
    rng = np.random.default_rng(seed)
    n_cells = len(x)
    n_keep = int(n_cells * subsample_frac)

    y_full, _ = define_foreground(
        x, mode=fg_mode, q=config.foreground_quantile, abs_threshold=abs_threshold, rng=rng
    )
    if y_full is None:
        return RobustnessResult(mean_correlation=np.nan, cv_anisotropy=np.nan, n_subsamples=0)

    radar_full = compute_rsp_radar(r, theta, y_full, config=config)
    rsp_full = radar_full.rsp

    correlations = []
    anisotropies = []

    for _i in range(n_subsample):
        indices = rng.choice(n_cells, size=n_keep, replace=False)

        x_sub = x[indices]
        r_sub = r[indices]
        theta_sub = theta[indices]

        y_sub, _ = define_foreground(
            x_sub, mode=fg_mode, q=config.foreground_quantile, abs_threshold=abs_threshold, rng=rng
        )
        if y_sub is None:
            continue

        radar_sub = compute_rsp_radar(r_sub, theta_sub, y_sub, config=config)
        rsp_sub = radar_sub.rsp

        mask_full = (
            radar_full.geom_supported_mask
            if radar_full.geom_supported_mask is not None
            else np.isfinite(rsp_full)
        )
        mask_sub = (
            radar_sub.geom_supported_mask
            if radar_sub.geom_supported_mask is not None
            else np.isfinite(rsp_sub)
        )
        mask = mask_full & mask_sub & np.isfinite(rsp_sub) & np.isfinite(rsp_full)

        if np.sum(mask) < 2 or np.std(rsp_sub[mask]) == 0 or np.std(rsp_full[mask]) == 0:
            corr = np.nan
        else:
            if corr_method == "spearman":
                corr, _ = spearmanr(rsp_sub[mask], rsp_full[mask])
            else:
                corr, _ = pearsonr(rsp_sub[mask], rsp_full[mask])

        correlations.append(corr)

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
