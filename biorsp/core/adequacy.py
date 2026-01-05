"""
Adequacy assessment for BioRSP.

This module provides logic to determine if a gene or feature has sufficient
spatial coverage and signal to be reliably analyzed by BioRSP.
"""

from typing import Optional

import numpy as np

from biorsp.core.typing import AdequacyReport
from biorsp.preprocess.geometry import get_sector_indices
from biorsp.utils.config import BioRSPConfig
from biorsp.utils.constants import (
    REASON_GENE_UNDERPOWERED,
    REASON_OK,
    REASON_SECTOR_BG_TOO_SMALL,
    REASON_SECTOR_FG_TOO_SMALL,
    REASON_SECTOR_MIXED_TOO_SMALL,
)
from biorsp.utils.helpers import weighted_quantile


def assess_adequacy(
    r: np.ndarray,
    theta: np.ndarray,
    y: np.ndarray,
    config: Optional[BioRSPConfig] = None,
    **kwargs,
) -> AdequacyReport:
    r"""
    Assess gene adequacy for BioRSP.

    A gene is considered adequate if it has:
    1. Sufficient total foreground cells (min_fg_total).
    2. Sufficient valid sectors (min_adequacy_fraction).

    A sector is valid if it has:
    1. Sufficient foreground cells (min_fg_sector).
    2. Sufficient background cells (min_bg_sector).
    3. Non-degenerate radial scale (scale >= min_scale).

    Parameters
    ----------
    r : np.ndarray
        (N,) array of normalized radial distances.
    theta : np.ndarray
        (N,) array of angles in radians.
    y : np.ndarray
        (N,) foreground weights or binary indicators.
    config : BioRSPConfig, optional
        Configuration object, by default BioRSPConfig().
    **kwargs
        Overrides for config parameters.

    Returns
    -------
    AdequacyReport
        Report on gene adequacy.
    """
    if config is None:
        # Handle n_sectors alias for B
        if "n_sectors" in kwargs:
            kwargs["B"] = kwargs.pop("n_sectors")
        config = BioRSPConfig(**kwargs)
    elif kwargs:
        from dataclasses import replace

        if "n_sectors" in kwargs:
            kwargs["B"] = kwargs.pop("n_sectors")
        config = replace(config, **kwargs)

    B = config.B
    sector_indices = get_sector_indices(theta, B, config.delta_deg)

    counts_fg = np.zeros(B)
    counts_bg = np.zeros(B)
    scale_mask = np.ones(B, dtype=bool)

    for b in range(B):
        idx = sector_indices[b]
        if idx.size > 0:
            y_s = y[idx]
            counts_fg[b] = np.sum(y_s)
            counts_bg[b] = np.sum(1.0 - y_s)

            # Scale guard
            if config.min_scale > 0:
                r_s = r[idx]
                if config.scale_mode == "pooled_iqr":
                    scale = np.percentile(r_s, 75) - np.percentile(r_s, 25)
                elif config.scale_mode == "bg_iqr":
                    w_bg = 1.0 - y_s
                    scale = weighted_quantile(r_s, w_bg, 0.75) - weighted_quantile(r_s, w_bg, 0.25)
                elif config.scale_mode == "fg_iqr":
                    w_fg = y_s
                    scale = weighted_quantile(r_s, w_fg, 0.75) - weighted_quantile(r_s, w_fg, 0.25)
                elif config.scale_mode == "pooled_mad":
                    med = np.median(r_s)
                    scale = 1.4826 * np.median(np.abs(r_s - med))
                else:
                    scale = 0.0

                if not np.isfinite(scale) or scale < config.min_scale:
                    scale_mask[b] = False

    fg_mask = counts_fg >= config.min_fg_sector
    bg_mask = counts_bg >= config.min_bg_sector
    sector_mask = fg_mask & bg_mask & scale_mask

    n_fg_total = np.sum(y)
    n_bg_total = np.sum(1.0 - y)
    adequacy_fraction = np.mean(sector_mask)

    is_adequate = (
        n_fg_total >= config.min_fg_total and adequacy_fraction >= config.min_adequacy_fraction
    )

    if n_fg_total < config.min_fg_total:
        reason = REASON_GENE_UNDERPOWERED
    elif is_adequate:
        reason = REASON_OK
    elif not np.any(fg_mask):
        reason = REASON_SECTOR_FG_TOO_SMALL
    elif not np.any(bg_mask):
        reason = REASON_SECTOR_BG_TOO_SMALL
    else:
        reason = REASON_SECTOR_MIXED_TOO_SMALL

    return AdequacyReport(
        is_adequate=is_adequate,
        reason=reason,
        counts_fg=counts_fg,
        counts_bg=counts_bg,
        sector_mask=sector_mask,
        n_foreground=float(n_fg_total),
        n_background=float(n_bg_total),
        adequacy_fraction=float(adequacy_fraction),
        sector_indices=sector_indices,
    )
