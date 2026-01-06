"""
Adequacy assessment for BioRSP.

This module provides logic to determine if a gene or feature has sufficient
spatial coverage and signal to be reliably analyzed by BioRSP.
"""

from typing import Optional

import numpy as np

from biorsp.core.qc import compute_gene_qc, compute_sector_qc
from biorsp.core.typing import AdequacyReport
from biorsp.preprocess.geometry import get_sector_indices
from biorsp.utils.config import BioRSPConfig
from biorsp.utils.constants import (
    REASON_FOREGROUND_TIE_UNSTABLE,
    REASON_GENE_UNDERPOWERED,
    REASON_OK,
    REASON_SECTOR_BG_TOO_SMALL,
    REASON_SECTOR_DEGENERATE_SCALE,
    REASON_SECTOR_FG_TOO_SMALL,
    REASON_SECTOR_MIXED_TOO_SMALL,
)


def assess_adequacy(
    r: np.ndarray,
    theta: np.ndarray,
    y: np.ndarray,
    config: Optional[BioRSPConfig] = None,
    x: Optional[np.ndarray] = None,
    **kwargs,
) -> AdequacyReport:
    r"""
    Assess gene adequacy for BioRSP.

    In 'principled' mode (default), uses effective sample sizes and coverage.
    In 'legacy' mode, uses hard count thresholds.

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
    x : np.ndarray, optional
        (N,) original expression values to check for tie-instability.
    **kwargs
        Overrides for config parameters.

    Returns
    -------
    AdequacyReport
        Report on gene adequacy.
    """
    if config is None:
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

    # Ensure identifiability for sparse or tied features by checking unique foreground values.
    if x is not None:
        fg_vals = x[y > 0]
        n_unique_fg = len(np.unique(fg_vals))
        if n_unique_fg < config.min_unique_foreground_values:
            return AdequacyReport(
                is_adequate=False,
                reason=REASON_FOREGROUND_TIE_UNSTABLE,
                counts_fg=np.zeros(B),
                counts_bg=np.zeros(B),
                sector_mask=np.zeros(B, dtype=bool),
                n_foreground=float(np.sum(y)),
                n_background=float(np.sum(1.0 - y)),
                adequacy_fraction=0.0,
                sector_indices=sector_indices,
                sector_reasons=[REASON_FOREGROUND_TIE_UNSTABLE] * B,
            )

    counts_fg = np.zeros(B)
    counts_bg = np.zeros(B)
    sector_mask = np.zeros(B, dtype=bool)
    sector_reasons = [REASON_OK] * B

    for b in range(B):
        idx = sector_indices[b]
        if idx.size == 0:
            sector_reasons[b] = "empty_sector"
            continue

        y_s = y[idx]
        counts_fg[b] = np.sum(y_s)
        counts_bg[b] = np.sum(1.0 - y_s)

        # Normalize scale by CDF (U-space) or legacy denominator to ensure comparability across sectors.
        if config.scale_mode == "u_space":
            denom = 1.0 if counts_bg[b] > 0 else 0.0
        else:
            from biorsp.core.engine import sector_signed_stat

            res = sector_signed_stat(r, y, idx, config=config)
            denom = res["denom"]

        if config.qc_mode == "principled":
            valid, reason, _ = compute_sector_qc(y_s, denom, config)
            sector_mask[b] = valid
            sector_reasons[b] = reason
        else:
            fg_ok = counts_fg[b] >= config.min_fg_sector
            bg_ok = counts_bg[b] >= config.min_bg_sector
            scale_ok = denom >= config.min_scale
            sector_mask[b] = fg_ok and bg_ok and scale_ok
            if not fg_ok:
                sector_reasons[b] = REASON_SECTOR_FG_TOO_SMALL
            elif not bg_ok:
                sector_reasons[b] = REASON_SECTOR_BG_TOO_SMALL
            elif not scale_ok:
                sector_reasons[b] = REASON_SECTOR_DEGENERATE_SCALE

    total_fg = np.sum(y)
    if config.qc_mode == "principled":
        is_adequate, gene_reason, metrics = compute_gene_qc(
            sector_mask, sector_reasons, total_fg, config
        )
        adequacy_fraction = metrics["coverage"]
    else:
        adequacy_fraction = np.mean(sector_mask)
        is_adequate = (total_fg >= config.min_fg_total) and (
            adequacy_fraction >= config.min_adequacy_fraction
        )
        if total_fg < config.min_fg_total:
            gene_reason = REASON_GENE_UNDERPOWERED
        elif is_adequate:
            gene_reason = REASON_OK
        elif not np.any(sector_mask):
            gene_reason = REASON_SECTOR_MIXED_TOO_SMALL
        else:
            gene_reason = REASON_GENE_UNDERPOWERED

    return AdequacyReport(
        is_adequate=is_adequate,
        reason=gene_reason,
        counts_fg=counts_fg,
        counts_bg=counts_bg,
        sector_mask=sector_mask,
        n_foreground=float(total_fg),
        n_background=float(np.sum(1.0 - y)),
        adequacy_fraction=float(adequacy_fraction),
        sector_indices=sector_indices,
        sector_reasons=sector_reasons,
    )
