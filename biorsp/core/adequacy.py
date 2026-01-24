"""Adequacy assessment for BioRSP.

This module provides the single abstention authority for BioRSP. If a feature
is inadequate, downstream statistics should be NaN and p/q-values should be None.
"""

from typing import Optional

import numpy as np

from biorsp.core.engine import SectorIndex
from biorsp.core.geometry import get_sector_indices
from biorsp.core.qc import compute_gene_qc, compute_sector_qc
from biorsp.core.typing import AdequacyReport
from biorsp.utils.config import BioRSPConfig
from biorsp.utils.constants import (
    REASON_FOREGROUND_TIE_UNSTABLE,
    REASON_OK,
    REASON_SECTOR_LOW_TOTAL,
)


def compute_effective_sample_size(weights: np.ndarray, eps: float = 1e-12) -> float:
    """Compute Kish effective sample size for weighted foreground/background."""
    sum_w = np.sum(weights)
    sum_w2 = np.sum(weights**2)
    if sum_w2 <= eps:
        return 0.0
    return float((sum_w**2) / sum_w2)


def assess_adequacy(
    r: np.ndarray,
    theta: np.ndarray,
    y: np.ndarray,
    config: Optional[BioRSPConfig] = None,
    x: Optional[np.ndarray] = None,
    sector_index: Optional[SectorIndex] = None,
    **kwargs,
) -> AdequacyReport:
    r"""Assess gene adequacy for BioRSP.

    Uses effective sample sizes and coverage-based criteria.

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
    if sector_index is not None and sector_index.sector_indices is not None:
        sector_indices = sector_index.sector_indices
    else:
        sector_indices = get_sector_indices(theta, B, config.delta_deg)

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
                metrics={
                    "min_unique_fg": config.min_unique_foreground_values,
                    "n_unique_fg": float(n_unique_fg),
                },
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
        total_cells = idx.size
        min_total = min(
            config.min_total_per_sector,
            float(config.min_fg_sector + config.min_bg_sector),
        )
        if total_cells < min_total:
            sector_reasons[b] = REASON_SECTOR_LOW_TOTAL
            continue

        if config.scale_mode == "u_space":
            denom = 1.0 if counts_bg[b] > 0 else 0.0
        else:
            from biorsp.core.engine import sector_signed_stat

            res = sector_signed_stat(r, y, idx, config=config)
            denom = res["denom"]

        valid, reason, _ = compute_sector_qc(y_s, denom, config)
        sector_mask[b] = valid
        sector_reasons[b] = reason

    total_fg = np.sum(y)
    is_adequate, gene_reason, metrics = compute_gene_qc(
        sector_mask, sector_reasons, total_fg, config
    )
    adequacy_fraction = metrics["coverage"]
    if config.foreground_mode == "weights":
        metrics["K_eff_fg"] = compute_effective_sample_size(y)
        metrics["K_eff_bg"] = compute_effective_sample_size(1.0 - y)

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
        metrics={
            "min_fg_sector": float(config.min_fg_sector),
            "min_bg_sector": float(config.min_bg_sector),
            "min_valid_sectors": float(config.min_valid_sectors),
            "min_coverage": float(config.min_coverage),
            "min_fg_total": float(config.min_fg_total),
            "n_foreground": float(total_fg),
            "n_background": float(np.sum(1.0 - y)),
        },
    )
