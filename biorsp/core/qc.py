"""
Principled Quality Control (QC) for BioRSP.

This module implements defensible QC criteria for both binary and weighted
foreground modes, ensuring consistent and reproducible adequacy assessment.
"""

# Use TYPE_CHECKING to avoid import cycles at runtime
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd

from biorsp.utils.config import BioRSPConfig
from biorsp.utils.constants import (
    EPS,
    REASON_GENE_LOW_COVERAGE,
    REASON_GENE_TOO_FEW_SECTORS,
    REASON_GENE_UNDERPOWERED,
    REASON_OK,
    REASON_SECTOR_BG_TOO_SMALL,
    REASON_SECTOR_DEGENERATE_SCALE,
    REASON_SECTOR_FG_TOO_SMALL,
)

if TYPE_CHECKING:
    from biorsp.core.results import FeatureResult


def kish_effective_sample_size(w: np.ndarray, eps: float = EPS) -> float:
    """
    Compute Kish's effective sample size for a set of weights.

    n_eff = (sum(w)^2) / (sum(w^2))

    Parameters
    ----------
    w : np.ndarray
        Weights.
    eps : float, optional
        Small value for stability, by default EPS.

    Returns
    -------
    float
        Effective sample size.
    """
    sum_w = np.sum(w)
    sum_w2 = np.sum(w**2)
    if sum_w2 < eps:
        return 0.0
    return (sum_w**2) / sum_w2


def compute_sector_qc(
    y_s: np.ndarray,
    denom: float,
    config: BioRSPConfig,
) -> Tuple[bool, str, Dict[str, float]]:
    """
    Compute QC status for a single sector.

    Parameters
    ----------
    y_s : np.ndarray
        Foreground weights/indicators for cells in the sector.
    denom : float
        Robust scale denominator for the sector.
    config : BioRSPConfig
        Configuration object.

    Returns
    -------
    Tuple[bool, str, Dict[str, float]]
        (is_valid, reason, metrics)
    """
    w_fg = y_s
    w_bg = 1.0 - y_s

    if config.foreground_mode == "weights":
        nF_eff = kish_effective_sample_size(w_fg)
        nB_eff = kish_effective_sample_size(w_bg)
        support_ok = (nF_eff >= config.min_fg_eff) and (nB_eff >= config.min_bg_eff)
        metrics = {"nF_eff": nF_eff, "nB_eff": nB_eff}
    else:
        nF = np.sum(w_fg)
        nB = np.sum(w_bg)
        support_ok = (nF >= config.min_fg_sector) and (nB >= config.min_bg_sector)
        metrics = {"nF": nF, "nB": nB}

    scale_ok = denom >= config.min_scale

    if not support_ok:
        if config.foreground_mode == "weights":
            reason = (
                REASON_SECTOR_FG_TOO_SMALL
                if metrics["nF_eff"] < config.min_fg_eff
                else REASON_SECTOR_BG_TOO_SMALL
            )
        else:
            reason = (
                REASON_SECTOR_FG_TOO_SMALL
                if metrics["nF"] < config.min_fg_sector
                else REASON_SECTOR_BG_TOO_SMALL
            )
        return False, reason, metrics

    if not scale_ok:
        return False, REASON_SECTOR_DEGENERATE_SCALE, metrics

    return True, REASON_OK, metrics


def compute_gene_qc(
    sector_valid_mask: np.ndarray,
    sector_reasons: List[str],
    total_fg_support: float,
    config: BioRSPConfig,
) -> Tuple[bool, str, Dict[str, Union[float, int]]]:
    """
    Compute QC status for a gene based on sector-level results.

    Parameters
    ----------
    sector_valid_mask : np.ndarray
        (B,) boolean mask of valid sectors.
    sector_reasons : List[str]
        List of reasons for each sector's QC status.
    total_fg_support : float
        Total foreground mass (sum of y).
    config : BioRSPConfig
        Configuration object.

    Returns
    -------
    Tuple[bool, str, Dict[str, Union[float, int]]]
        (is_adequate, reason, metrics)
    """
    B = len(sector_valid_mask)
    M_valid = int(np.sum(sector_valid_mask))
    coverage = M_valid / B

    metrics = {
        "M_valid": M_valid,
        "coverage": coverage,
        "total_fg_support": total_fg_support,
    }

    if config.foreground_mode == "weights":
        if total_fg_support < config.min_total_mF:
            return False, REASON_GENE_UNDERPOWERED, metrics
    else:
        if total_fg_support < config.min_fg_total:
            return False, REASON_GENE_UNDERPOWERED, metrics

    if coverage < config.min_coverage:
        return False, REASON_GENE_LOW_COVERAGE, metrics

    if M_valid < min(config.min_valid_sectors, B):
        return False, REASON_GENE_TOO_FEW_SECTORS, metrics

    return True, REASON_OK, metrics


def generate_default_qc_table() -> "pd.DataFrame":
    """
    Generate a table of default QC thresholds and their rationales.
    Requires pandas.

    Returns
    -------
    pd.DataFrame
        Table of defaults.
    """
    import pandas as pd

    from biorsp.utils.constants import (
        MIN_BG_EFF_DEFAULT,
        MIN_COVERAGE_DEFAULT,
        MIN_FG_EFF_DEFAULT,
        MIN_TOTAL_MF_DEFAULT,
        MIN_VALID_SECTORS_DEFAULT,
        N_BG_MIN_DEFAULT,
        N_FG_MIN_DEFAULT,
        N_FG_TOT_MIN_DEFAULT,
    )

    data = [
        {
            "Parameter": "min_fg_sector",
            "Default": N_FG_MIN_DEFAULT,
            "Mode": "Binary",
            "Rationale": "Minimum foreground cells per sector to avoid high variance in W1.",
        },
        {
            "Parameter": "min_bg_sector",
            "Default": N_BG_MIN_DEFAULT,
            "Mode": "Binary",
            "Rationale": "Minimum background cells per sector for stable reference distribution.",
        },
        {
            "Parameter": "min_fg_total",
            "Default": N_FG_TOT_MIN_DEFAULT,
            "Mode": "Binary",
            "Rationale": "Minimum total foreground cells to ensure gene-level power.",
        },
        {
            "Parameter": "min_fg_eff",
            "Default": MIN_FG_EFF_DEFAULT,
            "Mode": "Weighted",
            "Rationale": "Kish effective sample size for foreground to handle sparse weights.",
        },
        {
            "Parameter": "min_bg_eff",
            "Default": MIN_BG_EFF_DEFAULT,
            "Mode": "Weighted",
            "Rationale": "Kish effective sample size for background for stable reference.",
        },
        {
            "Parameter": "min_total_mF",
            "Default": MIN_TOTAL_MF_DEFAULT,
            "Mode": "Weighted",
            "Rationale": "Minimum total foreground mass (sum of weights) for gene-level power.",
        },
        {
            "Parameter": "min_coverage",
            "Default": MIN_COVERAGE_DEFAULT,
            "Mode": "Both",
            "Rationale": "Minimum fraction of valid sectors to infer spatial anisotropy.",
        },
        {
            "Parameter": "min_valid_sectors",
            "Default": MIN_VALID_SECTORS_DEFAULT,
            "Mode": "Both",
            "Rationale": "Absolute minimum number of valid sectors for RMS calculation.",
        },
    ]
    return pd.DataFrame(data)


def summarize_qc_behavior(results: List["FeatureResult"]) -> Dict[str, Any]:
    """
    Summarize QC behavior across a set of feature results.

    Parameters
    ----------
    results : List[FeatureResult]
        List of feature results.

    Returns
    -------
    Dict[str, Any]
        Summary statistics.
    """
    if not results:
        return {}

    n_total = len(results)
    n_adequate = sum(1 for r in results if r.adequacy.is_adequate)
    abstention_rate = (n_total - n_adequate) / n_total if n_total > 0 else 0.0

    m_valid_vals = [int(np.sum(r.adequacy.sector_mask)) for r in results]
    coverage_vals = [r.adequacy.adequacy_fraction for r in results]

    return {
        "n_total": n_total,
        "n_adequate": n_adequate,
        "abstention_rate": abstention_rate,
        "m_valid_mean": float(np.mean(m_valid_vals)),
        "m_valid_std": float(np.std(m_valid_vals)),
        "coverage_mean": float(np.mean(coverage_vals)),
        "coverage_std": float(np.std(coverage_vals)),
    }
