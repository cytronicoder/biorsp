"""
Results data models for BioRSP.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from biorsp.core.robustness import RobustnessResult
from biorsp.core.summaries import ScalarSummaries
from biorsp.core.typing import AdequacyReport, RadarResult
from biorsp.utils.config import BioRSPConfig


@dataclass
class FeatureResult:
    """
    Container for per-feature outputs.
    """

    feature: str
    threshold_quantile: float
    coverage_quantile: float
    coverage_prevalence: float
    adequacy: AdequacyReport
    summaries: ScalarSummaries
    foreground_info: Optional[dict] = None
    radar: Optional[RadarResult] = None
    feature_type: Optional[str] = None
    p_value: Optional[float] = None
    q_value: Optional[float] = None
    perm_mode: Optional[str] = None
    K_eff: Optional[int] = None
    empty_sector_count: Optional[int] = None
    sector_weight_mode: Optional[str] = None
    sector_weight_k: Optional[float] = None
    robustness: Optional[RobustnessResult] = None


@dataclass
class PairwiseResult:
    """
    Pairwise relationship metrics between two features.
    """

    feature_a: str
    feature_b: str
    correlation: float
    complementarity: float
    peak_distance: float


@dataclass
class TypingThresholds:
    """
    Thresholds used for coverage × anisotropy typing.
    """

    coverage_field: str
    method: str
    coverage_threshold: float
    anisotropy_threshold: float


@dataclass
class RunSummary:
    """
    Global summary metadata for a BioRSP run.
    """

    feature_results: Dict[str, FeatureResult]
    config: BioRSPConfig
    metadata: Dict[str, Any]
    typing_thresholds: Optional[TypingThresholds] = None
    pairwise: Optional[Dict[str, list]] = None

    def to_dataframe(self) -> pd.DataFrame:
        """Convert feature results to a pandas DataFrame."""
        rows = []
        for name, res in self.feature_results.items():
            # Count sector failure reasons
            fail_fg = 0
            fail_bg = 0
            fail_scale = 0
            if res.adequacy.sector_reasons:
                from biorsp.utils.constants import (
                    REASON_SECTOR_BG_TOO_SMALL,
                    REASON_SECTOR_DEGENERATE_SCALE,
                    REASON_SECTOR_FG_TOO_SMALL,
                )

                fail_fg = res.adequacy.sector_reasons.count(REASON_SECTOR_FG_TOO_SMALL)
                fail_bg = res.adequacy.sector_reasons.count(REASON_SECTOR_BG_TOO_SMALL)
                fail_scale = res.adequacy.sector_reasons.count(REASON_SECTOR_DEGENERATE_SCALE)

            row = {
                "feature": name,
                "is_adequate": res.adequacy.is_adequate,
                "abstain_reason": res.adequacy.reason,
                "coverage": res.adequacy.adequacy_fraction,
                "M_valid": int(np.sum(res.adequacy.sector_mask)),
                "total_fg": res.adequacy.n_foreground,
                "sector_fail_low_fg": fail_fg,
                "sector_fail_low_bg": fail_bg,
                "sector_fail_scale": fail_scale,
                "anisotropy": res.summaries.anisotropy,
                "p_value": res.p_value,
                "q_value": res.q_value,
                "K_eff": res.K_eff,
                "empty_sector_count": res.empty_sector_count,
                "feature_type": res.feature_type,
                "peak_distal": res.summaries.peak_distal,
                "peak_proximal": res.summaries.peak_proximal,
                "localization": res.summaries.localization_entropy,
                "r_mean": res.summaries.r_mean,
                "polarity": res.summaries.polarity,
            }
            rows.append(row)
        return pd.DataFrame(rows)


def assign_feature_types(
    feature_results: Dict[str, FeatureResult],
    coverage_field: str = "coverage_prevalence",
    method: str = "median",
    c_hi: Optional[float] = None,
    A_hi: Optional[float] = None,
) -> Tuple[Dict[str, FeatureResult], TypingThresholds]:
    """
    Assign coverage × anisotropy types (I-IV) for adequate features.

    Args:
        feature_results: Mapping of feature name to FeatureResult.
        coverage_field: Coverage field used for typing.
        method: Threshold method ('median' or 'user').
        c_hi: Coverage threshold (required if method='user').
        A_hi: Anisotropy threshold (required if method='user').

    Returns:
        Updated feature_results and TypingThresholds.
    """
    adequate = [
        fr
        for fr in feature_results.values()
        if fr.adequacy.is_adequate and np.isfinite(fr.summaries.anisotropy)
    ]

    coverage_values = np.array([float(getattr(fr, coverage_field)) for fr in adequate], dtype=float)
    anisotropy_values = np.array([fr.summaries.anisotropy for fr in adequate], dtype=float)

    if method == "median":
        c_hi_val = float(np.median(coverage_values)) if coverage_values.size > 0 else np.nan
        A_hi_val = float(np.median(anisotropy_values)) if anisotropy_values.size > 0 else np.nan
    elif method == "user":
        if c_hi is None or A_hi is None:
            raise ValueError("c_hi and A_hi must be provided when method='user'.")
        c_hi_val = float(c_hi)
        A_hi_val = float(A_hi)
    else:
        raise ValueError(f"Unknown typing threshold method: {method}")

    for fr in feature_results.values():
        if not fr.adequacy.is_adequate:
            fr.feature_type = None
            continue

        coverage_value = float(getattr(fr, coverage_field))
        high_cov = coverage_value >= c_hi_val
        high_ani = fr.summaries.anisotropy >= A_hi_val

        if high_cov and high_ani:
            fr.feature_type = "Type I"
        elif not high_cov and high_ani:
            fr.feature_type = "Type II"
        elif high_cov and not high_ani:
            fr.feature_type = "Type III"
        else:
            fr.feature_type = "Type IV"

    thresholds = TypingThresholds(
        coverage_field=coverage_field,
        method=method,
        coverage_threshold=c_hi_val,
        anisotropy_threshold=A_hi_val,
    )

    return feature_results, thresholds


__all__ = [
    "FeatureResult",
    "PairwiseResult",
    "TypingThresholds",
    "RunSummary",
    "assign_feature_types",
]
