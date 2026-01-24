"""Results data models for BioRSP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from biorsp.core.robustness import RobustnessResult
from biorsp.core.summaries import ScalarSummaries
from biorsp.core.typing import AdequacyReport, RadarResult
from biorsp.utils.config import BioRSPConfig


@dataclass
class FeatureResult:
    """Container for per-feature outputs."""

    feature: str
    threshold_quantile: float
    coverage_quantile: float
    coverage_prevalence: float
    adequacy: AdequacyReport
    summaries: ScalarSummaries
    foreground_info: dict | None = None
    radar: RadarResult | None = None
    feature_type: str | None = None
    p_value: float | None = None
    q_value: float | None = None
    perm_mode: str | None = None
    K_eff: int | None = None
    empty_sector_count: int | None = None
    sector_weight_mode: str | None = None
    sector_weight_k: float | None = None
    robustness: RobustnessResult | None = None


@dataclass
class PairwiseResult:
    """Pairwise relationship metrics between two features."""

    feature_a: str
    feature_b: str
    correlation: float
    complementarity: float
    peak_distance: float


@dataclass
class TypingThresholds:
    """Thresholds used for coverage × anisotropy typing."""

    coverage_field: str
    method: str
    coverage_threshold: float
    anisotropy_threshold: float


@dataclass
class RunSummary:
    """Global summary metadata for a BioRSP run."""

    feature_results: dict[str, FeatureResult]
    config: BioRSPConfig
    metadata: dict[str, Any]
    typing_thresholds: TypingThresholds | None = None
    pairwise: dict[str, list] | None = None

    def to_dataframe(self) -> pd.DataFrame:
        """Convert feature results to a pandas DataFrame.

        Returns:
            DataFrame containing per-feature summaries and metadata.
        """
        rows = []
        for name, res in self.feature_results.items():
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
                "coverage_bg": res.summaries.coverage_bg,
                "coverage_fg": res.summaries.coverage_fg,
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
    feature_results: dict[str, FeatureResult],
    coverage_field: str = "coverage_prevalence",
    method: str = "median",
    c_hi: float | None = None,
    A_hi: float | None = None,
) -> tuple[dict[str, FeatureResult], TypingThresholds]:
    """Assign coverage × anisotropy types (I–IV) for adequate features.

    Args:
        feature_results: Mapping of feature name to FeatureResult.
        coverage_field: Coverage field used for typing.
        method: Threshold method (`median` or `user`).
        c_hi: Coverage threshold (required if `method='user'`).
        A_hi: Anisotropy threshold (required if `method='user'`).

    Returns:
        Updated feature results and typing thresholds.

    Raises:
        ValueError: If `method='user'` and thresholds are missing, or if
            an unsupported method is provided.
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
