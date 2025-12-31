"""
Coverage × anisotropy typing for BioRSP features.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from .results import FeatureResult, TypingThresholds


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


__all__ = ["assign_feature_types"]
