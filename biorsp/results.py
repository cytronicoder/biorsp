"""
Results data models for BioRSP.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from .adequacy import AdequacyReport
from .radar import RadarResult
from .robustness import RobustnessResult
from .summaries import ScalarSummaries


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
    radar: Optional[RadarResult] = None
    feature_type: Optional[str] = None
    p_value: Optional[float] = None
    q_value: Optional[float] = None
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

    typing_thresholds: Optional[TypingThresholds]
    pairwise: Optional[Dict[str, list]] = None


__all__ = [
    "FeatureResult",
    "PairwiseResult",
    "TypingThresholds",
    "RunSummary",
]
