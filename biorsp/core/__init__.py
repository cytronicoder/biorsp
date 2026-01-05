"""Core computation modules for BioRSP."""

from biorsp.core.adequacy import assess_adequacy
from biorsp.core.engine import compute_anisotropy, compute_rsp_radar
from biorsp.core.inference import compute_diagnostic_null, compute_p_value
from biorsp.core.pairwise import compute_pairwise_relationships
from biorsp.core.results import (
    FeatureResult,
    PairwiseResult,
    RunSummary,
    TypingThresholds,
    assign_feature_types,
)
from biorsp.core.robustness import RobustnessResult, compute_robustness_score
from biorsp.core.summaries import ScalarSummaries, compute_scalar_summaries
from biorsp.core.typing import AdequacyReport, InferenceResult, RadarResult

__all__ = [
    "assess_adequacy",
    "compute_anisotropy",
    "compute_rsp_radar",
    "compute_p_value",
    "compute_diagnostic_null",
    "compute_pairwise_relationships",
    "FeatureResult",
    "PairwiseResult",
    "RunSummary",
    "TypingThresholds",
    "assign_feature_types",
    "RobustnessResult",
    "compute_robustness_score",
    "ScalarSummaries",
    "compute_scalar_summaries",
    "AdequacyReport",
    "InferenceResult",
    "RadarResult",
]
