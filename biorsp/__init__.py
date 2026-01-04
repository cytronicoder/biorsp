"""biorsp package.

Bayesian inference and robustness tools for RSP-style analyses.
"""

from . import constants
from ._version import __version__
from .config import BioRSPConfig
from .core import (
    assess_adequacy,
    compute_anisotropy,
    compute_rsp_radar,
)

# Public API
from .geometry import (
    angle_grid,
    compute_vantage,
    geometric_median,
    get_sector_indices,
    polar_coordinates,
    wrapped_circular_distance,
)
from .manifest import BioRSPManifest, create_manifest, save_manifest
from .pairwise import compute_pairwise_relationships
from .preprocessing import define_foreground, define_foreground_weights, normalize_radii
from .results import (
    FeatureResult,
    PairwiseResult,
    RunSummary,
    TypingThresholds,
    assign_feature_types,
)
from .robustness import RobustnessResult, compute_robustness_score
from .stats import bh_fdr, compute_p_value
from .summaries import ScalarSummaries, compute_scalar_summaries
from .typing import AdequacyReport, InferenceResult, RadarResult

__all__ = [
    "__version__",
    "BioRSPConfig",
    "geometric_median",
    "compute_vantage",
    "polar_coordinates",
    "wrapped_circular_distance",
    "angle_grid",
    "get_sector_indices",
    "define_foreground",
    "define_foreground_weights",
    "normalize_radii",
    "compute_rsp_radar",
    "compute_anisotropy",
    "assess_adequacy",
    "RadarResult",
    "AdequacyReport",
    "InferenceResult",
    "compute_scalar_summaries",
    "ScalarSummaries",
    "compute_p_value",
    "bh_fdr",
    "compute_robustness_score",
    "RobustnessResult",
    "compute_pairwise_relationships",
    "assign_feature_types",
    "FeatureResult",
    "PairwiseResult",
    "RunSummary",
    "TypingThresholds",
    "create_manifest",
    "save_manifest",
    "BioRSPManifest",
    "constants",
]
