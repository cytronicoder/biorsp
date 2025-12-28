"""biorsp package.

Bayesian inference and robustness tools for RSP-style analyses.
"""

from . import constants
from ._version import __version__
from .config import BioRSPConfig
from .foreground import binary_foreground, soft_foreground_weights

# Public API
from .geometry import (
    angle_grid,
    compute_vantage,
    geometric_median,
    polar_coordinates,
    wrapped_circular_distance,
)
from .inference import bh_fdr, compute_p_value
from .manifest import BioRSPManifest, create_manifest, save_manifest
from .pairwise import compute_pairwise_relationships
from .radar import RadarResult, compute_rsp_radar
from .results import FeatureResult, PairwiseResult, RunSummary, TypingThresholds
from .robustness import RobustnessResult, compute_robustness_score
from .summaries import ScalarSummaries, compute_scalar_summaries
from .typing import assign_feature_types

__all__ = [
    "__version__",
    "geometric_median",
    "compute_vantage",
    "polar_coordinates",
    "wrapped_circular_distance",
    "angle_grid",
    "binary_foreground",
    "soft_foreground_weights",
    "compute_rsp_radar",
    "RadarResult",
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
    "BioRSPConfig",
    "constants",
]
