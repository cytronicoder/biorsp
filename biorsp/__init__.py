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
    geometric_median,
    polar_coordinates,
    wrapped_circular_distance,
)
from .inference import compute_p_value
from .manifest import BioRSPManifest, create_manifest, save_manifest
from .radar import RadarResult, compute_rsp_radar
from .robustness import RobustnessResult, compute_robustness_score
from .summaries import ScalarSummaries, compute_scalar_summaries

__all__ = [
    "__version__",
    "geometric_median",
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
    "compute_robustness_score",
    "RobustnessResult",
    "create_manifest",
    "save_manifest",
    "BioRSPManifest",
    "BioRSPConfig",
    "constants",
]
