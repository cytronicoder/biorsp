"""BioRSP: Bayesian Inference and Robustness for RSP-style analyses."""

from biorsp._version import __version__
from biorsp.core import (
    FeatureResult,
    PairwiseResult,
    RunSummary,
    ScalarSummaries,
    assign_feature_types,
    compute_p_value,
    compute_rsp_radar,
    compute_scalar_summaries,
)

# Canonical entry point
from biorsp.main import run
from biorsp.preprocess import (
    compute_vantage,
    define_foreground,
    normalize_radii,
    polar_coordinates,
)
from biorsp.utils import BioRSPConfig, setup_logging

__all__ = [
    "__version__",
    "BioRSPConfig",
    "setup_logging",
    "run",
    "compute_rsp_radar",
    "compute_p_value",
    "compute_scalar_summaries",
    "assign_feature_types",
    "FeatureResult",
    "PairwiseResult",
    "RunSummary",
    "ScalarSummaries",
    "compute_vantage",
    "polar_coordinates",
    "normalize_radii",
    "define_foreground",
]
