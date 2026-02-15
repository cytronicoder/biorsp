"""Core compute subpackage."""

from biorsp.core.compute import (
    compute_rsp,
    compute_rsp_profile,
    compute_rsp_profile_from_boolean,
)
from biorsp.core.features import get_feature_vector, resolve_feature_index
from biorsp.core.geometry import compute_theta, compute_vantage_point
from biorsp.core.types import NullConfig, RSPConfig, RSPResult, ScopeResult

__all__ = [
    "RSPConfig",
    "NullConfig",
    "RSPResult",
    "ScopeResult",
    "compute_rsp",
    "compute_rsp_profile",
    "compute_rsp_profile_from_boolean",
    "compute_theta",
    "compute_vantage_point",
    "resolve_feature_index",
    "get_feature_vector",
]
