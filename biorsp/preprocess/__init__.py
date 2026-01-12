"""Preprocessing modules for BioRSP."""

from biorsp.core.geometry import (
    angle_grid,
    compute_vantage,
    geometric_median,
    get_sector_indices,
    polar_coordinates,
    wrapped_circular_distance,
)
from biorsp.preprocess.context import (
    BioRSPContext,
    discover_embedding_key,
    prepare_context,
    score_gene_with_context,
)
from biorsp.preprocess.foreground import define_foreground, define_foreground_weights
from biorsp.preprocess.normalization import normalize_radii
from biorsp.preprocess.stratification import get_strata_indices

__all__ = [
    "BioRSPContext",
    "discover_embedding_key",
    "prepare_context",
    "score_gene_with_context",
    "define_foreground",
    "define_foreground_weights",
    "angle_grid",
    "compute_vantage",
    "geometric_median",
    "get_sector_indices",
    "polar_coordinates",
    "wrapped_circular_distance",
    "normalize_radii",
    "get_strata_indices",
]
