"""BioRSP: lightweight utilities for reproducible spatial pattern statistics on single-cell data."""

from biorsp._version import __version__
from biorsp.geometry import compute_angles, compute_vantage
from biorsp.moran import extract_weights, morans_i
from biorsp.permutation import (
    build_donor_index,
    perm_null_emax,
    permute_foreground_within_donor,
    plot_null_distribution,
)
from biorsp.rsp import (
    compute_rsp_profile,
    compute_rsp_profile_from_boolean,
    plot_rsp_polar,
)
from biorsp.utils import ensure_dir, get_gene_vector, select_gene

__all__ = [
    "__version__",
    "compute_angles",
    "compute_vantage",
    "compute_rsp_profile",
    "compute_rsp_profile_from_boolean",
    "plot_rsp_polar",
    "extract_weights",
    "morans_i",
    "build_donor_index",
    "permute_foreground_within_donor",
    "perm_null_emax",
    "plot_null_distribution",
    "ensure_dir",
    "get_gene_vector",
    "select_gene",
]
