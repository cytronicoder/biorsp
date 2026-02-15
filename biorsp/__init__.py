"""BioRSP public API."""

from biorsp._version import __version__
from biorsp.core.compute import compute_rsp
from biorsp.core.features import resolve_feature_index
from biorsp.core.geometry import compute_vantage_point
from biorsp.plotting.rsp import plot_rsp, plot_umap_rsp_pair


def run_case_study(*args, **kwargs):
    """Lazy wrapper to avoid importing heavy pipeline dependencies at import time."""
    from biorsp.pipeline.hierarchy import run_case_study as _run_case_study

    return _run_case_study(*args, **kwargs)


__all__ = [
    "__version__",
    "compute_rsp",
    "plot_rsp",
    "plot_umap_rsp_pair",
    "compute_vantage_point",
    "resolve_feature_index",
    "run_case_study",
]
