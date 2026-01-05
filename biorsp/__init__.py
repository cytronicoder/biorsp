"""BioRSP: Bayesian Inference and Robustness for RSP-style analyses."""

from biorsp import simulations
from biorsp._version import __version__
from biorsp.core import (
    FeatureResult,
    PairwiseResult,
    RunSummary,
    ScalarSummaries,
    assess_adequacy,
    assign_feature_types,
    compute_p_value,
    compute_rsp_radar,
    compute_scalar_summaries,
)

# Canonical entry point
from biorsp.main import run
from biorsp.plotting import (
    plot_embedding,
    plot_localization_scatter,
    plot_phenotype_map,
    plot_radar,
    plot_radar_absolute,
    plot_radar_split,
    plot_summary,
)
from biorsp.preprocess import (
    compute_vantage,
    define_foreground,
    geometric_median,
    normalize_radii,
    polar_coordinates,
)
from biorsp.utils import (
    BioRSPConfig,
    add_common_args,
    config_from_args,
    ensure_outdir,
    get_features_to_run,
    save_run_manifest,
    setup_logging,
)

__all__ = [
    "__version__",
    "BioRSPConfig",
    "setup_logging",
    "run",
    "simulations",
    "compute_rsp_radar",
    "compute_p_value",
    "compute_scalar_summaries",
    "assign_feature_types",
    "assess_adequacy",
    "FeatureResult",
    "PairwiseResult",
    "RunSummary",
    "ScalarSummaries",
    "compute_vantage",
    "geometric_median",
    "polar_coordinates",
    "normalize_radii",
    "define_foreground",
    "plot_embedding",
    "plot_localization_scatter",
    "plot_phenotype_map",
    "plot_radar",
    "plot_radar_absolute",
    "plot_radar_split",
    "plot_summary",
    "add_common_args",
    "config_from_args",
    "ensure_outdir",
    "get_features_to_run",
    "save_run_manifest",
]
