"""BioRSP: Spatial transcriptomics analysis using radar-style gene profiles.

This package quantifies gene expression patterns through two complementary metrics:
- Coverage (C): Fraction of cells with biologically meaningful expression
- Spatial Organization (S): Extent of directional clustering in the radar profile

These metrics classify genes into interpretable archetypes:
- I: Ubiquitous (high C, low S): uniformly expressed across tissue
- II: Gradient (high C, high S): broad spatial domains
- III: Patchy (low C, high S): spatially restricted expression
- IV: Basal (low C, low S): random sparse expression

P-values are computed via permutation-based null distributions.
"""

from biorsp._version import __version__
from biorsp.api import classify_genes, score_gene_pairs, score_genes
from biorsp.utils.config import BioRSPConfig

__all__ = ["score_genes", "classify_genes", "score_gene_pairs", "BioRSPConfig", "__version__"]

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
    "score_genes",
    "score_gene_pairs",
    "classify_genes",
    "simulations",
    "compute_rsp_radar",
    "compute_p_value",
    "compute_scalar_summaries",
    "assign_feature_types",
    "assess_adequacy",
    "compute_pairwise_relationships",
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
