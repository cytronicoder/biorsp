"""
Biological Radar Scanning Plot (BioRSP)

A Python package for multiscale directional analysis of 2D embeddings
using radar scanning methodology.
"""

__version__ = "0.1.0"

# Core radar scanning classes and functions
from .radar_scan import (
    ScanParams,
    FeatureResult,
    RadarScanner,
)

# Preprocessing functions
from .preprocessing import (
    validate_inputs,
    to_polar,
    bin_angles,
    geometric_median,
    choose_center,
    standardize_feature,
    binarize_feature,
    residualize_feature,
    assign_radial_bands,
    density_weights,
    density_ratio,
)

# Statistical functions
from .stats import (
    make_kernel,
    expected_from_uniform,
    variance_binomial_like,
    variance_plugin,
    z_scores,
    aggregate_bands,
    enrichment_ratio,
    mean_resultant_length,
    circular_fourier,
    sector_sums_convolved,
    compute_Z_grid,
    compute_fg_bg_difference,
)

# Null model functions
from .null_models import (
    rotation_shifts,
    max_stat_under_rotations,
    max_stat_within_batch_rotations,
    label_permutation_within_bands,
    empirical_pvalue,
    bh_qvalues,
    max_from_Zheat,
    rotation_null_pvalue,
    within_batch_rotation_pvalue,
    permutation_null_pvalue,
    permutation_pvalue_fg_bg,
)

# Utility functions
from .utils import (
    config_hash,
    ensure_float64,
    ensure_c_contiguous,
    safe_divide,
    discretize_width_to_bins,
    argmax2d,
    effective_sample_size,
    check_random_state,
    bootstrap_indices,
    circ_fft_convolve,
    circ_prefix_convolve,
    circ_shift,
    circ_shift_batch,
)

# Plotting functions
from .plotting import (
    plot_rsp_heatmap,
    plot_rsp_grid,
    plot_rsp_summary,
    save_top_results,
)

__all__ = [
    # Version
    "__version__",
    # Core classes
    "ScanParams",
    "FeatureResult",
    "RadarScanner",
    # Preprocessing
    "validate_inputs",
    "to_polar",
    "bin_angles",
    "geometric_median",
    "choose_center",
    "standardize_feature",
    "binarize_feature",
    "residualize_feature",
    "assign_radial_bands",
    "density_weights",
    "density_ratio",
    # Statistics
    "make_kernel",
    "expected_from_uniform",
    "variance_binomial_like",
    "variance_plugin",
    "z_scores",
    "aggregate_bands",
    "enrichment_ratio",
    "mean_resultant_length",
    "circular_fourier",
    "sector_sums_convolved",
    "compute_Z_grid",
    "compute_fg_bg_difference",
    # Null models
    "rotation_shifts",
    "max_stat_under_rotations",
    "max_stat_within_batch_rotations",
    "label_permutation_within_bands",
    "empirical_pvalue",
    "bh_qvalues",
    "max_from_Zheat",
    "rotation_null_pvalue",
    "within_batch_rotation_pvalue",
    "permutation_null_pvalue",
    "permutation_pvalue_fg_bg",
    # Utilities
    "config_hash",
    "ensure_float64",
    "ensure_c_contiguous",
    "safe_divide",
    "discretize_width_to_bins",
    "argmax2d",
    "effective_sample_size",
    "check_random_state",
    "bootstrap_indices",
    "circ_fft_convolve",
    "circ_prefix_convolve",
    "circ_shift",
    "circ_shift_batch",
    # Plotting
    "plot_rsp_heatmap",
    "plot_rsp_grid",
    "plot_rsp_summary",
    "save_top_results",
]
