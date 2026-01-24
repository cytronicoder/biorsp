"""
Expression simulation: library sizes, signal fields, and count generation.

Implements various spatial patterns and confounded null models.
"""

from typing import Any, Dict, Tuple

import numpy as np
from numpy.random import Generator
from scipy.stats import nbinom

from .density import knn_density
from .geometry import compute_polar


def simulate_library_size(
    n: int, rng: Generator, model: str = "lognormal", params: Dict[str, Any] = None
) -> np.ndarray:
    """
    Generate per-cell library sizes (total UMI counts).

    Parameters
    ----------
    n : int
        Number of cells
    rng : Generator
        Random number generator
    model : str, optional
        Distribution model: 'lognormal', 'uniform', 'gamma'
    params : Dict, optional
        Model parameters

    Returns
    -------
    libsize : np.ndarray
        Library size per cell (n,)
    """
    params = params or {}

    if model == "lognormal":
        mean_lib = params.get("mean", 1000)
        sigma = params.get("sigma", 0.5)
        return rng.lognormal(np.log(mean_lib), sigma, n)

    elif model == "uniform":
        low = params.get("low", 500)
        high = params.get("high", 2000)
        return rng.uniform(low, high, n)

    elif model == "gamma":
        shape = params.get("shape", 2.0)
        scale = params.get("scale", 500)
        return rng.gamma(shape, scale, n)

    else:
        raise ValueError(f"Unknown libsize model: {model}")


def generate_signal_field(
    coords: np.ndarray, pattern: str, params: Dict[str, Any] = None
) -> np.ndarray:
    """
    Generate spatial signal field in [0, 1].

    Represents relative expression probability or intensity.

    NOTE ON BIORSP DETECTABILITY:
    BioRSP's Spatial Bias Score (S) measures *radial* organization - the shift in
    mean radius between foreground (expressing) and background (non-expressing)
    cells. Patterns that induce radial contrast are detectable:

    Detectable (high S):
    - core: center-enriched → FG radius < BG radius
    - rim: periphery-enriched → FG radius > BG radius
    - radial_gradient: smooth radial transition
    - wedge_core: angular domain with center enrichment (RADIAL + angular)
    - wedge_rim: angular domain with periphery enrichment (RADIAL + angular)

    NOT detectable by S alone (low S even if structured):
    - wedge: pure angular domain, no radial shift
    - two_wedges: symmetric angular, cancels radial effect

    For archetype validation, use patterns aligned to what S can detect:
    - regional_program → radial_gradient, core, rim (broad + radial structure)
    - niche_marker → wedge_core, wedge_rim (localized + radial structure)
    - housekeeping → uniform (no structure)
    - sparse_noise → uniform with low coverage (no structure)

    Parameters
    ----------
    coords : np.ndarray
        Cell coordinates (n, 2)
    pattern : str
        Pattern type: uniform, core, rim, wedge, wedge_core, wedge_rim,
        two_wedges, halfplane_gradient, radial_gradient
    params : Dict, optional
        Pattern-specific parameters

    Returns
    -------
    field : np.ndarray
        Signal field values in [0, 1] (n,)
    """
    params = params or {}
    vantage, r, theta = compute_polar(coords, center="median")
    r_norm = (r - r.min()) / (r.max() - r.min() + 1e-9)

    if pattern == "uniform":
        base = params.get("base", 0.5)
        return np.full(len(coords), base)

    elif pattern == "sparse":
        base = params.get("base", 0.05)
        return np.full(len(coords), base)

    elif pattern == "core":
        steepness = params.get("steepness", 5.0)
        return 1.0 / (1.0 + np.exp(steepness * (r_norm - 0.5)))

    elif pattern == "rim":
        steepness = params.get("steepness", 5.0)
        return 1.0 / (1.0 + np.exp(-steepness * (r_norm - 0.5)))

    elif pattern == "radial_gradient":
        direction = params.get("direction", "outward")
        strength = params.get("strength", 1.0)
        field = 1.0 - strength * r_norm if direction == "outward" else strength * r_norm
        return np.clip(field, 0.05, 1.0)

    elif pattern == "wedge":
        angle_center = params.get("angle_center", 0.0)
        width_rad = params.get("width_rad", np.pi / 4)
        diff = np.abs(np.arctan2(np.sin(theta - angle_center), np.cos(theta - angle_center)))
        mask = diff < width_rad
        field = np.full(len(coords), 0.05)
        field[mask] = 0.95
        return field

    elif pattern == "wedge_core":
        angle_center = params.get("angle_center", 0.0)
        width_rad = params.get("width_rad", np.pi / 3)
        steepness = params.get("steepness", 4.0)
        diff = np.abs(np.arctan2(np.sin(theta - angle_center), np.cos(theta - angle_center)))
        angular_weight = np.exp(-((diff / width_rad) ** 2))
        radial_weight = 1.0 / (1.0 + np.exp(steepness * (r_norm - 0.4)))
        field = angular_weight * radial_weight
        return np.clip(field, 0.02, 1.0)

    elif pattern == "wedge_rim":
        angle_center = params.get("angle_center", 0.0)
        width_rad = params.get("width_rad", np.pi / 3)
        steepness = params.get("steepness", 4.0)
        diff = np.abs(np.arctan2(np.sin(theta - angle_center), np.cos(theta - angle_center)))
        angular_weight = np.exp(-((diff / width_rad) ** 2))
        radial_weight = 1.0 / (1.0 + np.exp(-steepness * (r_norm - 0.6)))
        field = angular_weight * radial_weight
        return np.clip(field, 0.02, 1.0)

    elif pattern == "two_wedges":
        p1 = generate_signal_field(coords, "wedge", {**params, "angle_center": 0.0})
        p2 = generate_signal_field(coords, "wedge", {**params, "angle_center": np.pi})
        return np.maximum(p1, p2)

    elif pattern == "halfplane_gradient":
        phi = params.get("phi", 0.0)
        projection = coords[:, 0] * np.cos(phi) + coords[:, 1] * np.sin(phi)
        proj_norm = (projection - projection.min()) / (projection.max() - projection.min() + 1e-9)
        return proj_norm

    else:
        raise ValueError(f"Unknown pattern: {pattern}")


def generate_expression_from_field(
    field: np.ndarray,
    libsize: np.ndarray,
    rng: Generator,
    expr_model: str = "nb",
    params: Dict[str, Any] = None,
) -> np.ndarray:
    """
    Generate expression counts from signal field and library sizes.

    Parameters
    ----------
    field : np.ndarray
        Signal field in [0, 1] (n,)
    libsize : np.ndarray
        Library size per cell (n,)
    rng : Generator
        Random number generator
    expr_model : str, optional
        Expression model: 'nb', 'poisson', 'bernoulli'
    params : Dict, optional
        Model parameters

    Returns
    -------
    counts : np.ndarray
        Expression counts (n,)
    """
    params = params or {}

    if expr_model == "bernoulli":
        p = np.clip(field, 0, 1)
        return rng.binomial(1, p).astype(float)

    elif expr_model == "poisson":
        abundance = params.get("abundance", 1e-3)
        mu = libsize * field * abundance
        return rng.poisson(mu)

    elif expr_model == "nb":
        abundance = params.get("abundance", 1e-3)
        phi = params.get("phi", 10.0)
        mu = libsize * field * abundance
        counts = np.zeros(len(mu), dtype=int)
        nonzero_mask = mu > 1e-9
        if nonzero_mask.any():
            mu_nz = mu[nonzero_mask]
            var_nz = mu_nz + (mu_nz**2) / phi
            p_nb = np.clip(mu_nz / (var_nz + 1e-9), 1e-6, 1 - 1e-6)
            n_nb = np.clip(mu_nz**2 / (var_nz - mu_nz + 1e-9), 1e-3, 1e6)
            counts_nz = np.array(
                [nbinom.rvs(n_nb[i], p_nb[i], random_state=rng) for i in range(len(mu_nz))]
            )
            counts[nonzero_mask] = counts_nz
        return counts

    else:
        raise ValueError(f"Unknown expression model: {expr_model}")


def generate_confounded_null(
    coords: np.ndarray,
    libsize: np.ndarray,
    rng: Generator,
    null_type: str,
    params: Dict[str, Any] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Generate null expression for calibration tests.

    IMPORTANT: These are NULL hypotheses - no true spatial pattern should exist.
    For each null type, there is a corresponding valid permutation scheme.

    Null Types and Valid Permutations
    ---------------------------------
    iid:
        Expression is IID across cells, independent of coordinates and covariates.
        Valid permutation: global shuffle (any cell can exchange with any other).

    depth_confounded:
        Expression depends on library size (depth), and depth varies spatially,
        but conditional on depth, expression is spatially independent.
        Valid permutation: STRATIFIED shuffle within depth bins.
        WARNING: Using global permutation will produce FPR >> alpha!

    mask_stress:
        Same as IID but with very low prevalence (~1-2%), stressing sector masking.
        Valid permutation: global shuffle.

    NOT INCLUDED (creates true spatial signal):
    density_confounded: When expression scales with cell density and density
    varies spatially, this creates TRUE radial signal. Use in archetypes
    benchmark as a stress test, not in calibration.

    Parameters
    ----------
    coords : np.ndarray
        Coordinates (n, 2)
    libsize : np.ndarray
        Library sizes (n,)
    rng : Generator
        Random number generator
    null_type : str
        Null model: 'iid', 'depth_confounded', 'mask_stress'
    params : Dict, optional
        Model parameters

    Returns
    -------
    counts : np.ndarray
        Expression counts (n,)
    meta : Dict
        Metadata about the null including recommended permutation scheme
    """
    params = params or {}

    if null_type == "iid":
        base_prob = params.get("base_prob", 0.1)
        field = np.full(len(coords), base_prob)
        counts = generate_expression_from_field(field, libsize, rng, expr_model="nb", params=params)
        meta = {
            "null_type": "iid",
            "base_prob": base_prob,
            "valid_permutation": "global",
            "exchangeability": "full",
        }
        return counts, meta

    elif null_type == "depth_confounded":
        depth_effect = params.get("depth_effect", 0.5)
        n_bins = params.get("n_depth_bins", 5)

        libsize_norm = (libsize - libsize.min()) / (libsize.max() - libsize.min() + 1e-9)
        depth_bins = np.digitize(libsize_norm, np.linspace(0, 1, n_bins + 1)[1:-1])

        field = 0.1 + depth_effect * libsize_norm
        counts = generate_expression_from_field(field, libsize, rng, expr_model="nb", params=params)
        meta = {
            "null_type": "depth_confounded",
            "depth_effect": depth_effect,
            "n_depth_bins": n_bins,
            "depth_bins": depth_bins,  # Include for stratified permutation
            "valid_permutation": "stratified",
            "exchangeability": "within_depth_strata",
        }
        return counts, meta

    elif null_type == "density_confounded":
        import warnings

        warnings.warn(
            "density_confounded creates TRUE spatial signal and should not be "
            "used for calibration tests. The FPR will be elevated because there "
            "IS a real spatial effect. Use 'iid' or 'depth_confounded' instead.",
            UserWarning,
            stacklevel=2,
        )
        density = knn_density(coords, k=20)
        density_effect = params.get("density_effect", 0.5)
        field = 0.1 + density_effect * (density / density.max())
        counts = generate_expression_from_field(field, libsize, rng, expr_model="nb", params=params)
        meta = {
            "null_type": "density_confounded",
            "density_effect": density_effect,
            "valid_permutation": "NONE_NOT_A_NULL",  # Mark as not valid for calibration
            "exchangeability": "NONE",
            "warning": "Creates true spatial signal - not a null hypothesis",
        }
        return counts, meta

    elif null_type == "mask_stress":
        base_prob = params.get("base_prob", 0.01)  # Very low prevalence
        field = np.full(len(coords), base_prob)
        counts = generate_expression_from_field(field, libsize, rng, expr_model="nb", params=params)
        meta = {
            "null_type": "mask_stress",
            "base_prob": base_prob,
            "valid_permutation": "global",
            "exchangeability": "full",
        }
        return counts, meta

    else:
        raise ValueError(f"Unknown null type: {null_type}")


def get_permutation_scheme(null_type: str) -> str:
    """
    Get the valid permutation scheme for a null type.

    This maps null hypotheses to their exchangeable permutation strategies.

    Parameters
    ----------
    null_type : str
        The null hypothesis type

    Returns
    -------
    scheme : str
        'global' for IID-like nulls, 'stratified' for confounded nulls
    """
    scheme_map = {
        "iid": "global",
        "mask_stress": "global",
        "depth_confounded": "stratified",
        "density_confounded": "INVALID",  # Not a null
    }
    return scheme_map.get(null_type, "global")


def create_stratification_bins(
    libsize: np.ndarray,
    n_bins: int = 5,
) -> np.ndarray:
    """
    Create stratification bins for depth-confounded permutation tests.

    Parameters
    ----------
    libsize : np.ndarray
        Library size per cell
    n_bins : int
        Number of quantile bins

    Returns
    -------
    bins : np.ndarray
        Bin assignment for each cell (0 to n_bins-1)
    """
    libsize_norm = (libsize - libsize.min()) / (libsize.max() - libsize.min() + 1e-9)
    bins = np.digitize(libsize_norm, np.linspace(0, 1, n_bins + 1)[1:-1])
    return bins


def generate_expression_targeted(
    coords: np.ndarray,
    libsize: np.ndarray,
    rng: Generator,
    pattern: str,
    target_coverage: float,
    pattern_params: Dict[str, Any] = None,
    expr_params: Dict[str, Any] = None,
    detection_threshold: float = 1.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Generate expression with targeted coverage (prevalence).

    Adjusts the expression level to achieve approximately the desired coverage
    (fraction of cells with counts >= detection_threshold).

    Parameters
    ----------
    coords : np.ndarray
        Cell coordinates (n, 2)
    libsize : np.ndarray
        Library size per cell (n,)
    rng : Generator
        Random number generator
    pattern : str
        Spatial pattern type (uniform, wedge, core, rim, etc.)
    target_coverage : float
        Target fraction of cells with expression >= detection_threshold.
        For structured patterns, this is the coverage within the "expressing" region.
    pattern_params : Dict, optional
        Pattern-specific parameters
    expr_params : Dict, optional
        Expression model parameters
    detection_threshold : float, optional
        Minimum count to consider a cell as expressing (default: 1)

    Returns
    -------
    counts : np.ndarray
        Expression counts (n,)
    meta : Dict
        Metadata including achieved coverage and pattern info
    """
    pattern_params = pattern_params or {}
    expr_params = expr_params or {}

    field = generate_signal_field(coords, pattern, pattern_params)

    expr_params.get("phi", 10.0)
    base_abundance = expr_params.get("abundance", 1e-3)

    lo, hi = 1e-6, 1e-1
    best_abundance = base_abundance
    best_diff = 1.0

    for _ in range(15):  # Binary search iterations
        mid = np.sqrt(lo * hi)  # Geometric mean
        test_params = {**expr_params, "abundance": mid}
        test_counts = generate_expression_from_field(
            field, libsize, rng, expr_model="nb", params=test_params
        )
        achieved_cov = np.mean(test_counts >= detection_threshold)

        diff = abs(achieved_cov - target_coverage)
        if diff < best_diff:
            best_diff = diff
            best_abundance = mid

        if achieved_cov < target_coverage:
            lo = mid
        else:
            hi = mid
        if diff < 0.02:
            break
    final_params = {**expr_params, "abundance": best_abundance}
    counts = generate_expression_from_field(
        field, libsize, rng, expr_model="nb", params=final_params
    )
    achieved_coverage = np.mean(counts >= detection_threshold)

    meta = {
        "pattern": pattern,
        "target_coverage": target_coverage,
        "achieved_coverage": achieved_coverage,
        "abundance_used": best_abundance,
        **pattern_params,
    }

    return counts, meta


def generate_factorial_gene(
    coords: np.ndarray,
    libsize: np.ndarray,
    rng: Generator,
    coverage_regime: str,
    organization_regime: str,
    pattern_variant: str = "wedge_core",
    coverage_params: Dict[str, Any] = None,
    pattern_params: Dict[str, Any] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Generate gene expression with factorial control over coverage and organization.

    This is the main entry point for the 2x2 archetype design:
    - Coverage: high vs. low (controlled prevalence)
    - Organization: structured (spatial pattern) vs. unstructured (iid)

    IMPORTANT: For structured patterns, use ONLY patterns that BioRSP's S score
    can detect (patterns with radial contrast):
    - core, rim: radial enrichment
    - wedge_core, wedge_rim: angular + radial structure (DETECTABLE)
    - radial_gradient: smooth radial transition

    Do NOT use pure angular patterns (wedge, two_wedges) as they have no radial
    contrast and will not produce high S scores.

    Parameters
    ----------
    coords : np.ndarray
        Cell coordinates (n, 2)
    libsize : np.ndarray
        Library size per cell (n,)
    rng : Generator
        Random number generator
    coverage_regime : str
        'high' (70-90% prevalence) or 'low' (5-20% prevalence)
    organization_regime : str
        'structured' (spatial pattern) or 'iid' (random scatter)
    pattern_variant : str, optional
        For structured: 'core', 'rim', 'wedge_core', 'wedge_rim', 'radial_gradient'
        Default: 'wedge_core' (angular + radial, detectable by S)
    coverage_params : Dict, optional
        Override default coverage targets
    pattern_params : Dict, optional
        Pattern-specific parameters

    Returns
    -------
    counts : np.ndarray
        Expression counts (n,)
    meta : Dict
        Ground truth metadata including archetype and generative parameters
    """
    coverage_params = coverage_params or {}
    pattern_params = pattern_params or {}
    coverage_defaults = {
        "high": {"target": 0.75, "range": (0.60, 0.90)},
        "low": {"target": 0.12, "range": (0.05, 0.25)},
    }
    if coverage_regime not in coverage_defaults:
        raise ValueError(f"Unknown coverage_regime: {coverage_regime}")
    cov_settings = coverage_defaults[coverage_regime]
    target_cov = coverage_params.get("target", cov_settings["target"])
    cov_jitter = rng.uniform(-0.05, 0.05)
    target_cov = np.clip(target_cov + cov_jitter, 0.02, 0.98)
    DETECTABLE_PATTERNS = {"core", "rim", "wedge_core", "wedge_rim", "radial_gradient"}
    if organization_regime == "structured" and pattern_variant not in DETECTABLE_PATTERNS:
        import warnings

        warnings.warn(
            f"Pattern '{pattern_variant}' may not be detectable by BioRSP's S score. "
            f"Consider using one of: {DETECTABLE_PATTERNS}",
            stacklevel=2,
        )
    if organization_regime == "iid":
        pattern = "uniform"
        eff_params = {"base": target_cov}
    elif organization_regime == "structured":
        pattern = pattern_variant
        eff_params = {**pattern_params}
    else:
        raise ValueError(f"Unknown organization_regime: {organization_regime}")

    counts, expr_meta = generate_expression_targeted(
        coords=coords,
        libsize=libsize,
        rng=rng,
        pattern=pattern,
        target_coverage=target_cov,
        pattern_params=eff_params,
    )

    archetype_map = {
        ("high", "iid"): "housekeeping",
        ("high", "structured"): "regional_program",
        ("low", "iid"): "sparse_noise",
        ("low", "structured"): "niche_marker",
    }
    archetype = archetype_map[(coverage_regime, organization_regime)]

    n_expr_cells = np.sum(counts >= 1)
    observed_coverage = n_expr_cells / len(counts)

    meta = {
        "Archetype": archetype,
        "coverage_regime": coverage_regime,
        "organization_regime": organization_regime,
        "pattern_variant": pattern_variant if organization_regime == "structured" else "none",
        "target_coverage": target_cov,
        "observed_coverage": observed_coverage,
        "n_expr_cells": int(n_expr_cells),
        **expr_meta,
    }

    return counts, meta


def generate_factorial_gene_with_beta(
    coords: np.ndarray,
    libsize: np.ndarray,
    rng: Generator,
    base_prevalence: float,
    spatial_beta: float,
    pattern_mechanism: str = "radial_gradient",
    pattern_params: Dict[str, Any] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Generate expression with explicit control over prevalence and spatial effect strength.

    This implements the principled logistic model:
        logit(p_i) = logit(base_prevalence) + beta * (f_i - mean(f))

    Where f_i is a zero-centered spatial modulation field.

    Parameters
    ----------
    coords : np.ndarray
        Cell coordinates (n, 2)
    libsize : np.ndarray
        Library size per cell (n,)
    rng : Generator
        Random number generator
    base_prevalence : float
        Target overall prevalence (controls C)
    spatial_beta : float
        Spatial effect strength (controls S):
        - beta = 0: no spatial structure (iid)
        - beta > 0: spatial structure with strength proportional to beta
    pattern_mechanism : str
        Spatial pattern: 'core', 'rim', 'wedge_core', 'wedge_rim', 'radial_gradient'
    pattern_params : Dict, optional
        Pattern-specific parameters

    Returns
    -------
    counts : np.ndarray
        Expression counts (n,)
    meta : Dict
        Complete ground truth metadata
    """
    pattern_params = pattern_params or {}

    field_raw = generate_signal_field(coords, pattern_mechanism, pattern_params)

    field_centered = field_raw - np.mean(field_raw)

    base_p_clip = np.clip(base_prevalence, 1e-6, 1 - 1e-6)
    logit_base = np.log(base_p_clip / (1 - base_p_clip))

    logit_p = logit_base + spatial_beta * field_centered
    p_spatial = 1.0 / (1.0 + np.exp(-logit_p))

    counts = generate_expression_from_field(
        p_spatial, libsize, rng, expr_model="nb", params={"phi": 10.0, "abundance": 1e-3}
    )

    n_expr_cells = np.sum(counts >= 1)
    observed_coverage = n_expr_cells / len(counts)

    high_coverage = base_prevalence >= 0.30
    high_spatial = spatial_beta >= 0.5  # Threshold for "structured"

    archetype_map = {
        (True, False): "housekeeping",
        (True, True): "regional_program",
        (False, False): "sparse_noise",
        (False, True): "niche_marker",
    }
    archetype = archetype_map[(high_coverage, high_spatial)]

    return counts, {
        "Archetype": archetype,
        "base_prevalence": base_prevalence,
        "spatial_beta": spatial_beta,
        "pattern_mechanism": pattern_mechanism,
        "target_coverage": base_prevalence,
        "observed_coverage": observed_coverage,
        "n_expr_cells": int(n_expr_cells),
        "field_mean": float(np.mean(field_raw)),
        "field_std": float(np.std(field_raw)),
        "p_spatial_mean": float(np.mean(p_spatial)),
        "p_spatial_std": float(np.std(p_spatial)),
    }
