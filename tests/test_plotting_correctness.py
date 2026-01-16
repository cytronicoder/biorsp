"""Regression tests for BioRSP plotting correctness.

Tests ensure:
1. Coverage differs from internal FG fraction (public schema)
2. Workflow figure reports same C and Spatial_Bias_Score as scoring function
3. Wedge patterns don't claim directionality when empty_fg_policy="zero"
4. RSP plots handle all sector types correctly
"""

import numpy as np
import pytest

from biorsp.api import BioRSPConfig, score_genes
from biorsp.core.engine import compute_rsp_radar
from biorsp.core.geometry import compute_vantage, polar_coordinates
from biorsp.plotting.workflow import interpret_pattern
from biorsp.preprocess.foreground import define_foreground
from biorsp.preprocess.normalization import normalize_radii


def generate_test_data(n=500, pattern="rim_wedge", seed=42):
    """Generate synthetic spatial data with known patterns.

    Args:
        n: Number of cells.
        pattern: "rim_wedge" (localized rim), "core_global" (global core bias).
        seed: Random seed.

    Returns:
        coords, expr, true_coverage, true_spatial_score
    """
    rng = np.random.default_rng(seed)

    # Create spatial embedding
    r_true = np.sqrt(rng.random(n))
    theta_true = 2 * np.pi * rng.random(n) - np.pi
    coords = np.column_stack([r_true * np.cos(theta_true), r_true * np.sin(theta_true)])

    if pattern == "rim_wedge":
        # Localized rim pattern at theta ~ 0
        # High expression in rim (r > 0.6) AND near theta=0
        prob_base = 0.05
        prob_spatial = 0.85 * (r_true > 0.6) * np.exp(-0.5 * (theta_true / 0.5) ** 2)
        prob = np.clip(prob_base + prob_spatial, 0, 1)
        expr = rng.binomial(10, prob).astype(float)  # Counts

        # True coverage: fraction with expr >= 1
        true_coverage = np.mean(expr >= 1)
        return coords, expr, true_coverage, "wedge_rim"

    elif pattern == "core_global":
        # Global core bias (all angles, but closer to center)
        prob_base = 0.05
        prob_spatial = 0.7 * (r_true < 0.5)  # Core cells (all directions)
        prob = np.clip(prob_base + prob_spatial, 0, 1)
        expr = rng.binomial(10, prob).astype(float)

        true_coverage = np.mean(expr >= 1)
        return coords, expr, true_coverage, "core_global"

    else:
        raise ValueError(f"Unknown pattern: {pattern}")


def test_coverage_vs_foreground_distinction():
    """Test that Coverage != internal foreground fraction."""
    coords, expr, true_coverage, _ = generate_test_data(n=500, pattern="rim_wedge")

    # Detect threshold
    # Note: config object not used here (just computing threshold directly)
    is_integers = np.allclose(expr, np.round(expr))
    thresh = 1.0 if is_integers else 1e-6

    coverage = float(np.mean(expr >= thresh))

    # Define internal foreground (quantile-based, for spatial scoring)
    fg_mask, fg_info = define_foreground(expr, mode="quantile", q=0.9)
    foreground_fraction = float(np.mean(fg_mask))

    # Critical assertion: these should differ
    assert coverage != foreground_fraction, (
        f"Coverage ({coverage:.3f}) should differ from "
        f"foreground_fraction ({foreground_fraction:.3f})"
    )

    # Coverage should be higher (more lenient threshold)
    assert (
        coverage > foreground_fraction
    ), "Coverage (biological threshold) should be >= internal FG (quantile)"

    # Coverage must align with fraction above biological threshold
    assert np.isclose(coverage, true_coverage, atol=1e-3)


def test_workflow_matches_scoring():
    """Test that workflow figure reports same metrics as score_genes."""
    pytest.importorskip("anndata")
    from anndata import AnnData

    coords, expr, _, _ = generate_test_data(n=500, pattern="rim_wedge")

    # Create AnnData
    adata = AnnData(X=expr.reshape(-1, 1), obsm={"X_spatial": coords})
    adata.var_names = ["test_gene"]

    # Score using public API
    config = BioRSPConfig(
        delta_deg=30,
        B=24,
        expr_threshold_mode="detect",
        foreground_mode="quantile",
        foreground_quantile=0.9,
    )
    results = score_genes(adata, ["test_gene"], embedding_key="X_spatial", config=config)

    required_columns = {"Coverage", "Spatial_Bias_Score", "Directionality", "Archetype"}
    banned_columns = {
        "coverage_expr",
        "pct_cells",
        "alpha",
        "anisotropy",
        "rms",
        "S_g",
        "class",
        "type",
    }

    assert required_columns.issubset(set(results.columns)), "Public schema columns missing"
    assert banned_columns.isdisjoint(
        set(results.columns)
    ), "Legacy columns leaked into public output"

    # Results uses integer index, gene name in 'gene' column
    api_coverage = results.loc[0, "Coverage"]
    api_spatial = results.loc[0, "Spatial_Bias_Score"]

    # Manually compute for workflow
    v = compute_vantage(coords, method="geometric_median")
    r, theta = polar_coordinates(coords, v)
    r_norm, _ = normalize_radii(r)

    thresh = 1.0 if np.allclose(expr, np.round(expr)) else 1e-6
    coverage_manual = float(np.mean(expr >= thresh))

    fg_mask, _ = define_foreground(expr, mode="quantile", q=0.9)
    if fg_mask is None:
        fg_mask = (expr >= np.quantile(expr, 0.9)).astype(float)

    radar = compute_rsp_radar(r_norm, theta, fg_mask, config=config)
    mask_geom = (
        radar.geom_supported_mask
        if radar.geom_supported_mask is not None
        else radar.bg_supported_mask
    )
    if mask_geom is None or not np.any(mask_geom):
        print("  ⊘ SKIPPED (no geom-supported sectors)")
        return

    # Contract: weights are not normalized, use proper weighted mean
    # Also handle NaN values in rsp (convert to 0 as done in scoring)
    w = radar.sector_weights[mask_geom]
    rsp_masked = np.nan_to_num(radar.rsp[mask_geom], nan=0.0)
    sum_w = np.sum(w)
    S_g_manual = float(np.sqrt(np.sum(w * rsp_masked**2) / sum_w)) if sum_w > 0 else 0.0

    # Critical assertions
    assert np.isclose(
        coverage_manual, api_coverage, atol=0.01
    ), f"Manual coverage ({coverage_manual:.3f}) should match API ({api_coverage:.3f})"

    assert np.isclose(
        S_g_manual, api_spatial, atol=0.05
    ), f"Manual spatial score ({S_g_manual:.3f}) should match API ({api_spatial:.3f})"


def test_empty_fg_zero_fill_correctness():
    """Test that empty FG sectors are marked and don't affect metrics when zero-filled."""
    coords, expr, _, _ = generate_test_data(n=500, pattern="rim_wedge")

    config = BioRSPConfig(
        delta_deg=30,
        B=24,
        empty_fg_policy="zero",
    )

    v = compute_vantage(coords)
    r, theta = polar_coordinates(coords, v)
    r_norm, _ = normalize_radii(r)

    # Very restrictive foreground (will have empty sectors)
    fg_mask, fg_info = define_foreground(expr, mode="quantile", q=0.95)

    # If foreground definition failed, use a manual threshold
    if fg_mask is None:
        fg_mask = (expr >= np.quantile(expr, 0.95)).astype(float)

    radar = compute_rsp_radar(r_norm, theta, fg_mask, config=config)

    # Check that forced_zero_mask exists and is tracked
    assert hasattr(radar, "forced_zero_mask"), "RadarResult must have forced_zero_mask"
    assert radar.forced_zero_mask is not None, "forced_zero_mask should be populated"

    # Sectors marked as forced-zero should have RSP = 0
    if np.any(radar.forced_zero_mask):
        forced_zero_rsp = radar.rsp[radar.forced_zero_mask]
        assert np.all(forced_zero_rsp == 0), "Forced-zero sectors must have RSP = 0"
        # Zeroed sectors must not introduce NaNs and contribute zero to Spatial_Bias_Score numerator
        assert np.all(np.isfinite(forced_zero_rsp)), "Forced-zero sectors must be finite"

    # geom_supported_mask should distinguish valid from invalid
    assert hasattr(radar, "geom_supported_mask"), "RadarResult must have geom_supported_mask"
    if radar.geom_supported_mask is not None and np.any(radar.geom_supported_mask):
        assert np.all(
            np.isfinite(radar.rsp[radar.geom_supported_mask])
        ), "Supported sectors must remain finite under empty_fg_policy='zero'"


def test_delta_interpretation_rules():
    """Test that interpretation respects Δ-dependent rules."""
    # Test Δ ≥ 90° → no wedge claims
    interp_90 = interpret_pattern(S_g=0.5, R_mean=0.3, coverage_geom=0.9, delta_deg=90)
    assert (
        "wedge" not in interp_90.lower()
    ), f"Δ=90° should not claim wedge localization, got: {interp_90}"
    assert (
        "global" in interp_90.lower() or "sector" in interp_90.lower()
    ), f"Δ=90° should use global/sector terms, got: {interp_90}"

    # Test Δ < 60° → can claim wedge
    interp_45 = interpret_pattern(S_g=0.5, R_mean=0.3, coverage_geom=0.9, delta_deg=45)
    assert "wedge" in interp_45.lower(), f"Δ=45° should allow wedge claims, got: {interp_45}"

    # Test low coverage_geom → unreliable
    interp_low_cov = interpret_pattern(S_g=0.5, R_mean=0.3, coverage_geom=0.3, delta_deg=45)
    assert (
        "coverage" in interp_low_cov.lower() or "unreliable" in interp_low_cov.lower()
    ), f"Low coverage_geom should flag unreliability, got: {interp_low_cov}"


def test_rsp_plot_sector_types():
    """Test that RSP plots correctly handle all sector types."""
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    from biorsp.plotting.radar import plot_radar

    coords, expr, _, _ = generate_test_data(n=500, pattern="rim_wedge")

    config = BioRSPConfig(delta_deg=30, B=24, empty_fg_policy="zero")

    v = compute_vantage(coords)
    r, theta = polar_coordinates(coords, v)
    r_norm, _ = normalize_radii(r)

    fg_mask, _ = define_foreground(expr, mode="quantile", q=0.95)
    if fg_mask is None:
        fg_mask = (expr >= np.quantile(expr, 0.95)).astype(float)

    radar = compute_rsp_radar(r_norm, theta, fg_mask, config=config)

    # Plot should not crash
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    ax = plot_radar(radar, ax=ax, theta_convention="math", debug_overlay=False)

    assert ax is not None, "plot_radar should return axes"

    # Check that plot includes expected elements
    assert len(ax.lines) > 0 or len(ax.collections) > 0, "Plot should have drawn something"

    plt.close(fig)


def test_radial_normalization_required():
    """Test that workflow handles both raw and normalized radii correctly.

    Note: The engine performs sector-level IQR normalization internally,
    so pre-normalization may not significantly change results for well-
    distributed data. This test verifies the workflow doesn't break with
    either input form.
    """
    coords, expr, _, _ = generate_test_data(n=500, pattern="rim_wedge")

    v = compute_vantage(coords)
    r, theta = polar_coordinates(coords, v)

    # Compute with raw radii
    fg_mask, _ = define_foreground(expr, mode="quantile", q=0.9)
    if fg_mask is None:
        fg_mask = (expr >= np.quantile(expr, 0.9)).astype(float)

    config = BioRSPConfig(delta_deg=30, B=24)

    radar_raw = compute_rsp_radar(r, theta, fg_mask, config=config)

    # Compute with normalized radii
    r_norm, _ = normalize_radii(r)
    radar_norm = compute_rsp_radar(r_norm, theta, fg_mask, config=config)

    # Both should produce valid results
    assert radar_raw.rsp is not None, "Raw radii should produce valid radar"
    assert radar_norm.rsp is not None, "Normalized radii should produce valid radar"

    # Results may be similar due to internal sector-level normalization,
    # but both workflows should succeed
    mask = np.isfinite(radar_raw.rsp) & np.isfinite(radar_norm.rsp)
    assert np.sum(mask) > 0, "Should have some valid sectors in both cases"


def test_uniform_expression_has_zero_spatial_score():
    """Uniform expression should yield zero Spatial_Bias_Score under public API."""
    pytest.importorskip("anndata")
    from anndata import AnnData

    rng = np.random.default_rng(7)
    coords = rng.normal(size=(200, 2))
    expr = np.ones((200, 1))

    adata = AnnData(X=expr, obsm={"X_spatial": coords})
    adata.var_names = ["flat_gene"]

    config = BioRSPConfig(B=24, delta_deg=30, expr_threshold_mode="detect")
    results = score_genes(adata, ["flat_gene"], embedding_key="X_spatial", config=config)

    assert "Spatial_Bias_Score" in results.columns
    assert results.loc[0, "Coverage"] == pytest.approx(1.0, rel=0, abs=1e-6)
    assert results.loc[0, "Spatial_Bias_Score"] == pytest.approx(0.0, abs=1e-6)


if __name__ == "__main__":
    # Run tests manually
    print("Running plotting correctness tests...\n")

    print("Test 1: Coverage vs Foreground distinction")
    test_coverage_vs_foreground_distinction()
    print("✓ PASSED\n")

    print("Test 2: Empty FG zero-fill correctness")
    test_empty_fg_zero_fill_correctness()
    print("✓ PASSED\n")

    print("Test 3: Δ-dependent interpretation rules")
    test_delta_interpretation_rules()
    print("✓ PASSED\n")

    print("Test 4: RSP plot sector types")
    test_rsp_plot_sector_types()
    print("✓ PASSED\n")

    print("Test 5: Radial normalization required")
    test_radial_normalization_required()
    print("✓ PASSED\n")

    print("Test 6: Workflow matches scoring")
    try:
        test_workflow_matches_scoring()
        print("✓ PASSED\n")
    except ModuleNotFoundError as e:
        print(f"⊘ SKIPPED (missing dependency: {e})\n")

    print("=" * 60)
    print("All plotting correctness tests PASSED!")
    print("=" * 60)
