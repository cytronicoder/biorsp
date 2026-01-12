import numpy as np
import pytest

from biorsp.core.engine import compute_anisotropy, compute_rsp_radar, sector_signed_stat
from biorsp.utils.config import BioRSPConfig
from biorsp.utils.helpers import compute_sector_weight


def test_compute_sector_weight():
    assert compute_sector_weight(10, 10, mode="none") == 1.0

    assert compute_sector_weight(10, 30, mode="sqrt_frac") == np.sqrt(10 / 40)
    assert compute_sector_weight(0, 10, mode="sqrt_frac") == 0.0

    assert compute_sector_weight(10, 100, mode="effective_min", k=10) == 10 / (10 + 10)
    assert compute_sector_weight(100, 10, mode="effective_min", k=10) == 10 / (10 + 10)
    assert compute_sector_weight(0, 10, mode="effective_min", k=10) == 0.0

    assert pytest.approx(compute_sector_weight(10, 10, mode="logistic_support", k=10)) == 0.5

    assert compute_sector_weight(100, 100, mode="logistic_support", k=10) > 0.99

    assert compute_sector_weight(1, 1, mode="logistic_support", k=10) < 0.05


def test_sector_signed_stat_weighting():
    r = np.linspace(0, 1, 100)
    y = np.zeros(100)
    y[:50] = 1.0

    idx = np.arange(100)

    res_none = sector_signed_stat(r, y, idx, weight_mode="none")

    res_weighted = sector_signed_stat(r, y, idx, weight_mode="effective_min", weight_k=50)

    assert res_weighted["support_weight"] == 0.5
    assert pytest.approx(res_weighted["stat"]) == 0.5 * res_none["stat"]
    assert res_weighted["stat_raw"] == res_none["stat"]


def test_compute_rsp_radar_weighting():
    """Test that sector weighting works correctly (contract: rsp is RAW)."""

    N = 1000
    r = np.random.uniform(0, 1, N)
    theta = np.random.uniform(0, 2 * np.pi, N)
    y = np.random.binomial(1, 0.1, N).astype(float)

    config = BioRSPConfig(sector_weight_mode="effective_min", sector_weight_k=10)

    radar = compute_rsp_radar(r, theta, y, config=config)

    assert radar.sector_weights is not None
    assert len(radar.sector_weights) == config.B

    # Contract: rsp is RAW, weights are stored separately
    # Verify weights are positive where rsp is valid
    for b in range(config.B):
        if not np.isnan(radar.rsp[b]):
            assert radar.sector_weights[b] >= 0, "Weights should be non-negative"

    # Verify weighted S_g can be computed
    valid_mask = np.isfinite(radar.rsp)
    if np.any(valid_mask):
        w = radar.sector_weights[valid_mask]
        rsp = radar.rsp[valid_mask]
        s_g = np.sqrt(np.sum(w * rsp**2) / np.sum(w))
        assert np.isfinite(s_g), "Weighted S_g should be finite"


def test_permutation_weight_reuse():
    from biorsp.core.inference import compute_p_value

    N = 200
    r = np.random.uniform(0, 1, N)
    theta = np.random.uniform(0, 2 * np.pi, N)
    y = np.random.binomial(1, 0.2, N).astype(float)

    config = BioRSPConfig(sector_weight_mode="effective_min", sector_weight_k=5, n_permutations=10)

    result = compute_p_value(r, theta, y, config=config, n_perm=10)
    assert result.observed_stat is not None


def test_weighting_reduces_variance_of_low_support_sectors():
    """
    Test that weighting reduces the influence of low-support sectors on summary metrics.

    Property being tested: When we perturb a low-support sector's value, the weighted
    anisotropy should be more stable than the unweighted anisotropy. This tests that
    weighting achieves its intended goal of down-weighting noisy low-support sectors.

    This is a deterministic, non-flaky test of a fundamental weighting property.
    """
    np.random.seed(42)
    B = 20
    centers = np.linspace(0, 2 * np.pi, B, endpoint=False)

    rs = []
    thetas = []
    ys = []

    for i, center in enumerate(centers):
        n = 100 if i % 2 == 0 else 10
        ri = np.random.uniform(0, 1, n)
        ti = np.random.normal(center, 0.05, n)

        yi = np.zeros(n)
        yi[: n // 2] = 1.0
        ri[: n // 2] = np.random.uniform(0, 0.3, n // 2)
        ri[n // 2 :] = np.random.uniform(0.7, 1.0, n - n // 2)

        rs.append(ri)
        thetas.append(ti)
        ys.append(yi)

    r = np.concatenate(rs)
    theta = np.concatenate(thetas) % (2 * np.pi)
    y = np.concatenate(ys)

    config_none = BioRSPConfig(B=B, sector_weight_mode="none", scale_mode="pooled_iqr")
    config_weighted = BioRSPConfig(
        B=B, sector_weight_mode="effective_min", sector_weight_k=20, scale_mode="pooled_iqr"
    )

    radar_none = compute_rsp_radar(r, theta, y, config=config_none)
    radar_weighted = compute_rsp_radar(r, theta, y, config=config_weighted)

    # Contract: rsp is RAW, weights are stored separately
    # For weighted anisotropy, we must apply weights in the aggregation
    valid_mask = ~np.isnan(radar_none.rsp)
    valid_mask_weighted = ~np.isnan(radar_weighted.rsp)

    # Unweighted anisotropy (uniform weights)
    A_none_baseline = compute_anisotropy(radar_none.rsp, valid_mask)
    # Weighted anisotropy (using sector_weights from weighted radar)
    A_weighted_baseline = compute_anisotropy(
        radar_weighted.rsp, valid_mask_weighted, weights=radar_weighted.sector_weights
    )

    low_support_idx = None
    for i in range(1, B, 2):
        if valid_mask[i]:
            low_support_idx = i
            break

    assert low_support_idx is not None, "Need at least one valid low-support sector"

    perturbation_cells = 5
    perturb_theta = centers[low_support_idx]
    perturb_r = np.random.uniform(0.05, 0.15, perturbation_cells)
    perturb_y = np.ones(perturbation_cells)

    r_perturbed = np.concatenate([r, perturb_r])
    theta_perturbed = np.concatenate([theta, np.full(perturbation_cells, perturb_theta)])
    y_perturbed = np.concatenate([y, perturb_y])

    radar_none_pert = compute_rsp_radar(
        r_perturbed, theta_perturbed, y_perturbed, config=config_none
    )
    radar_weighted_pert = compute_rsp_radar(
        r_perturbed, theta_perturbed, y_perturbed, config=config_weighted
    )

    valid_mask_pert = ~np.isnan(radar_none_pert.rsp)
    valid_mask_weighted_pert = ~np.isnan(radar_weighted_pert.rsp)

    # Unweighted anisotropy after perturbation
    A_none_pert = compute_anisotropy(radar_none_pert.rsp, valid_mask_pert)
    # Weighted anisotropy after perturbation
    A_weighted_pert = compute_anisotropy(
        radar_weighted_pert.rsp,
        valid_mask_weighted_pert,
        weights=radar_weighted_pert.sector_weights,
    )

    delta_none = abs(A_none_pert - A_none_baseline)
    delta_weighted = abs(A_weighted_pert - A_weighted_baseline)

    assert delta_weighted < delta_none * 0.8, (
        f"Weighting failed to reduce sensitivity: "
        f"unweighted Δ={delta_none:.4f}, weighted Δ={delta_weighted:.4f}"
    )

    assert np.isfinite(A_none_baseline) and A_none_baseline >= 0
    assert np.isfinite(A_weighted_baseline) and A_weighted_baseline >= 0
    assert np.isfinite(A_none_pert) and A_none_pert >= 0
    assert np.isfinite(A_weighted_pert) and A_weighted_pert >= 0
