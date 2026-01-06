import numpy as np
import pytest

from biorsp.core.engine import compute_anisotropy, compute_rsp_radar, sector_signed_stat
from biorsp.utils.config import BioRSPConfig
from biorsp.utils.helpers import compute_sector_weight


def test_compute_sector_weight():
    # Test none mode
    assert compute_sector_weight(10, 10, mode="none") == 1.0

    # Test sqrt_frac
    # w = sqrt(nF / (nF + nB))
    assert compute_sector_weight(10, 30, mode="sqrt_frac") == np.sqrt(10 / 40)
    assert compute_sector_weight(0, 10, mode="sqrt_frac") == 0.0

    # Test effective_min
    # w = min(nF, nB) / (min(nF, nB) + k)
    assert compute_sector_weight(10, 100, mode="effective_min", k=10) == 10 / (10 + 10)
    assert compute_sector_weight(100, 10, mode="effective_min", k=10) == 10 / (10 + 10)
    assert compute_sector_weight(0, 10, mode="effective_min", k=10) == 0.0

    # Test logistic_support
    # w = sigmoid((min(nF, nB) - k) / (k/4))
    # At m=k, w should be 0.5
    assert pytest.approx(compute_sector_weight(10, 10, mode="logistic_support", k=10)) == 0.5
    # High support -> 1
    assert compute_sector_weight(100, 100, mode="logistic_support", k=10) > 0.99
    # Low support -> 0
    assert compute_sector_weight(1, 1, mode="logistic_support", k=10) < 0.05


def test_sector_signed_stat_weighting():
    r = np.linspace(0, 1, 100)
    y = np.zeros(100)
    y[:50] = 1.0  # Foreground at small radii

    idx = np.arange(100)

    # No weighting
    res_none = sector_signed_stat(r, y, idx, weight_mode="none")

    # With weighting
    res_weighted = sector_signed_stat(r, y, idx, weight_mode="effective_min", weight_k=50)

    # nF=50, nB=50. w = 50 / (50 + 50) = 0.5
    assert res_weighted["support_weight"] == 0.5
    assert pytest.approx(res_weighted["stat"]) == 0.5 * res_none["stat"]
    assert res_weighted["stat_raw"] == res_none["stat"]


def test_compute_rsp_radar_weighting():
    # Create a synthetic case where one sector has low support
    N = 1000
    r = np.random.uniform(0, 1, N)
    theta = np.random.uniform(0, 2 * np.pi, N)
    y = np.random.binomial(1, 0.1, N).astype(float)

    config = BioRSPConfig(sector_weight_mode="effective_min", sector_weight_k=10)

    radar = compute_rsp_radar(r, theta, y, config=config)

    assert radar.sector_weights is not None
    assert len(radar.sector_weights) == config.B

    # Check that rsp = weight * raw_stat
    for b in range(config.B):
        if not np.isnan(radar.rsp[b]):
            # If we re-run without weighting, we should get the raw stat
            config_none = BioRSPConfig(
                sector_weight_mode="none", B=config.B, delta_deg=config.delta_deg
            )
            # We need to use the same sector indices to be sure
            from biorsp.preprocess.geometry import get_sector_indices

            sector_indices = get_sector_indices(theta, config.B, config.delta_deg)
            radar_none = compute_rsp_radar(
                r, theta, y, config=config_none, sector_indices=sector_indices
            )

            assert pytest.approx(radar.rsp[b]) == radar.sector_weights[b] * radar_none.rsp[b]


def test_permutation_weight_reuse():
    from biorsp.core.inference import compute_p_value

    N = 200
    r = np.random.uniform(0, 1, N)
    theta = np.random.uniform(0, 2 * np.pi, N)
    y = np.random.binomial(1, 0.2, N).astype(float)

    config = BioRSPConfig(sector_weight_mode="effective_min", sector_weight_k=5, n_permutations=10)

    # This should run without error and use the weights
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

    # Create synthetic data with alternating high and low support
    rs = []
    thetas = []
    ys = []

    for i, center in enumerate(centers):
        n = 100 if i % 2 == 0 else 10  # High vs low support
        ri = np.random.uniform(0, 1, n)
        ti = np.random.normal(center, 0.05, n)
        # Effect: foreground at small radii
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

    # Compute baseline radar profiles
    radar_none = compute_rsp_radar(r, theta, y, config=config_none)
    radar_weighted = compute_rsp_radar(r, theta, y, config=config_weighted)

    # Compute baseline anisotropies
    valid_mask = ~np.isnan(radar_none.rsp)
    A_none_baseline = compute_anisotropy(radar_none.rsp, valid_mask)
    A_weighted_baseline = compute_anisotropy(radar_weighted.rsp, valid_mask)

    # Now perturb a LOW-SUPPORT sector (odd indices have n=10)
    # Find a low-support sector that is valid in both
    low_support_idx = None
    for i in range(1, B, 2):  # odd sectors have low support
        if valid_mask[i]:
            low_support_idx = i
            break

    assert low_support_idx is not None, "Need at least one valid low-support sector"

    # Add a small number of cells to this specific sector with extreme radii
    # This simulates noise that should be down-weighted
    perturbation_cells = 5
    perturb_theta = centers[low_support_idx]
    perturb_r = np.random.uniform(0.05, 0.15, perturbation_cells)  # Very small radii
    perturb_y = np.ones(perturbation_cells)  # Foreground

    r_perturbed = np.concatenate([r, perturb_r])
    theta_perturbed = np.concatenate([theta, np.full(perturbation_cells, perturb_theta)])
    y_perturbed = np.concatenate([y, perturb_y])

    # Recompute with perturbation
    radar_none_pert = compute_rsp_radar(
        r_perturbed, theta_perturbed, y_perturbed, config=config_none
    )
    radar_weighted_pert = compute_rsp_radar(
        r_perturbed, theta_perturbed, y_perturbed, config=config_weighted
    )

    valid_mask_pert = ~np.isnan(radar_none_pert.rsp)
    A_none_pert = compute_anisotropy(radar_none_pert.rsp, valid_mask_pert)
    A_weighted_pert = compute_anisotropy(radar_weighted_pert.rsp, valid_mask_pert)

    # Compute absolute changes
    delta_none = abs(A_none_pert - A_none_baseline)
    delta_weighted = abs(A_weighted_pert - A_weighted_baseline)

    # Key assertion: Weighting should make the anisotropy more robust to
    # perturbations in low-support sectors
    # The weighted change should be smaller than the unweighted change
    assert delta_weighted < delta_none * 0.8, (
        f"Weighting failed to reduce sensitivity: "
        f"unweighted Δ={delta_none:.4f}, weighted Δ={delta_weighted:.4f}"
    )

    # Also verify both produce finite, non-negative anisotropies
    assert np.isfinite(A_none_baseline) and A_none_baseline >= 0
    assert np.isfinite(A_weighted_baseline) and A_weighted_baseline >= 0
    assert np.isfinite(A_none_pert) and A_none_pert >= 0
    assert np.isfinite(A_weighted_pert) and A_weighted_pert >= 0
