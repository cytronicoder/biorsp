import numpy as np
import pytest
from biorsp.utils import compute_sector_weight
from biorsp.core import sector_signed_stat, compute_rsp_radar, compute_anisotropy
from biorsp.config import BioRSPConfig

def test_compute_sector_weight():
    # Test none mode
    assert compute_sector_weight(10, 10, mode="none") == 1.0
    
    # Test sqrt_frac
    # w = sqrt(nF / (nF + nB))
    assert compute_sector_weight(10, 30, mode="sqrt_frac") == np.sqrt(10/40)
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
    y[:50] = 1.0 # Foreground at small radii
    
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
    theta = np.random.uniform(0, 2*np.pi, N)
    y = np.random.binomial(1, 0.1, N).astype(float)
    
    config = BioRSPConfig(sector_weight_mode="effective_min", sector_weight_k=10)
    
    radar = compute_rsp_radar(r, theta, y, config=config)
    
    assert radar.sector_weights is not None
    assert len(radar.sector_weights) == config.B
    
    # Check that rsp = weight * raw_stat
    for b in range(config.B):
        if not np.isnan(radar.rsp[b]):
            # If we re-run without weighting, we should get the raw stat
            config_none = BioRSPConfig(sector_weight_mode="none", B=config.B, delta_deg=config.delta_deg)
            # We need to use the same sector indices to be sure
            from biorsp.geometry import get_sector_indices
            sector_indices = get_sector_indices(theta, config.B, config.delta_deg)
            radar_none = compute_rsp_radar(r, theta, y, config=config_none, sector_indices=sector_indices)
            
            assert pytest.approx(radar.rsp[b]) == radar.sector_weights[b] * radar_none.rsp[b]

def test_permutation_weight_reuse():
    from biorsp.inference import compute_p_value
    
    N = 200
    r = np.random.uniform(0, 1, N)
    theta = np.random.uniform(0, 2*np.pi, N)
    y = np.random.binomial(1, 0.2, N).astype(float)
    
    config = BioRSPConfig(sector_weight_mode="effective_min", sector_weight_k=5, n_permutations=10)
    
    # This should run without error and use the weights
    result = compute_p_value(r, theta, y, config=config, n_perm=10)
    assert result.observed_stat is not None

def test_weighting_reduces_jaggedness():
    # Create a profile with alternating high and low support but same effect
    np.random.seed(42)
    B = 20
    centers = np.linspace(0, 2*np.pi, B, endpoint=False)
    
    rs = []
    thetas = []
    ys = []
    
    for i, center in enumerate(centers):
        if i % 2 == 0:
            n = 100 # High support
        else:
            n = 10  # Low support
            
        ri = np.random.uniform(0, 1, n)
        ti = np.random.normal(center, 0.05, n)
        # Effect: foreground at small radii
        yi = np.zeros(n)
        yi[:n//2] = 1.0
        ri[:n//2] = np.random.uniform(0, 0.3, n//2)
        ri[n//2:] = np.random.uniform(0.7, 1.0, n - n//2)
        
        rs.append(ri)
        thetas.append(ti)
        ys.append(yi)
        
    r = np.concatenate(rs)
    theta = np.concatenate(thetas) % (2*np.pi)
    y = np.concatenate(ys)
    
    config_none = BioRSPConfig(B=B, sector_weight_mode="none")
    config_weighted = BioRSPConfig(B=B, sector_weight_mode="effective_min", sector_weight_k=20)
    
    radar_none = compute_rsp_radar(r, theta, y, config=config_none)
    radar_weighted = compute_rsp_radar(r, theta, y, config=config_weighted)
    
    # Compute "jaggedness" as RMS of differences between adjacent sectors
    def get_jaggedness(rsp):
        valid = ~np.isnan(rsp)
        if np.sum(valid) < 2:
            return 0
        # Circular diff
        v_rsp = rsp[valid]
        diffs = np.diff(v_rsp, append=v_rsp[0])
        return np.sqrt(np.mean(diffs**2))
    
    j_none = get_jaggedness(radar_none.rsp)
    j_weighted = get_jaggedness(radar_weighted.rsp)
    
    # Weighted should be less jagged because low-support sectors (which are noisy) are shrunk towards 0
    assert j_weighted < j_none
