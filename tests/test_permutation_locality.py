import numpy as np
from scipy.stats import kstest

from biorsp.core.engine import assign_sectors, compute_rsp_radar
from biorsp.core.geometry import prepare_polar
from biorsp.core.scoring import (
    _build_knn_blocks,
    _compute_spatial_score_from_radar,
    _permute_p_value,
)
from biorsp.utils.config import BioRSPConfig


def _compute_p_value(coords, y, config, knn_blocks=None, perm_mode="global", seed=0):
    prep = prepare_polar(
        coords,
        seed=seed,
        vantage="centroid",
        fixed_vantage=None,
        radius_norm="quantile",
        radius_q=0.95,
    )
    sector_index = assign_sectors(
        prep.theta,
        prep.r_norm,
        B=config.B,
        n_radial=config.n_r_bins,
        radial_rule=config.radial_rule,
        seed=seed,
    )
    sector_index.sector_indices = None
    radar = compute_rsp_radar(
        prep.r_norm,
        prep.theta,
        y,
        config=config,
        sector_index=sector_index,
    )
    s_obs, _, _, _, _ = _compute_spatial_score_from_radar(radar)
    p_val, _, _, _, _ = _permute_p_value(
        coords,
        prep.r_norm,
        prep.theta,
        y,
        sector_index,
        config,
        s_obs,
        fixed_geom_mask=radar.geom_supported_mask,
        fixed_weights=radar.sector_weights,
        knn_blocks=knn_blocks,
        perm_mode=perm_mode,
        rng_seed=seed,
    )
    return p_val


def test_permutation_pvalue_floor_and_uniform_iid():
    rng = np.random.default_rng(0)
    n_cells = 200
    coords = rng.normal(size=(n_cells, 2))

    config = BioRSPConfig(
        B=12,
        delta_deg=30.0,
        n_permutations=200,
        perm_mode_scoring="global",
        min_total_per_sector=1,
        min_fg_sector=1,
        min_bg_sector=1,
        min_scale=0.0,
    )

    p_values = []
    for i in range(25):
        y = rng.integers(0, 2, size=n_cells)
        p_values.append(_compute_p_value(coords, y, config, perm_mode="global", seed=i))

    p_values = np.array(p_values)
    p_min = 1.0 / (config.n_permutations + 1)
    assert np.all(p_values >= p_min)
    ks_stat, p = kstest(p_values, "uniform")
    assert ks_stat < 0.35
    assert p > 0.001


def test_structured_null_knn_block_improves_calibration():
    rng = np.random.default_rng(2)
    n_cells = 240
    coords = rng.normal(size=(n_cells, 2))

    config = BioRSPConfig(
        B=12,
        delta_deg=30.0,
        n_permutations=200,
        perm_mode_scoring="global",
        knn_k=8,
        knn_block_size=30,
        min_total_per_sector=1,
        min_fg_sector=1,
        min_bg_sector=1,
        min_scale=0.0,
    )

    knn_blocks = _build_knn_blocks(coords, config.knn_k, config.knn_block_size, seed=3)
    block_labels = np.zeros(n_cells)
    for idx, block in enumerate(knn_blocks):
        block_labels[block] = idx % 2

    p_global = []
    p_local = []
    for i in range(20):
        noise = rng.normal(scale=0.05, size=n_cells)
        y = (block_labels + noise > 0.5).astype(float)
        p_global.append(_compute_p_value(coords, y, config, perm_mode="global", seed=i))
        p_local.append(
            _compute_p_value(
                coords, y, config, knn_blocks=knn_blocks, perm_mode="knn_block", seed=i
            )
        )

    assert np.median(p_local) > np.median(p_global)
