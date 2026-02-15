import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-test")

import numpy as np

from biorsp.staged_pipeline import run_scope_staged


def _toy_scope(tmp_path: Path):
    rng = np.random.default_rng(3)
    n_cells = 160
    angles = np.linspace(0.0, 2.0 * np.pi, n_cells, endpoint=False)
    donor_ids = np.array(["d1"] * (n_cells // 2) + ["d2"] * (n_cells // 2))

    X = np.zeros((n_cells, 3), dtype=float)

    # Strong directional signal.
    X[:40, 0] = 1.0

    # Diffuse signal.
    diffuse_idx = rng.choice(n_cells, size=50, replace=False)
    X[diffuse_idx, 1] = 1.0

    # Near-null weak signal.
    weak_idx = rng.choice(n_cells, size=20, replace=False)
    X[weak_idx, 2] = 1.0

    genes = ["G_STRONG", "G_DIFFUSE", "G_WEAK"]
    scope_ctx = {
        "scope_id": "toy_scope",
        "scope_name": "Toy Scope",
        "scope_level": "global",
        "out_dir": tmp_path / "scope",
        "figure_dir": tmp_path / "figures",
        "angles": angles,
        "umap_xy": np.c_[np.cos(angles), np.sin(angles)],
        "donor_ids": donor_ids,
    }
    return scope_ctx, X, genes


def test_adaptive_permutations_escalate_selectively(tmp_path):
    scope_ctx, X, genes = _toy_scope(tmp_path)
    params = {
        "discovery_mode": True,
        "pipeline_mode": "compute",
        "bins_screen": 24,
        "bins_confirm": 36,
        "stage1_top_k_global": 3,
        "stage2_top_k_global": 3,
        "min_fg_global": 5,
        "min_prev_global": 0.01,
        "perm_init": 20,
        "perm_mid": 60,
        "perm_final": 120,
        "p_escalate_1": 0.6,
        "p_escalate_2": 0.2,
        "seed": 17,
    }
    out = run_scope_staged(
        scope_ctx=scope_ctx,
        X=X,
        genes=genes,
        labels=None,
        qc_covariates=None,
        cache_dir=tmp_path / "cache",
        params=params,
    )
    stage3 = out["stage3"]
    assert not stage3.empty
    assert "n_perm_final_used" in stage3.columns
    assert int(stage3["n_perm_final_used"].max()) >= params["perm_mid"]
    assert int(stage3["n_perm_final_used"].min()) == params["perm_init"]


def test_compute_then_plot_mode_separation(tmp_path):
    scope_ctx, X, genes = _toy_scope(tmp_path)
    common = {
        "discovery_mode": True,
        "bins_screen": 24,
        "bins_confirm": 36,
        "stage1_top_k_global": 3,
        "stage2_top_k_global": 2,
        "min_fg_global": 5,
        "min_prev_global": 0.01,
        "perm_init": 20,
        "perm_mid": 40,
        "perm_final": 60,
        "p_escalate_1": 0.6,
        "p_escalate_2": 0.2,
        "plot_top_k": 2,
        "skip_umap_plots": True,
        "skip_pair_plots": True,
        "skip_rsp_plots": False,
        "seed": 21,
    }

    compute_params = dict(common)
    compute_params["pipeline_mode"] = "compute"
    compute_out = run_scope_staged(
        scope_ctx=scope_ctx,
        X=X,
        genes=genes,
        labels=None,
        qc_covariates=None,
        cache_dir=tmp_path / "cache",
        params=compute_params,
    )

    pngs_after_compute = list((tmp_path / "figures").glob("*.png"))
    assert pngs_after_compute == []

    plot_params = dict(common)
    plot_params["pipeline_mode"] = "plot"
    plot_out = run_scope_staged(
        scope_ctx=scope_ctx,
        X=X,
        genes=genes,
        labels=None,
        qc_covariates=None,
        cache_dir=tmp_path / "cache",
        params=plot_params,
    )
    reps = plot_out["representatives"]
    pngs_after_plot = list((tmp_path / "figures").glob("*_rsp_polar.png"))
    assert len(pngs_after_plot) == int(reps.shape[0])
    assert len(pngs_after_plot) > 0

    # Sanity check parity for observed profile statistic columns persisted through plot-only run.
    np.testing.assert_allclose(
        compute_out["stage3"].sort_values("gene")["T_obs"].to_numpy(),
        plot_out["stage3"].sort_values("gene")["T_obs"].to_numpy(),
        atol=0.0,
        rtol=0.0,
    )
