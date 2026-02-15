import numpy as np
import pandas as pd

from biorsp.scoring import (
    bh_fdr,
    classify_row,
    compute_T,
    coverage_from_null,
    peak_count,
    qc_metrics,
    robust_z,
)


def test_compute_t_and_robust_z():
    e_phi = np.array([-0.2, 0.1, 0.4, -0.3], dtype=float)
    assert np.isclose(compute_T(e_phi), 0.4)

    null = np.array([0.1, 0.2, 0.1, 0.3, 0.2], dtype=float)
    z = robust_z(0.4, null)
    assert np.isfinite(z)
    assert z > 0


def test_bh_fdr_basic():
    pvals = np.array([0.01, 0.02, 0.10, 0.20], dtype=float)
    qvals = bh_fdr(pvals)
    assert qvals.shape == pvals.shape
    assert np.all((qvals >= 0.0) & (qvals <= 1.0))
    assert np.all(np.diff(np.sort(qvals)) >= -1e-12)


def test_coverage_and_peak_count():
    e_obs = np.array([0.0, 0.9, 0.0, 0.0, 0.85, 0.0, 0.0, 0.0], dtype=float)
    null_e = np.tile(
        np.array([0.0, 0.05, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0], dtype=float),
        (40, 1),
    )
    cov = coverage_from_null(e_obs, null_e, q=0.95)
    k = peak_count(e_obs, null_e, smooth_w=3, q_prom=0.95)
    assert 0.0 <= cov <= 1.0
    assert k >= 1


def test_qc_metrics_and_classify():
    n = 12
    expr = np.arange(n, dtype=float)
    obs = pd.DataFrame(
        {
            "total_counts": np.arange(n, dtype=float),
            "pct_counts_mt": np.arange(n, 0, -1, dtype=float),
            "pct_counts_ribo": np.ones(n, dtype=float),
        }
    )
    qc = qc_metrics(
        expr_or_f=expr,
        adata_obs=obs,
        covariate_candidates={
            "total_counts": ["total_counts"],
            "pct_counts_mt": ["pct_counts_mt"],
            "pct_counts_ribo": ["pct_counts_ribo"],
        },
    )
    assert qc["depth_key"] == "total_counts"
    assert np.isfinite(qc["rho_depth"])
    assert qc["qc_risk"] >= 0.0

    thresholds = {"q_sig": 0.05, "high_prev": 0.6, "qc_thresh": 0.35}
    row = {
        "underpowered": False,
        "q_T": 0.01,
        "prev": 0.2,
        "peaks_K": 2,
        "qc_risk": 0.1,
    }
    assert classify_row(row, thresholds) == "Localized–multimodal"

    row["qc_risk"] = 0.5
    assert classify_row(row, thresholds) == "QC-driven"

    row["underpowered"] = True
    assert classify_row(row, thresholds) == "Underpowered"
