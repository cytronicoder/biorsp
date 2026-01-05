import numpy as np
import pytest

from biorsp.core.qc import compute_gene_qc, compute_sector_qc, kish_effective_sample_size
from biorsp.utils.config import BioRSPConfig
from biorsp.utils.constants import (
    REASON_GENE_LOW_COVERAGE,
    REASON_GENE_TOO_FEW_SECTORS,
    REASON_GENE_UNDERPOWERED,
    REASON_OK,
    REASON_SECTOR_DEGENERATE_SCALE,
    REASON_SECTOR_FG_TOO_SMALL,
)


def test_kish_effective_sample_size():
    # Uniform weights: n_eff should be N
    w_uniform = np.ones(10)
    assert kish_effective_sample_size(w_uniform) == pytest.approx(10.0)

    # One heavy weight, others zero: n_eff should be 1
    w_heavy = np.zeros(10)
    w_heavy[0] = 1.0
    assert kish_effective_sample_size(w_heavy) == pytest.approx(1.0)

    # Half ones, half zeros: n_eff should be 5
    w_half = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    assert kish_effective_sample_size(w_half) == pytest.approx(5.0)


def test_sector_qc_binary():
    config = BioRSPConfig(qc_mode="principled", min_fg_sector=10, min_bg_sector=50, min_scale=0.01)

    # Pass
    y_pass = np.concatenate([np.ones(10), np.zeros(50)])
    valid, reason, metrics = compute_sector_qc(y_pass, 0.05, config)
    assert valid is True
    assert reason == REASON_OK
    assert metrics["nF"] == 10
    assert metrics["nB"] == 50

    # Fail low fg
    y_fail_fg = np.concatenate([np.ones(5), np.zeros(50)])
    valid, reason, metrics = compute_sector_qc(y_fail_fg, 0.05, config)
    assert valid is False
    assert reason == REASON_SECTOR_FG_TOO_SMALL

    # Fail degenerate scale
    valid, reason, metrics = compute_sector_qc(y_pass, 0.001, config)
    assert valid is False
    assert reason == REASON_SECTOR_DEGENERATE_SCALE


def test_sector_qc_weighted():
    config = BioRSPConfig(
        foreground_mode="weights",
        qc_mode="principled",
        min_fg_eff=5.0,
        min_bg_eff=10.0,
        min_scale=0.01,
    )

    # Pass
    y_pass = np.concatenate([np.ones(5), np.zeros(10)])
    valid, reason, metrics = compute_sector_qc(y_pass, 0.05, config)
    assert valid is True
    assert reason == REASON_OK
    assert metrics["nF_eff"] == pytest.approx(5.0)
    assert metrics["nB_eff"] == pytest.approx(10.0)

    # Fail low fg_eff (sparse weights)
    y_sparse = np.zeros(15)
    y_sparse[0] = 5.0  # sum is 5, but n_eff is 1
    valid, reason, metrics = compute_sector_qc(y_sparse, 0.05, config)
    assert valid is False
    assert reason == REASON_SECTOR_FG_TOO_SMALL
    assert metrics["nF_eff"] == pytest.approx(1.0)


def test_gene_qc():
    config = BioRSPConfig(
        qc_mode="principled",
        min_fg_total=100,
        min_coverage=0.5,
        min_valid_sectors=3,
        B=10,
    )

    # Pass
    mask_pass = np.array([True, True, True, True, True, False, False, False, False, False])
    reasons = [REASON_OK] * 5 + ["low_fg"] * 5
    valid, reason, metrics = compute_gene_qc(mask_pass, reasons, 150, config)
    assert valid is True
    assert reason == REASON_OK
    assert metrics["coverage"] == 0.5

    # Fail low total signal
    valid, reason, metrics = compute_gene_qc(mask_pass, reasons, 50, config)
    assert valid is False
    assert reason == REASON_GENE_UNDERPOWERED

    # Fail low coverage
    mask_low_cov = np.array([True, True, False, False, False, False, False, False, False, False])
    valid, reason, metrics = compute_gene_qc(mask_low_cov, reasons, 150, config)
    assert valid is False
    assert reason == REASON_GENE_LOW_COVERAGE

    # Fail too few sectors
    config_strict = BioRSPConfig(min_valid_sectors=5, B=10, min_coverage=0.1)
    mask_few = np.array([True, True, True, False, False, False, False, False, False, False])
    valid, reason, metrics = compute_gene_qc(mask_few, reasons, 150, config_strict)
    assert valid is False
    assert reason == REASON_GENE_TOO_FEW_SECTORS
