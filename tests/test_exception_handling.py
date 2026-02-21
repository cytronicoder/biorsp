from __future__ import annotations

import logging

import numpy as np
import pytest
import scipy.sparse as sp

from biorsp import evaluation, genomewide


def test_evaluation_safe_moran_expected_error_warns_and_continues(monkeypatch, caplog):
    caplog.set_level(logging.WARNING)

    def _raise(*_args, **_kwargs):
        raise ValueError("bad layer")

    monkeypatch.setattr(evaluation, "_get_expr_vector", _raise)
    val = evaluation._safe_moran(
        adata=object(),
        feature="GENE1",
        layer="lognorm",
        W=sp.eye(2),
        logger=logging.getLogger("test"),
        qc_mode="cleaned",
        stratum_name="Fibroblast",
    )
    assert np.isnan(val)
    assert "Moran skipped" in caplog.text
    assert "GENE1" in caplog.text


def test_evaluation_safe_moran_unexpected_error_propagates(monkeypatch):
    monkeypatch.setattr(
        evaluation, "_get_expr_vector", lambda *_args, **_kwargs: np.array([1.0, 2.0])
    )

    def _raise_runtime(*_args, **_kwargs):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(evaluation, "morans_i", _raise_runtime)
    with pytest.raises(RuntimeError, match="unexpected"):
        evaluation._safe_moran(
            adata=object(),
            feature="GENE1",
            layer="lognorm",
            W=sp.eye(2),
            logger=logging.getLogger("test"),
            qc_mode="cleaned",
            stratum_name="Fibroblast",
        )


def test_genomewide_safe_moran_expected_error_warns_and_continues(monkeypatch, caplog):
    caplog.set_level(logging.WARNING)
    monkeypatch.setattr(
        genomewide,
        "morans_i",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("bad expr")),
    )
    val = genomewide._safe_moran(
        expr=np.array([1.0, 2.0]),
        W=sp.eye(2),
        logger=logging.getLogger("test"),
        stratum_name="Fibroblast",
        feature="GENE1",
    )
    assert np.isnan(val)
    assert "Moran skipped" in caplog.text
    assert "GENE1" in caplog.text


def test_genomewide_safe_moran_unexpected_error_propagates(monkeypatch):
    monkeypatch.setattr(
        genomewide,
        "morans_i",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("unexpected")),
    )
    with pytest.raises(RuntimeError, match="unexpected"):
        genomewide._safe_moran(
            expr=np.array([1.0, 2.0]),
            W=sp.eye(2),
            logger=logging.getLogger("test"),
            stratum_name="Fibroblast",
            feature="GENE1",
        )
