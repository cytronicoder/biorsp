"""Scoring utilities and higher-level gene diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from biorsp.core.compute import compute_rsp
from biorsp.core.types import RSPConfig
from biorsp.scoring import bh_fdr as _bh_fdr
from biorsp.scoring import classify_row as _classify_row
from biorsp.scoring import compute_T as _compute_T
from biorsp.scoring import coverage_from_null as _coverage_from_null
from biorsp.scoring import detect_qc_columns as _detect_qc_columns
from biorsp.scoring import (
    donor_effective_counts as _donor_effective_counts,
)
from biorsp.scoring import (
    evaluate_underpowered as _evaluate_underpowered,
)
from biorsp.scoring import peak_count as _peak_count
from biorsp.scoring import qc_metrics as _qc_metrics
from biorsp.scoring import qc_risk_from_covariates as _qc_risk_from_covariates
from biorsp.scoring import robust_z as _robust_z

DEFAULT_QC_THRESH = 0.35


def donor_effective_counts(
    donor_ids: np.ndarray,
    f: np.ndarray,
    min_fg_per_donor: int = 10,
    min_bg_per_donor: int = 10,
) -> dict[str, Any]:
    """Re-export donor-effective support summary from canonical scoring module."""
    return _donor_effective_counts(
        donor_ids=donor_ids,
        f=f,
        min_fg_per_donor=min_fg_per_donor,
        min_bg_per_donor=min_bg_per_donor,
    )


def evaluate_underpowered(
    *,
    donor_ids: np.ndarray,
    f: np.ndarray,
    n_perm: int,
    p_min: float = 0.005,
    min_fg_total: int = 50,
    min_fg_per_donor: int = 10,
    min_bg_per_donor: int = 10,
    d_eff_min: int = 2,
    min_perm: int = 200,
) -> dict[str, Any]:
    """Re-export donor-effective underpowered gating from canonical scoring module."""
    return _evaluate_underpowered(
        donor_ids=donor_ids,
        f=f,
        n_perm=n_perm,
        p_min=p_min,
        min_fg_total=min_fg_total,
        min_fg_per_donor=min_fg_per_donor,
        min_bg_per_donor=min_bg_per_donor,
        d_eff_min=d_eff_min,
        min_perm=min_perm,
    )


def compute_T(E_theta: np.ndarray) -> float:
    return _compute_T(E_theta)


def robust_z(x_obs: float, x_null: np.ndarray, eps: float = 1e-12) -> float:
    return _robust_z(x_obs, x_null, eps)


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    return _bh_fdr(pvals)


def coverage_from_null(E_obs: np.ndarray, null_E: np.ndarray, q: float = 0.95) -> float:
    return _coverage_from_null(E_obs, null_E, q)


def peak_count(
    E_obs: np.ndarray, null_E: np.ndarray, smooth_w: int = 3, q_prom: float = 0.95
) -> int:
    return _peak_count(E_obs, null_E, smooth_w, q_prom)


def qc_metrics(
    expr_or_f: np.ndarray,
    adata_obs: pd.DataFrame,
    covariate_candidates: dict[str, list[str]],
) -> dict[str, Any]:
    return _qc_metrics(expr_or_f, adata_obs, covariate_candidates)


def detect_qc_columns(
    adata_obs: pd.DataFrame,
    covariate_candidates: dict[str, list[str]],
) -> dict[str, str | None]:
    return _detect_qc_columns(adata_obs, covariate_candidates)


def qc_risk_from_covariates(
    expr_or_f: np.ndarray,
    adata_obs: pd.DataFrame,
    qc_columns: dict[str, str | None],
) -> dict[str, float]:
    return _qc_risk_from_covariates(expr_or_f, adata_obs, qc_columns)


def classify_row(row: dict[str, Any] | pd.Series, thresholds: dict[str, Any]) -> str:
    return _classify_row(row, thresholds)


@dataclass(frozen=True)
class ScoreGeneConfig:
    basis: str = "X_umap"
    bins: int = 72
    threshold: float = 0.0
    center_method: str = "median"


def make_foreground_mask(x: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    arr = np.asarray(x, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError("x must contain at least one value.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("x must be finite.")
    return arr > float(threshold)


def score_gene(
    *,
    expr: np.ndarray,
    embedding_xy: np.ndarray,
    feature_label: str,
    config: ScoreGeneConfig | None = None,
) -> dict[str, Any]:
    cfg = config or ScoreGeneConfig()
    out = compute_rsp(
        expr=np.asarray(expr, dtype=float),
        embedding_xy=np.asarray(embedding_xy, dtype=float),
        config=RSPConfig(
            basis=cfg.basis,
            bins=int(cfg.bins),
            center_method=cfg.center_method,
            threshold=float(cfg.threshold),
            feature_label=feature_label,
        ),
        feature_label=feature_label,
    )
    return {
        "gene": feature_label,
        "anisotropy": float(out.anisotropy),
        "peak_direction": float(out.peak_direction),
        "breadth": float(out.breadth),
        "coverage": float(out.coverage),
        "E_max": float(out.E_max),
    }


def diagnose_random_gene_scores(*args, **kwargs) -> dict[str, Any]:
    raise NotImplementedError(
        "diagnose_random_gene_scores is removed from the public scoring module in this refactor."
    )
