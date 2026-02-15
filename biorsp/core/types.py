"""Typed configuration and result containers for BioRSP core operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RSPConfig:
    """Core configuration for one RSP computation."""

    basis: str = "X_umap"
    bins: int = 72
    center_method: str = "median"
    threshold: float = 0.0
    feature_label: str | None = None


@dataclass(frozen=True)
class NullConfig:
    """Permutation/null configuration."""

    n_perm: int = 300
    seed: int = 0
    donor_stratified: bool = True


@dataclass(frozen=True)
class RSPResult:
    """Output of `compute_rsp`.

    - `theta`: angular bin centers in radians.
    - `R_theta`: directional contrast curve over theta.
    """

    theta: np.ndarray
    R_theta: np.ndarray
    anisotropy: float
    peak_direction: float
    peak_directions: np.ndarray
    breadth: float
    coverage: float
    E_max: float
    feature_label: str
    feature_index: int | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScopeResult:
    """Scope-level summary returned by hierarchy pipeline."""

    scope_name: str
    scope_kind: str
    outdir: str
    n_cells: int
    n_genes_scored: int
    n_plots_rsp: int
    n_plots_pair: int
    metadata_path: str
    stats_path: str
