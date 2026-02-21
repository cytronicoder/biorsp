"""Power/feasibility helpers for donor-aware gating."""

from __future__ import annotations

from typing import Any

import numpy as np

from biorsp.scoring import donor_effective_counts as _donor_effective_counts
from biorsp.scoring import evaluate_underpowered as _evaluate_underpowered


def donor_effective_counts(
    donor_ids: np.ndarray,
    f: np.ndarray,
    min_fg_per_donor: int = 10,
    min_bg_per_donor: int = 10,
) -> dict[str, Any]:
    """Return donor-effective support summary for one gene."""
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
    """Apply donor-effective gating for underpowered genes."""
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
