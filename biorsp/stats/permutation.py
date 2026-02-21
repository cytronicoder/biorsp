"""Permutation null models for BioRSP statistics.

This module forwards to :mod:`biorsp.permutation` so root and stats APIs share
identical permutation logic.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from biorsp.permutation import (
    check_mode_consistency as _check_mode_consistency_root,
)
from biorsp.permutation import (
    perm_null_emax as _perm_null_emax_root,
)
from biorsp.permutation import (
    perm_null_T as _perm_null_T_root,
)
from biorsp.permutation import (
    perm_null_T_and_profile as _perm_null_T_and_profile_root,
)
from biorsp.permutation import (
    permute_foreground_within_donor as _permute_foreground_within_donor_root,
)


def permute_foreground_within_donor(
    f_obs: np.ndarray,
    donor_to_idx: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    """Compatibility wrapper for donor-stratified foreground permutation."""
    return _permute_foreground_within_donor_root(
        f_obs=f_obs,
        donor_to_idx=donor_to_idx,
        rng=rng,
    )


def perm_null_emax(
    expr: np.ndarray,
    theta: np.ndarray,
    donor_ids: np.ndarray,
    n_bins: int,
    n_perm: int,
    seed: int = 0,
) -> tuple[np.ndarray, float, float, float]:
    """Compatibility wrapper using ``theta`` naming."""
    return _perm_null_emax_root(
        expr=expr,
        angles=theta,
        donor_ids=donor_ids,
        n_bins=n_bins,
        n_perm=n_perm,
        seed=seed,
    )


def perm_null_T_and_profile(
    expr: np.ndarray | None = None,
    theta: np.ndarray | None = None,
    donor_ids: np.ndarray | None = None,
    n_bins: int = 36,
    n_perm: int = 300,
    seed: int = 0,
    donor_stratified: bool = True,
    *,
    f: np.ndarray | None = None,
    angles: np.ndarray | None = None,
    perm_indices: np.ndarray | None = None,
    perm_start: int = 0,
    perm_end: int | None = None,
    previous_null_T: np.ndarray | None = None,
    previous_null_E_phi: np.ndarray | None = None,
    bin_id: np.ndarray | None = None,
    bin_counts_total: np.ndarray | None = None,
    validate_stratified_counts: bool = False,
    validate_perm_checks: int = 5,
    return_null_profiles: bool = True,
    mode: str = "raw",
    smooth_w: int = 1,
) -> dict[str, Any]:
    """Compatibility wrapper for the canonical permutation implementation."""
    return _perm_null_T_and_profile_root(
        f=f if f is not None else expr,
        angles=angles if angles is not None else theta,
        donor_ids=donor_ids,
        n_bins=n_bins,
        n_perm=n_perm,
        seed=seed,
        donor_stratified=donor_stratified,
        expr=expr,
        theta=theta,
        perm_indices=perm_indices,
        perm_start=perm_start,
        perm_end=perm_end,
        previous_null_T=previous_null_T,
        previous_null_E_phi=previous_null_E_phi,
        bin_id=bin_id,
        bin_counts_total=bin_counts_total,
        validate_stratified_counts=validate_stratified_counts,
        validate_perm_checks=validate_perm_checks,
        return_null_profiles=return_null_profiles,
        mode=mode,
        smooth_w=smooth_w,
    )


def perm_null_T(
    f: np.ndarray,
    angles: np.ndarray,
    donor_ids: np.ndarray | None,
    n_bins: int,
    n_perm: int,
    seed: int,
    mode: str = "raw",
    smooth_w: int = 1,
    donor_stratified: bool = True,
    return_null_T: bool = True,
    return_obs_profile: bool = False,
    *,
    return_null_profiles: bool = False,
    debug: bool = False,
    debug_checks: int = 5,
    bin_id: np.ndarray | None = None,
    bin_counts_total: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compatibility wrapper for canonical mode-consistent max-stat permutation."""
    return _perm_null_T_root(
        f=f,
        angles=angles,
        donor_ids=donor_ids,
        n_bins=n_bins,
        n_perm=n_perm,
        seed=seed,
        mode=mode,
        smooth_w=smooth_w,
        donor_stratified=donor_stratified,
        return_null_T=return_null_T,
        return_obs_profile=return_obs_profile,
        return_null_profiles=return_null_profiles,
        debug=debug,
        debug_checks=debug_checks,
        bin_id=bin_id,
        bin_counts_total=bin_counts_total,
    )


def check_mode_consistency(
    f: np.ndarray,
    angles: np.ndarray,
    donor_ids: np.ndarray | None,
    n_bins: int,
    n_perm: int,
    seed: int,
    donor_stratified: bool = True,
) -> dict[str, float]:
    """Compatibility wrapper for mode-consistency self-check."""
    return _check_mode_consistency_root(
        f=f,
        angles=angles,
        donor_ids=donor_ids,
        n_bins=n_bins,
        n_perm=n_perm,
        seed=seed,
        donor_stratified=donor_stratified,
    )
