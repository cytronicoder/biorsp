"""Donor-stratified permutation tests for RSP."""

from __future__ import annotations

import hashlib
import warnings
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from biorsp.rsp import compute_rsp_profile_from_boolean
from biorsp.scoring import compute_T
from biorsp.smoothing import circular_moving_average
from biorsp.utils import ensure_dir


def build_donor_index(adata, donor_col: str = "donor") -> dict[str, np.ndarray]:
    """Build mapping from donor ID to indices.

    Raises informative errors if donor column missing or insufficient donors.

    Args:
        adata: AnnData object.
        donor_col: Column name for donor IDs.

    Returns:
        Dictionary mapping donor IDs to cell indices.

    Raises:
        KeyError: If donor column not found.
        ValueError: If fewer than two donors present.
    """
    if donor_col not in adata.obs:
        raise KeyError(
            f"adata.obs['{donor_col}'] is required (missing donor labels risks pseudoreplication)."
        )
    donor_ids = np.asarray(adata.obs[donor_col])
    unique_donors = np.unique(donor_ids)
    if unique_donors.size < 2:
        raise ValueError(
            "At least two donors are required for donor-stratified permutation."
        )
    mapping: dict[str, np.ndarray] = {}
    for donor in unique_donors:
        idx = np.nonzero(donor_ids == donor)[0]
        mapping[str(donor)] = idx.astype(int)
    return mapping


def permute_foreground_within_donor(
    f: np.ndarray, donor_to_idx: dict[str, np.ndarray], rng: np.random.Generator
) -> np.ndarray:
    """Permute boolean foreground within donors while preserving per-donor counts.

    Args:
        f: Boolean foreground vector.
        donor_to_idx: Mapping from donor IDs to cell indices.
        rng: NumPy random generator.

    Returns:
        Permuted boolean foreground vector.

    Raises:
        ValueError: If foreground count exceeds donor cell count.
    """
    f_bool = np.asarray(f, dtype=bool).ravel()
    N = f_bool.size
    out = np.zeros(N, dtype=bool)
    for donor, idx in donor_to_idx.items():
        idx_arr = np.asarray(idx, dtype=int)
        if idx_arr.size == 0:
            continue
        k = int(f_bool[idx_arr].sum())
        if k > idx_arr.size:
            raise ValueError(
                f"Foreground count exceeds donor cell count for donor '{donor}'."
            )
        if k == 0:
            continue
        chosen = rng.choice(idx_arr, size=k, replace=False)
        out[chosen] = True
    return out


def perm_null_emax(
    expr: np.ndarray,
    angles: np.ndarray,
    donor_ids: np.ndarray,
    n_bins: int,
    n_perm: int,
    seed: int = 0,
) -> tuple[np.ndarray, float, float, float]:
    """Build null distribution of E_max by donor-stratified permutation.

    Args:
        expr: Gene expression vector (length N).
        angles: Angles in radians (length N).
        donor_ids: Donor ID per cell (length N).
        n_bins: Number of angular bins.
        n_perm: Number of permutations.
        seed: Random seed.

    Returns:
        Tuple of (null_emax, E_max_obs, phi_max_obs, p_value) where:
        - null_emax: Null E_max values (length n_perm).
        - E_max_obs: Observed E_max.
        - phi_max_obs: Observed phi_max.
        - p_value: One-sided permutation p-value.

    Raises:
        ValueError: If donor_ids length doesn't match expression vector length.
    """
    f_obs = np.asarray(expr).ravel() > 0
    donor_ids_arr = np.asarray(donor_ids)
    if donor_ids_arr.size != f_obs.size:
        raise ValueError("donor_ids length must match expression vector length.")

    # Build donor index
    unique_donors = np.unique(donor_ids_arr)
    if unique_donors.size < 2:
        raise ValueError("At least two donors are required for permutation testing.")
    donor_to_idx = {
        str(d): np.nonzero(donor_ids_arr == d)[0].astype(int) for d in unique_donors
    }

    E_phi_obs, phi_max_obs, E_max_obs = compute_rsp_profile_from_boolean(
        f_obs, angles, n_bins
    )

    rng = np.random.default_rng(seed)
    null_emax = np.zeros(n_perm, dtype=float)
    for i in range(n_perm):
        f_perm = permute_foreground_within_donor(f_obs, donor_to_idx, rng)
        _, _, E_max_perm = compute_rsp_profile_from_boolean(f_perm, angles, n_bins)
        null_emax[i] = E_max_perm

    p = (1.0 + np.sum(null_emax >= E_max_obs)) / (1.0 + n_perm)
    return null_emax, float(E_max_obs), float(phi_max_obs), float(p)


def _normalize_mode(mode: str, smooth_w: int) -> tuple[str, int]:
    mode_norm = str(mode).strip().lower()
    if mode_norm not in {"raw", "smoothed"}:
        raise ValueError("mode must be one of {'raw', 'smoothed'}.")
    if int(smooth_w) != smooth_w:
        raise ValueError("smooth_w must be an integer.")
    w = int(smooth_w)
    if mode_norm == "raw":
        return mode_norm, 1
    if w < 1 or w % 2 == 0:
        raise ValueError("smooth_w must be an odd integer >=1 in smoothed mode.")
    return mode_norm, w


def _resolve_fg_and_angles(
    *,
    f: np.ndarray | None,
    expr: np.ndarray | None,
    angles: np.ndarray | None,
    theta: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    if f is None:
        if expr is None:
            raise ValueError("Provide either `f` (preferred) or `expr`.")
        f_obs = np.asarray(expr, dtype=float).ravel() > 0
    else:
        f_obs = np.asarray(f, dtype=bool).ravel()

    ang_input = angles if angles is not None else theta
    if ang_input is None:
        raise ValueError("`angles` (or legacy alias `theta`) is required.")
    ang_arr = np.asarray(ang_input, dtype=float).ravel()

    if f_obs.size == 0:
        raise ValueError("Foreground vector must contain at least one value.")
    if f_obs.size != ang_arr.size:
        raise ValueError("Foreground vector and angles must have the same length.")
    return f_obs, ang_arr


def _prepare_donor_permutation(
    *,
    f_obs: np.ndarray,
    donor_ids: np.ndarray | None,
    donor_stratified: bool,
) -> tuple[bool, dict[str, np.ndarray] | None, tuple[int, ...], str | None]:
    n_cells = int(f_obs.size)
    used_donor_stratified = False
    donor_to_idx: dict[str, np.ndarray] | None = None
    warning_msg: str | None = None

    if donor_stratified:
        if donor_ids is None:
            warning_msg = (
                "donor_stratified=True but donor_ids are missing; "
                "falling back to global permutation."
            )
        else:
            donor_arr = np.asarray(donor_ids)
            if donor_arr.size != n_cells:
                raise ValueError("donor_ids length must match foreground length.")
            donor_labels = np.unique(donor_arr)
            if donor_labels.size < 2:
                warning_msg = (
                    "donor_stratified=True but <2 unique donors detected; "
                    "falling back to global permutation."
                )
            else:
                donor_to_idx = {
                    str(d): np.flatnonzero(donor_arr == d).astype(int)
                    for d in donor_labels
                }
                used_donor_stratified = True

    if used_donor_stratified and donor_to_idx is not None:
        signature = tuple(
            int(np.asarray(f_obs[idx], dtype=bool).sum())
            for idx in donor_to_idx.values()
        )
    else:
        signature = (int(np.sum(f_obs)),)

    return used_donor_stratified, donor_to_idx, signature, warning_msg


def _signature_payload(count_signature: tuple[int, ...]) -> str:
    payload = ",".join(str(x) for x in count_signature).encode("ascii")
    signature_hash = hashlib.blake2b(payload, digest_size=8).hexdigest()
    return f"{signature_hash}:{count_signature}"


def _apply_mode(E_phi: np.ndarray, mode: str, smooth_w: int) -> np.ndarray:
    if mode == "raw":
        return np.asarray(E_phi, dtype=float)
    return circular_moving_average(np.asarray(E_phi, dtype=float), int(smooth_w))


def _apply_mode_matrix(
    E_phi_matrix: np.ndarray, mode: str, smooth_w: int
) -> np.ndarray:
    arr = np.asarray(E_phi_matrix, dtype=float)
    if arr.ndim != 2:
        raise ValueError("E_phi_matrix must be 2D with shape (n_perm, n_bins).")
    if mode == "raw":
        return arr
    out = np.empty_like(arr, dtype=float)
    for i in range(arr.shape[0]):
        out[i, :] = circular_moving_average(arr[i, :], int(smooth_w))
    return out


def mode_max_stat_from_profiles(
    E_obs_raw: np.ndarray,
    null_E_raw: np.ndarray,
    mode: str = "raw",
    smooth_w: int = 1,
) -> dict[str, Any]:
    """Compute mode-aware max-stat summaries from observed/null RSP profiles."""
    mode_norm, smooth_w_eff = _normalize_mode(mode, smooth_w)
    e_obs = _apply_mode(np.asarray(E_obs_raw, dtype=float), mode_norm, smooth_w_eff)
    null_e = _apply_mode_matrix(
        np.asarray(null_E_raw, dtype=float), mode_norm, smooth_w_eff
    )
    t_obs = compute_T(e_obs)
    null_t = np.max(np.abs(null_e), axis=1)
    n_perm = int(null_t.size)
    p_t = float((1.0 + np.sum(null_t >= t_obs)) / (1.0 + n_perm))
    return {
        "E_phi_obs": e_obs,
        "null_E_phi": null_e,
        "T_obs": float(t_obs),
        "null_T": null_t,
        "p_T": p_t,
        "mode": mode_norm,
        "smooth_w": int(smooth_w_eff),
        "n_perm_used": n_perm,
    }


def _perm_null_core(
    *,
    f: np.ndarray | None = None,
    expr: np.ndarray | None = None,
    angles: np.ndarray | None = None,
    theta: np.ndarray | None = None,
    donor_ids: np.ndarray | None = None,
    n_bins: int = 36,
    n_perm: int = 300,
    seed: int = 0,
    donor_stratified: bool = True,
    mode: str = "raw",
    smooth_w: int = 1,
    perm_indices: np.ndarray | None = None,
    perm_start: int = 0,
    perm_end: int | None = None,
    bin_id: np.ndarray | None = None,
    bin_counts_total: np.ndarray | None = None,
    return_null_T: bool = True,
    return_obs_profile: bool = False,
    return_null_profiles: bool = False,
    validate_stratified_counts: bool = False,
    validate_perm_checks: int = 5,
) -> dict[str, Any]:
    mode_norm, smooth_w_eff = _normalize_mode(mode, smooth_w)
    f_obs, ang_arr = _resolve_fg_and_angles(f=f, expr=expr, angles=angles, theta=theta)

    if int(n_bins) <= 0:
        raise ValueError("n_bins must be a positive integer.")
    if int(n_perm) <= 0:
        raise ValueError("n_perm must be a positive integer.")
    n_bins_int = int(n_bins)
    n_perm_int = int(n_perm)
    n_cells = int(f_obs.size)
    n_fg_total = int(np.sum(f_obs))

    start_i = int(perm_start)
    end_i = int(start_i + n_perm_int if perm_end is None else perm_end)
    if start_i < 0 or end_i < start_i:
        raise ValueError(
            "Invalid permutation slice: require 0 <= perm_start <= perm_end."
        )

    used_donor_stratified, donor_to_idx, fg_sig_tuple, warning_msg = (
        _prepare_donor_permutation(
            f_obs=f_obs,
            donor_ids=donor_ids,
            donor_stratified=bool(donor_stratified),
        )
    )
    if warning_msg is not None:
        warnings.warn(warning_msg, RuntimeWarning, stacklevel=2)

    sig_payload = _signature_payload(fg_sig_tuple)

    n_draw = max(0, end_i - start_i)
    if n_fg_total == 0 or n_fg_total == n_cells:
        zeros_profile = np.zeros(n_bins_int, dtype=float)
        zeros_t = np.zeros(n_draw, dtype=float)
        zeros_e = np.zeros((n_draw, n_bins_int), dtype=float)
        out: dict[str, Any] = {
            "T_obs": 0.0,
            "p_T": 1.0,
            "mode": mode_norm,
            "smooth_w": int(smooth_w_eff),
            "n_bins": n_bins_int,
            "n_perm": n_perm_int,
            "n_perm_used": n_draw,
            "donor_stratified": bool(used_donor_stratified),
            "used_donor_stratified": bool(used_donor_stratified),
            "stratified_counts_signature": fg_sig_tuple,
            "stratified_counts_signature_hash": sig_payload,
            "warning": (
                "Degenerate foreground (all/none cells); returning zero-valued "
                "null profile/statistics."
            ),
        }
        if return_null_T:
            out["null_T"] = zeros_t
        if return_obs_profile:
            out["E_phi_obs"] = zeros_profile
        if return_null_profiles:
            out["null_E_phi"] = zeros_e
        return out

    E_phi_obs_raw, _, _ = compute_rsp_profile_from_boolean(
        f_obs,
        ang_arr,
        n_bins_int,
        bin_id=bin_id,
        bin_counts_total=bin_counts_total,
    )
    E_phi_obs = _apply_mode(E_phi_obs_raw, mode_norm, smooth_w_eff)
    T_obs = compute_T(E_phi_obs)

    rng = np.random.default_rng(int(seed))

    if perm_indices is None:
        perm_indices_arr = None
        n_draw = n_perm_int
    else:
        perm_indices_arr = np.asarray(perm_indices, dtype=np.int32)
        if perm_indices_arr.ndim != 2 or perm_indices_arr.shape[1] != n_cells:
            raise ValueError(
                "perm_indices must have shape (n_perm_available, n_cells)."
            )
        if end_i > perm_indices_arr.shape[0]:
            raise ValueError("perm_end exceeds available precomputed permutation rows.")
        n_draw = max(0, end_i - start_i)

    null_t_all = np.zeros(n_draw, dtype=float)
    null_e = (
        np.zeros((n_draw, n_bins_int), dtype=float) if return_null_profiles else None
    )

    expected_donor_fg: dict[str, int] | None = None
    debug_budget = int(max(0, validate_perm_checks))
    debug_checked = 0
    if used_donor_stratified and donor_to_idx is not None:
        expected_donor_fg = {
            k: int(np.sum(f_obs[idx])) for k, idx in donor_to_idx.items()
        }

    for i in range(n_draw):
        if perm_indices_arr is not None:
            row_idx = start_i + i
            f_perm = f_obs[perm_indices_arr[row_idx]]
        elif used_donor_stratified and donor_to_idx is not None:
            f_perm = permute_foreground_within_donor(f_obs, donor_to_idx, rng)
        else:
            f_perm = rng.permutation(f_obs)

        if (
            validate_stratified_counts
            and expected_donor_fg is not None
            and debug_checked < debug_budget
            and donor_to_idx is not None
        ):
            for donor_key, idx in donor_to_idx.items():
                if int(np.sum(f_perm[idx])) != int(expected_donor_fg[donor_key]):
                    raise RuntimeError(
                        f"Stratified count mismatch for donor '{donor_key}' at perm {i}."
                    )
            debug_checked += 1

        E_phi_perm_raw, _, _ = compute_rsp_profile_from_boolean(
            f_perm,
            ang_arr,
            n_bins_int,
            bin_id=bin_id,
            bin_counts_total=bin_counts_total,
        )
        E_phi_perm = _apply_mode(E_phi_perm_raw, mode_norm, smooth_w_eff)
        null_t_all[i] = compute_T(E_phi_perm)
        if null_e is not None:
            null_e[i, :] = E_phi_perm

    p_T = float((1.0 + np.sum(null_t_all >= T_obs)) / (1.0 + n_draw))
    out = {
        "T_obs": float(T_obs),
        "p_T": p_T,
        "mode": mode_norm,
        "smooth_w": int(smooth_w_eff),
        "n_bins": n_bins_int,
        "n_perm": n_perm_int,
        "n_perm_used": int(n_draw),
        "donor_stratified": bool(used_donor_stratified),
        "used_donor_stratified": bool(used_donor_stratified),
        "stratified_counts_signature": fg_sig_tuple,
        "stratified_counts_signature_hash": sig_payload,
    }
    if warning_msg is not None:
        out["warning"] = warning_msg
    if return_null_T:
        out["null_T"] = null_t_all
    if return_obs_profile:
        out["E_phi_obs"] = E_phi_obs
    if return_null_profiles:
        out["null_E_phi"] = (
            null_e if null_e is not None else np.zeros((0, n_bins_int), dtype=float)
        )
    return out


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
    """Permutation null for ``T = max(abs(E(theta)))`` with mode-consistent smoothing.

    If ``mode='smoothed'``, smoothing is applied to observed and permuted profiles
    before computing ``T``. P-values use plus-one correction.
    """

    return _perm_null_core(
        f=f,
        angles=angles,
        donor_ids=donor_ids,
        n_bins=n_bins,
        n_perm=n_perm,
        seed=seed,
        donor_stratified=donor_stratified,
        mode=mode,
        smooth_w=smooth_w,
        return_null_T=return_null_T,
        return_obs_profile=return_obs_profile,
        return_null_profiles=return_null_profiles,
        validate_stratified_counts=bool(debug),
        validate_perm_checks=int(debug_checks),
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
    """Verify that raw mode equals smoothed mode with ``smooth_w=1``."""
    raw = perm_null_T(
        f=f,
        angles=angles,
        donor_ids=donor_ids,
        n_bins=n_bins,
        n_perm=n_perm,
        seed=seed,
        mode="raw",
        smooth_w=1,
        donor_stratified=donor_stratified,
        return_null_T=True,
        return_obs_profile=True,
    )
    sm1 = perm_null_T(
        f=f,
        angles=angles,
        donor_ids=donor_ids,
        n_bins=n_bins,
        n_perm=n_perm,
        seed=seed,
        mode="smoothed",
        smooth_w=1,
        donor_stratified=donor_stratified,
        return_null_T=True,
        return_obs_profile=True,
    )
    t_diff = float(abs(float(raw["T_obs"]) - float(sm1["T_obs"])))
    p_diff = float(abs(float(raw["p_T"]) - float(sm1["p_T"])))
    if t_diff >= 1e-10:
        raise AssertionError(f"Mode inconsistency: |T_raw - T_sm1|={t_diff:.3e}.")
    if not np.array_equal(np.asarray(raw["null_T"]), np.asarray(sm1["null_T"])):
        raise AssertionError(
            "Mode inconsistency: null_T differs between raw and smoothed(w=1)."
        )
    if p_diff > 0.0:
        raise AssertionError(
            f"Mode inconsistency: p_raw != p_sm1 (|diff|={p_diff:.3e})."
        )
    return {
        "T_raw": float(raw["T_obs"]),
        "T_smoothed_w1": float(sm1["T_obs"]),
        "p_raw": float(raw["p_T"]),
        "p_smoothed_w1": float(sm1["p_T"]),
        "T_abs_diff": t_diff,
        "p_abs_diff": p_diff,
    }


def perm_null_T_and_profile(
    f: np.ndarray | None = None,
    angles: np.ndarray | None = None,
    donor_ids: np.ndarray | None = None,
    n_bins: int = 36,
    n_perm: int = 300,
    seed: int = 0,
    donor_stratified: bool = True,
    *,
    expr: np.ndarray | None = None,
    theta: np.ndarray | None = None,
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
    """Compatibility API for profile-returning permutation tests.

    This path delegates to the same core as :func:`perm_null_T` to keep
    statistic/p-value logic mode-consistent across all callers.
    """

    if previous_null_T is not None or previous_null_E_phi is not None:
        warnings.warn(
            "previous_null_* inputs are ignored; null-distribution reuse is disabled "
            "to preserve gene-conditional validity.",
            RuntimeWarning,
            stacklevel=2,
        )

    if donor_ids is None and donor_stratified:
        donor_stratified = False

    out = _perm_null_core(
        f=f,
        expr=expr,
        angles=angles,
        theta=theta,
        donor_ids=donor_ids,
        n_bins=n_bins,
        n_perm=n_perm,
        seed=seed,
        donor_stratified=donor_stratified,
        mode=mode,
        smooth_w=smooth_w,
        perm_indices=perm_indices,
        perm_start=perm_start,
        perm_end=perm_end,
        bin_id=bin_id,
        bin_counts_total=bin_counts_total,
        return_null_T=True,
        return_obs_profile=True,
        return_null_profiles=return_null_profiles,
        validate_stratified_counts=validate_stratified_counts,
        validate_perm_checks=validate_perm_checks,
    )

    if "null_T" not in out:
        out["null_T"] = np.zeros(0, dtype=float)
    if "E_phi_obs" not in out:
        out["E_phi_obs"] = np.zeros(int(n_bins), dtype=float)
    if "null_E_phi" not in out:
        out["null_E_phi"] = np.zeros((0, int(n_bins)), dtype=float)
    return out


def plot_null_distribution(
    null_emax: np.ndarray, observed_emax: float, out_png: str, title: str | None = None
) -> None:
    """Plot null E_max distribution with observed value line.

    Args:
        null_emax: Array of null E_max values.
        observed_emax: Observed E_max value.
        out_png: Output file path.
        title: Optional plot title.
    """
    arr = np.asarray(null_emax, dtype=float).ravel()
    import os

    out_dir = os.path.dirname(out_png) or "."
    ensure_dir(out_dir)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(arr, bins=20, color="steelblue", alpha=0.7, edgecolor="black")
    ax.axvline(
        observed_emax, color="red", linestyle="--", linewidth=2, label="Observed E_max"
    )
    ax.set_xlabel("E_max")
    ax.set_ylabel("Count")
    if title:
        ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
