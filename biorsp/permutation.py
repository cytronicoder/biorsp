"""Donor-stratified permutation tests for RSP."""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np

from biorsp.rsp import compute_rsp_profile_from_boolean
from biorsp.scoring import compute_T
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


def perm_null_T_and_profile(
    expr: np.ndarray,
    angles: np.ndarray,
    donor_ids: np.ndarray | None,
    n_bins: int,
    n_perm: int,
    seed: int = 0,
    donor_stratified: bool = True,
    *,
    perm_indices: np.ndarray | None = None,
    perm_start: int = 0,
    perm_end: int | None = None,
    previous_null_T: np.ndarray | None = None,
    previous_null_E_phi: np.ndarray | None = None,
    bin_id: np.ndarray | None = None,
    bin_counts_total: np.ndarray | None = None,
) -> dict[str, np.ndarray | float | bool | str]:
    """Permutation nulls for max-absolute RSP anisotropy and full RSP profiles.

    Foreground is deterministically defined as ``expr > 0``.

    Args:
        expr: Expression vector, shape ``(n_cells,)``.
        angles: Angular coordinates (radians), shape ``(n_cells,)``.
        donor_ids: Donor IDs per cell, optional.
        n_bins: Number of angular bins for RSP.
        n_perm: Number of permutations.
        seed: Random seed.
        donor_stratified: If True, attempt donor-stratified permutation.

    Returns:
        Dictionary containing:
        - ``null_T``: Null distribution of ``max(abs(E_phi))`` with shape ``(n_perm,)``.
        - ``null_E_phi``: Null RSP profiles with shape ``(n_perm, n_bins)``.
        - ``E_phi_obs``: Observed RSP profile, shape ``(n_bins,)``.
        - ``T_obs``: Observed anisotropy statistic.
        - ``p_T``: Permutation p-value.
        - ``used_donor_stratified``: Whether donor-stratified shuffling was used.
        - ``warning``: Optional warning message when global fallback is used.
    """
    expr_arr = np.asarray(expr, dtype=float).ravel()
    ang_arr = np.asarray(angles, dtype=float).ravel()
    if expr_arr.size == 0:
        raise ValueError("expr must contain at least one value.")
    if expr_arr.size != ang_arr.size:
        raise ValueError("expr and angles must have the same length.")
    if int(n_bins) <= 0:
        raise ValueError("n_bins must be a positive integer.")
    if int(n_perm) <= 0:
        raise ValueError("n_perm must be a positive integer.")

    n_bins_int = int(n_bins)
    n_perm_int = int(n_perm)
    f_obs = expr_arr > 0
    n_fg = int(f_obs.sum())
    n_cells = int(f_obs.size)
    start_i = int(perm_start)
    end_i = int(start_i + n_perm_int if perm_end is None else perm_end)
    if start_i < 0 or end_i < start_i:
        raise ValueError("Invalid permutation slice: require 0 <= perm_start <= perm_end.")

    warning_msg: str | None = None
    used_donor_stratified = False
    donor_to_idx: dict[str, np.ndarray] | None = None

    prev_null_T = (
        np.asarray(previous_null_T, dtype=float).ravel()
        if previous_null_T is not None
        else np.zeros(0, dtype=float)
    )
    if previous_null_E_phi is None:
        prev_null_E = np.zeros((0, n_bins_int), dtype=float)
    else:
        prev_null_E = np.asarray(previous_null_E_phi, dtype=float)
        if prev_null_E.ndim != 2 or prev_null_E.shape[1] != n_bins_int:
            raise ValueError("previous_null_E_phi must have shape (n_prev, n_bins).")
    if prev_null_E.shape[0] != prev_null_T.size:
        raise ValueError("previous_null_T and previous_null_E_phi length mismatch.")

    if donor_stratified:
        if perm_indices is not None:
            used_donor_stratified = True
        else:
            if donor_ids is None:
                warning_msg = (
                    "donor_stratified=True but donor_ids are missing; "
                    "falling back to global permutation."
                )
            else:
                donor_ids_arr = np.asarray(donor_ids)
                if donor_ids_arr.size != f_obs.size:
                    raise ValueError("donor_ids length must match expression vector length.")
                unique_donors = np.unique(donor_ids_arr)
                if unique_donors.size < 2:
                    warning_msg = (
                        "donor_stratified=True but <2 unique donors detected; "
                        "falling back to global permutation."
                    )
                else:
                    donor_to_idx = {
                        str(d): np.nonzero(donor_ids_arr == d)[0].astype(int)
                        for d in unique_donors
                    }
                    used_donor_stratified = True

    if warning_msg is not None:
        warnings.warn(warning_msg, RuntimeWarning, stacklevel=2)

    if perm_indices is not None:
        perm_indices_arr = np.asarray(perm_indices, dtype=np.int32)
        if perm_indices_arr.ndim != 2 or perm_indices_arr.shape[1] != n_cells:
            raise ValueError("perm_indices must have shape (n_perm_available, n_cells).")
        if end_i > perm_indices_arr.shape[0]:
            raise ValueError("perm_end exceeds available precomputed permutation rows.")
    else:
        perm_indices_arr = None

    if n_fg == 0 or n_fg == n_cells:
        zeros_profile = np.zeros(n_bins_int, dtype=float)
        n_new = max(0, end_i - start_i)
        zeros_null = np.zeros((prev_null_E.shape[0] + n_new, n_bins_int), dtype=float)
        zeros_t = np.zeros(prev_null_T.size + n_new, dtype=float)
        if prev_null_E.shape[0] > 0:
            zeros_null[: prev_null_E.shape[0], :] = prev_null_E
            zeros_t[: prev_null_T.size] = prev_null_T
        return {
            "null_T": zeros_t,
            "null_E_phi": zeros_null,
            "E_phi_obs": zeros_profile,
            "T_obs": 0.0,
            "p_T": 1.0,
            "used_donor_stratified": bool(used_donor_stratified),
            "n_perm_used": int(zeros_t.size),
            "warning": (
                "Degenerate foreground (all/none cells in expr>0); "
                "returning zero-valued null profile/statistics."
            ),
        }

    E_phi_obs, _, _ = compute_rsp_profile_from_boolean(
        f_obs,
        ang_arr,
        n_bins_int,
        bin_id=bin_id,
        bin_counts_total=bin_counts_total,
    )
    T_obs = compute_T(E_phi_obs)

    rng = np.random.default_rng(seed)
    n_new = max(0, end_i - start_i)
    new_null_E = np.zeros((n_new, n_bins_int), dtype=float)
    new_null_T = np.zeros(n_new, dtype=float)
    for row_offset, i in enumerate(range(start_i, end_i)):
        if perm_indices_arr is not None:
            f_perm = f_obs[perm_indices_arr[i]]
        elif used_donor_stratified and donor_to_idx is not None:
            f_perm = permute_foreground_within_donor(f_obs, donor_to_idx, rng)
        else:
            f_perm = rng.permutation(f_obs)
        E_phi_perm, _, _ = compute_rsp_profile_from_boolean(
            f_perm,
            ang_arr,
            n_bins_int,
            bin_id=bin_id,
            bin_counts_total=bin_counts_total,
        )
        new_null_E[row_offset, :] = E_phi_perm
        new_null_T[row_offset] = compute_T(E_phi_perm)

    if prev_null_T.size > 0:
        null_T = np.concatenate([prev_null_T, new_null_T])
        null_E_phi = np.vstack([prev_null_E, new_null_E]) if new_null_E.size else prev_null_E
    else:
        null_T = new_null_T
        null_E_phi = new_null_E

    p_T = float((1.0 + np.sum(null_T >= T_obs)) / (1.0 + null_T.size))
    out: dict[str, np.ndarray | float | bool | str] = {
        "null_T": null_T,
        "null_E_phi": null_E_phi,
        "E_phi_obs": E_phi_obs,
        "T_obs": float(T_obs),
        "p_T": p_T,
        "used_donor_stratified": bool(used_donor_stratified),
        "n_perm_used": int(null_T.size),
    }
    if warning_msg is not None:
        out["warning"] = warning_msg
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
