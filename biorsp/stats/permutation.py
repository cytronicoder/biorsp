"""Permutation null models for BioRSP statistics."""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np

from biorsp.core.compute import compute_rsp_profile_from_boolean


def _compute_t(r_theta: np.ndarray) -> float:
    arr = np.asarray(r_theta, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError("r_theta must be non-empty.")
    return float(np.max(np.abs(arr)))


def build_donor_index(adata, donor_col: str = "donor") -> dict[str, np.ndarray]:
    if donor_col not in adata.obs:
        raise KeyError(
            f"adata.obs['{donor_col}'] is required (missing donor labels risks pseudoreplication)."
        )
    donor_ids = np.asarray(adata.obs[donor_col])
    uniq = np.unique(donor_ids)
    if uniq.size < 2:
        raise ValueError("At least two donors are required for donor-stratified permutation.")
    return {str(d): np.flatnonzero(donor_ids == d).astype(int) for d in uniq}


def permute_foreground_within_donor(
    foreground: np.ndarray,
    donor_to_idx: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    f = np.asarray(foreground, dtype=bool).ravel()
    out = np.zeros(f.size, dtype=bool)
    for idx in donor_to_idx.values():
        idx_arr = np.asarray(idx, dtype=int)
        if idx_arr.size == 0:
            continue
        k = int(f[idx_arr].sum())
        if k <= 0:
            continue
        if k > idx_arr.size:
            raise ValueError("Foreground count exceeds donor block size.")
        chosen = rng.choice(idx_arr, size=k, replace=False)
        out[chosen] = True
    return out


def perm_null_emax(
    expr: np.ndarray,
    theta: np.ndarray,
    donor_ids: np.ndarray,
    n_bins: int,
    n_perm: int,
    seed: int = 0,
) -> tuple[np.ndarray, float, float, float]:
    f_obs = np.asarray(expr, dtype=float).ravel() > 0.0
    donor_arr = np.asarray(donor_ids)
    if donor_arr.size != f_obs.size:
        raise ValueError("donor_ids length must match expression length.")

    uniq = np.unique(donor_arr)
    if uniq.size < 2:
        raise ValueError("At least two donors are required for permutation testing.")
    donor_map = {str(d): np.flatnonzero(donor_arr == d).astype(int) for d in uniq}

    r_obs, phi_max, e_max_obs, _ = compute_rsp_profile_from_boolean(f_obs, theta, n_bins)
    _ = r_obs

    rng = np.random.default_rng(int(seed))
    null_e = np.zeros(int(n_perm), dtype=float)
    for i in range(int(n_perm)):
        f_perm = permute_foreground_within_donor(f_obs, donor_map, rng)
        _, _, e_perm, _ = compute_rsp_profile_from_boolean(f_perm, theta, n_bins)
        null_e[i] = float(e_perm)

    p = float((1.0 + np.sum(null_e >= e_max_obs)) / (1.0 + null_e.size))
    return null_e, float(e_max_obs), float(phi_max), p


def perm_null_T_and_profile(
    expr: np.ndarray,
    theta: np.ndarray,
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
    x = np.asarray(expr, dtype=float).ravel()
    th = np.asarray(theta, dtype=float).ravel()
    if x.size == 0 or x.size != th.size:
        raise ValueError("expr/theta must be same non-zero length.")

    n_bins_i = int(n_bins)
    n_perm_i = int(n_perm)
    if n_bins_i <= 0 or n_perm_i <= 0:
        raise ValueError("n_bins and n_perm must be positive.")

    f_obs = x > 0.0
    n_fg = int(f_obs.sum())
    n_cells = int(f_obs.size)

    start_i = int(perm_start)
    end_i = int(start_i + n_perm_i if perm_end is None else perm_end)
    if start_i < 0 or end_i < start_i:
        raise ValueError("Invalid permutation slice.")

    prev_t = (
        np.asarray(previous_null_T, dtype=float).ravel()
        if previous_null_T is not None
        else np.zeros(0, dtype=float)
    )
    if previous_null_E_phi is None:
        prev_e = np.zeros((0, n_bins_i), dtype=float)
    else:
        prev_e = np.asarray(previous_null_E_phi, dtype=float)
        if prev_e.ndim != 2 or prev_e.shape[1] != n_bins_i:
            raise ValueError("previous_null_E_phi shape mismatch.")
    if prev_e.shape[0] != prev_t.size:
        raise ValueError("previous null length mismatch.")

    used_donor_stratified = False
    warning_msg: str | None = None
    donor_to_idx: dict[str, np.ndarray] | None = None

    if donor_stratified:
        if perm_indices is not None:
            used_donor_stratified = True
        elif donor_ids is None:
            warning_msg = (
                "donor_stratified=True but donor_ids missing; falling back to global permutation."
            )
        else:
            donor_arr = np.asarray(donor_ids)
            if donor_arr.size != n_cells:
                raise ValueError("donor_ids length mismatch.")
            uniq = np.unique(donor_arr)
            if uniq.size < 2:
                warning_msg = (
                    "donor_stratified=True but <2 donors detected; falling back to global permutation."
                )
            else:
                donor_to_idx = {
                    str(d): np.flatnonzero(donor_arr == d).astype(int) for d in uniq
                }
                used_donor_stratified = True

    if warning_msg is not None:
        warnings.warn(warning_msg, RuntimeWarning, stacklevel=2)

    if n_fg == 0 or n_fg == n_cells:
        zeros_profile = np.zeros(n_bins_i, dtype=float)
        n_new = max(0, end_i - start_i)
        out_t = np.concatenate([prev_t, np.zeros(n_new, dtype=float)])
        out_e = (
            np.vstack([prev_e, np.zeros((n_new, n_bins_i), dtype=float)])
            if prev_e.size > 0
            else np.zeros((n_new, n_bins_i), dtype=float)
        )
        return {
            "null_T": out_t,
            "null_E_phi": out_e,
            "E_phi_obs": zeros_profile,
            "T_obs": 0.0,
            "p_T": 1.0,
            "used_donor_stratified": bool(used_donor_stratified),
            "n_perm_used": int(out_t.size),
            "warning": "Degenerate foreground (all/none cells).",
        }

    e_obs, _, _, _ = compute_rsp_profile_from_boolean(
        f_obs,
        th,
        n_bins_i,
        bin_id=bin_id,
        bin_counts_total=bin_counts_total,
    )
    t_obs = _compute_t(e_obs)

    rng = np.random.default_rng(int(seed))
    n_new = max(0, end_i - start_i)
    new_e = np.zeros((n_new, n_bins_i), dtype=float)
    new_t = np.zeros(n_new, dtype=float)

    perm_arr: np.ndarray | None
    if perm_indices is None:
        perm_arr = None
    else:
        perm_arr = np.asarray(perm_indices, dtype=np.int32)
        if perm_arr.ndim != 2 or perm_arr.shape[1] != n_cells:
            raise ValueError("perm_indices must have shape (n_perm_available, n_cells).")
        if end_i > perm_arr.shape[0]:
            raise ValueError("perm_end exceeds available permutation rows.")

    for row_off, i in enumerate(range(start_i, end_i)):
        if perm_arr is not None:
            f_perm = f_obs[perm_arr[i]]
        elif used_donor_stratified and donor_to_idx is not None:
            f_perm = permute_foreground_within_donor(f_obs, donor_to_idx, rng)
        else:
            f_perm = rng.permutation(f_obs)

        e_perm, _, _, _ = compute_rsp_profile_from_boolean(
            f_perm,
            th,
            n_bins_i,
            bin_id=bin_id,
            bin_counts_total=bin_counts_total,
        )
        new_e[row_off, :] = e_perm
        new_t[row_off] = _compute_t(e_perm)

    null_t = np.concatenate([prev_t, new_t]) if prev_t.size > 0 else new_t
    null_e = np.vstack([prev_e, new_e]) if prev_e.size > 0 else new_e
    p_t = float((1.0 + np.sum(null_t >= t_obs)) / (1.0 + null_t.size))

    out: dict[str, np.ndarray | float | bool | str] = {
        "null_T": null_t,
        "null_E_phi": null_e,
        "E_phi_obs": e_obs,
        "T_obs": float(t_obs),
        "p_T": p_t,
        "used_donor_stratified": bool(used_donor_stratified),
        "n_perm_used": int(null_t.size),
    }
    if warning_msg is not None:
        out["warning"] = warning_msg
    return out


def plot_null_distribution(
    null_emax: np.ndarray,
    observed_emax: float,
    out_png: str,
    title: str | None = None,
) -> None:
    arr = np.asarray(null_emax, dtype=float).ravel()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(arr, bins=20, color="steelblue", alpha=0.7, edgecolor="black")
    ax.axvline(observed_emax, color="red", linestyle="--", linewidth=2)
    ax.set_xlabel("E_max")
    ax.set_ylabel("Count")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
