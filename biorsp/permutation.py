"""Donor-stratified permutation tests for RSP."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from biorsp.rsp import compute_rsp_profile_from_boolean
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
