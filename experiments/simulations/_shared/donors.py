"""Donor assignment and donor-effect helpers for simulations."""

from __future__ import annotations

import numpy as np


def assign_donors(
    N: int, D: int, rng: np.random.Generator, mode: str = "even"
) -> np.ndarray:
    """Assign donors approximately evenly, then shuffle deterministically by RNG."""
    n = int(N)
    d = int(D)
    if n <= 0 or d <= 0:
        raise ValueError("N and D must be positive.")
    if str(mode) != "even":
        raise ValueError(f"Unsupported donor assignment mode: {mode}")
    base = n // d
    rem = n % d
    counts = np.full(d, base, dtype=int)
    counts[:rem] += 1
    donor_ids = np.repeat(np.arange(d, dtype=np.int16), counts)
    rng.shuffle(donor_ids)
    return donor_ids


def sample_donor_effects(
    D: int, sigma_eta: float, rng: np.random.Generator
) -> np.ndarray:
    """Sample donor random effects on the logit scale."""
    return rng.normal(loc=0.0, scale=float(sigma_eta), size=int(D)).astype(float)


def simulate_donor_offsets(
    D: int, sigma_eta: float, rng: np.random.Generator
) -> np.ndarray:
    """Alias for donor offset simulation with unified naming."""
    return sample_donor_effects(int(D), float(sigma_eta), rng)


def donor_effect_vector(donor_ids: np.ndarray, eta_d: np.ndarray) -> np.ndarray:
    """Broadcast donor-level effects to per-cell vector."""
    donor_arr = np.asarray(donor_ids, dtype=int).ravel()
    eta_arr = np.asarray(eta_d, dtype=float).ravel()
    if donor_arr.size == 0:
        return np.zeros(0, dtype=float)
    if np.any(donor_arr < 0) or np.any(donor_arr >= eta_arr.size):
        raise ValueError("donor_ids contain out-of-range donor indices.")
    return eta_arr[donor_arr]


def donor_stratified_shuffle_indices(
    donor_ids: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Return permutation indices that shuffle cells within donor blocks."""
    donor_arr = np.asarray(donor_ids, dtype=int).ravel()
    idx = np.arange(donor_arr.size, dtype=int)
    out = idx.copy()
    for d in np.unique(donor_arr):
        ii = idx[donor_arr == int(d)]
        if ii.size <= 1:
            continue
        out[ii] = ii[rng.permutation(ii.size)]
    return out


def effective_donors(n_fg_by_donor: np.ndarray, min_fg: int = 10) -> int:
    """Count donors with foreground support above threshold."""
    fg = np.asarray(n_fg_by_donor, dtype=float).ravel()
    return int(np.sum(fg >= int(min_fg)))
