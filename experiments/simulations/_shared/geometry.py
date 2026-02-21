"""Standard geometry samplers used across simulation experiments."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import expit


def sample_disk_gaussian(
    N: int, rng: np.random.Generator
) -> tuple[np.ndarray, dict[str, Any]]:
    """Sample isotropic Gaussian disk points in 2D."""
    X = rng.normal(loc=0.0, scale=1.0, size=(int(N), 2)).astype(float)
    return X, {"geometry": "disk_gaussian", "mean": 0.0, "std": 1.0}


def sample_ring_annulus(
    N: int,
    rng: np.random.Generator,
    r0: float = 2.0,
    sigma_r: float = 0.3,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Sample a noisy ring/annulus in polar coordinates."""
    n = int(N)
    u = rng.uniform(0.0, 2.0 * np.pi, size=n)
    r = np.clip(rng.normal(loc=float(r0), scale=float(sigma_r), size=n), 0.05, None)
    X = np.column_stack([r * np.cos(u), r * np.sin(u)]).astype(float)
    return X, {"geometry": "ring_annulus", "r0": float(r0), "sigma_r": float(sigma_r)}


def sample_density_gradient_disk(
    N: int,
    rng: np.random.Generator,
    k: float = 1.5,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Sample Gaussian points with x-gradient density via accept/reject."""
    n = int(N)
    out = np.zeros((n, 2), dtype=float)
    filled = 0
    k_val = float(k)
    while filled < n:
        need = n - filled
        batch = max(2 * need, 512)
        cand = rng.normal(loc=0.0, scale=1.0, size=(batch, 2)).astype(float)
        p_accept = expit(k_val * cand[:, 0])
        keep = cand[rng.random(batch) < p_accept]
        if keep.size == 0:
            continue
        take = min(need, keep.shape[0])
        out[filled : filled + take, :] = keep[:take, :]
        filled += take
    return out, {"geometry": "density_gradient_disk", "k": k_val}


def sample_two_islands(
    N: int,
    rng: np.random.Generator,
    sep: float = 4.0,
    sigma: float = 0.8,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Sample two Gaussian islands separated on x-axis."""
    n = int(N)
    n1 = n // 2
    n2 = n - n1
    c1 = np.array([-float(sep) / 2.0, 0.0], dtype=float)
    c2 = np.array([float(sep) / 2.0, 0.0], dtype=float)
    x1 = rng.normal(loc=c1, scale=float(sigma), size=(n1, 2))
    x2 = rng.normal(loc=c2, scale=float(sigma), size=(n2, 2))
    X = np.vstack([x1, x2]).astype(float)
    labels = np.concatenate([np.zeros(n1, dtype=int), np.ones(n2, dtype=int)])
    perm = rng.permutation(n)
    X = X[perm]
    labels = labels[perm]
    return X, {
        "geometry": "two_islands",
        "sep": float(sep),
        "sigma": float(sigma),
        "island_labels": labels,
    }


def sample_geometry(
    geometry: str,
    N: int,
    rng: np.random.Generator,
    **kwargs: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Dispatch helper for standard geometry names."""
    if geometry == "disk_gaussian":
        return sample_disk_gaussian(N, rng)
    if geometry == "ring_annulus":
        return sample_ring_annulus(N, rng, **kwargs)
    if geometry == "density_gradient_disk":
        return sample_density_gradient_disk(N, rng, **kwargs)
    if geometry == "two_islands":
        return sample_two_islands(N, rng, **kwargs)
    raise ValueError(f"Unsupported geometry '{geometry}'.")


def make_geometry(
    name: str,
    N: int,
    rng: np.random.Generator,
    **kwargs: Any,
) -> dict[str, Any]:
    """Unified geometry API used by standardized simulation runners."""
    X, meta = sample_geometry(str(name), int(N), rng, **kwargs)
    return {
        "name": str(name),
        "X": np.asarray(X, dtype=float),
        "metadata": dict(meta),
        "default_origin": (0.0, 0.0),
    }
