"""Radial symmetric profile (RSP) computation and plotting."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from biorsp.geometry import validate_angles
from biorsp.utils import ensure_dir


def _bin_angles(angles: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Assign angles to bins and return (bin_edges, bin_idx).

    Args:
        angles: Array of angles in radians.
        n_bins: Number of bins.

    Returns:
        Tuple of (bin_edges, bin_idx).

    Raises:
        ValueError: If n_bins is not positive.
    """
    if n_bins <= 0:
        raise ValueError("n_bins must be a positive integer.")
    ang = validate_angles(angles)
    bin_edges = np.linspace(0, 2 * np.pi, n_bins + 1, endpoint=True)
    bin_idx = np.digitize(ang, bin_edges, right=False) - 1
    # Ensure rightmost edge falls into last bin
    bin_idx = np.where(bin_idx == n_bins, n_bins - 1, bin_idx)
    return bin_edges, bin_idx.astype(int)


def compute_rsp_profile(
    expr: np.ndarray, angles: np.ndarray, n_bins: int
) -> tuple[np.ndarray, float, float]:
    """Compute the radial symmetric profile for a gene expression vector.

    Args:
        expr: Gene expression vector (length N).
        angles: Angles in radians (length N).
        n_bins: Number of bins spanning [0, 2π).

    Returns:
        Tuple of (E_phi array, phi_max, E_max).
    """
    f = np.asarray(expr).ravel() > 0
    return compute_rsp_profile_from_boolean(f, angles, n_bins)


def compute_rsp_profile_from_boolean(
    f: np.ndarray, angles: np.ndarray, n_bins: int
) -> tuple[np.ndarray, float, float]:
    """Compute RSP profile given boolean foreground array."""
    f_bool = np.asarray(f, dtype=bool).ravel()
    ang = validate_angles(angles)
    if f_bool.size != ang.size:
        raise ValueError("Foreground vector and angles must have the same length.")

    nF = int(f_bool.sum())
    nB = f_bool.size - nF
    if nF == 0 or nB == 0:
        raise ValueError(
            "RSP undefined when all/none cells are foreground; adjust threshold or choose another gene."
        )

    bin_edges, bin_idx = _bin_angles(ang, n_bins)

    foreground_counts = np.bincount(bin_idx[f_bool], minlength=n_bins)
    total_counts = np.bincount(bin_idx, minlength=n_bins)
    background_counts = total_counts - foreground_counts

    pF = foreground_counts / nF
    pB = background_counts / nB
    E_phi = pF - pB

    E_max = float(E_phi.max())
    b_max = int(E_phi.argmax())
    phi_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    phi_max = float(phi_centers[b_max] % (2 * np.pi))

    if E_phi.shape[0] != n_bins:
        raise AssertionError("E_phi length mismatch with n_bins.")
    if not (0.0 <= phi_max < 2 * np.pi + 1e-12):
        raise AssertionError("phi_max is outside [0, 2π).")

    return E_phi, phi_max, E_max


def plot_rsp_polar(E_phi: np.ndarray, out_png: str, title: str | None = None) -> None:
    """Save a polar plot of the RSP profile.

    Args:
        E_phi: RSP values per bin.
        out_png: Output file path.
        title: Optional plot title.
    """
    E_arr = np.asarray(E_phi, dtype=float).ravel()
    n_bins = E_arr.size
    bin_edges = np.linspace(0, 2 * np.pi, n_bins + 1, endpoint=True)
    theta = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Close the curve
    theta_closed = np.concatenate([theta, theta[:1]])
    E_closed = np.concatenate([E_arr, E_arr[:1]])

    import os

    out_dir = os.path.dirname(out_png) or "."
    ensure_dir(out_dir)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="polar")
    ax.plot(theta_closed, E_closed, lw=2)
    ax.fill_between(theta_closed, 0, E_closed, alpha=0.2)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(135)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
