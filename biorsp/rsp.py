"""Radial symmetric profile (RSP) computation and plotting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp

from biorsp.geometry import compute_vantage, validate_angles
from biorsp.utils import ensure_dir


def _bin_angles(angles: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Assign angles to bins and return (bin_edges, bin_idx)."""
    if n_bins <= 0:
        raise ValueError("n_bins must be a positive integer.")
    ang = validate_angles(angles)
    bin_edges = np.linspace(0, 2 * np.pi, n_bins + 1, endpoint=True)
    bin_idx = np.digitize(ang, bin_edges, right=False) - 1
    bin_idx = np.where(bin_idx == n_bins, n_bins - 1, bin_idx)
    return bin_edges, bin_idx.astype(int)


def _assert_theta_orientation_debug(tol: float = 1e-8) -> None:
    """Fail loudly if canonical E/N/W/S directions do not map to expected theta values."""
    center = np.array([0.0, 0.0], dtype=float)
    points = np.array(
        [
            [1.0, 0.0],   # East
            [0.0, 1.0],   # North
            [-1.0, 0.0],  # West
            [0.0, -1.0],  # South
        ],
        dtype=float,
    )
    dx = points[:, 0] - center[0]
    dy = points[:, 1] - center[1]
    theta = np.mod(np.arctan2(dy, dx), 2 * np.pi)
    expected = np.array([0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0], dtype=float)
    if not np.allclose(theta, expected, atol=tol, rtol=0.0):
        raise RuntimeError(
            "Theta orientation mismatch in debug sanity check. "
            f"Expected {expected.tolist()}, observed {theta.tolist()}."
        )


def compute_theta_from_embedding(
    embedding_xy: np.ndarray,
    center_xy: np.ndarray,
    debug: bool = False,
) -> np.ndarray:
    """Compute theta from embedding coordinates using math-coordinate convention.

    theta = mod(arctan2(dy, dx), 2*pi), where dx = x - x0 and dy = y - y0.
    """
    if debug:
        _assert_theta_orientation_debug()

    emb = np.asarray(embedding_xy, dtype=float)
    ctr = np.asarray(center_xy, dtype=float).ravel()
    if emb.ndim != 2 or emb.shape[1] < 2:
        raise ValueError(f"Embedding must have shape (N, 2+); received {emb.shape}.")
    if ctr.size != 2:
        raise ValueError(f"center_xy must be length 2; received size={ctr.size}.")
    if not np.isfinite(emb[:, :2]).all():
        raise ValueError("Embedding contains NaN/inf values.")
    if not np.isfinite(ctr).all():
        raise ValueError("center_xy contains NaN/inf values.")

    dx = emb[:, 0] - ctr[0]
    dy = emb[:, 1] - ctr[1]
    theta = np.mod(np.arctan2(dy, dx), 2 * np.pi)
    return validate_angles(theta)


def compute_rsp_profile(
    expr: np.ndarray,
    angles: np.ndarray,
    n_bins: int,
) -> tuple[np.ndarray, float, float]:
    """Compute the radial symmetric profile for a gene expression vector."""
    f = np.asarray(expr).ravel() > 0
    return compute_rsp_profile_from_boolean(f, angles, n_bins)


def compute_rsp_profile_from_boolean(
    f: np.ndarray,
    angles: np.ndarray,
    n_bins: int,
    *,
    bin_id: np.ndarray | None = None,
    bin_counts_total: np.ndarray | None = None,
) -> tuple[np.ndarray, float, float]:
    """Compute RSP profile given boolean foreground array.

    Normalization is unchanged from the legacy implementation:
    - ``pF[b] = foreground_count[b] / n_foreground``
    - ``pB[b] = background_count[b] / n_background``
    - ``E_phi[b] = pF[b] - pB[b]``
    """
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

    if bin_id is None:
        bin_edges, bin_idx = _bin_angles(ang, n_bins)
        total_counts = np.bincount(bin_idx, minlength=n_bins)
    else:
        bin_idx = np.asarray(bin_id, dtype=np.int32).ravel()
        if bin_idx.size != f_bool.size:
            raise ValueError("bin_id must have the same length as foreground vector.")
        if np.any(bin_idx < 0) or np.any(bin_idx >= int(n_bins)):
            raise ValueError("bin_id contains values outside [0, n_bins).")
        bin_edges = np.linspace(0, 2 * np.pi, n_bins + 1, endpoint=True)
        if bin_counts_total is None:
            total_counts = np.bincount(bin_idx, minlength=n_bins)
        else:
            total_counts = np.asarray(bin_counts_total, dtype=float).ravel()
            if total_counts.size != int(n_bins):
                raise ValueError("bin_counts_total length must equal n_bins.")

    foreground_counts = np.bincount(bin_idx[f_bool], minlength=n_bins)
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


def _resolve_expr_matrix(adata: Any, layer: str | None, use_raw: bool) -> tuple[Any, Any]:
    if layer is not None and use_raw:
        raise ValueError("Use either layer or use_raw, not both.")
    if use_raw:
        if getattr(adata, "raw", None) is None:
            raise ValueError("use_raw=True requested but adata.raw is missing.")
        return adata.raw.X, adata.raw.var_names
    if layer is not None:
        if layer not in adata.layers:
            raise ValueError(f"Layer '{layer}' not found in adata.layers.")
        return adata.layers[layer], adata.var_names
    return adata.X, adata.var_names


def _extract_expr_vector(
    expr_matrix: Any,
    var_names: Any,
    gene: str,
    var_index: int | None = None,
) -> tuple[np.ndarray, str]:
    if var_index is not None:
        idx = int(var_index)
        if idx < 0 or idx >= len(var_names):
            raise IndexError(f"var_index {idx} out of bounds for n_vars={len(var_names)}.")
        label = str(gene) if str(gene).strip() != "" else str(var_names[idx])
    else:
        if gene not in var_names:
            raise KeyError(f"Feature '{gene}' not found in selected expression namespace.")
        loc = var_names.get_loc(gene)
        if isinstance(loc, (int, np.integer)):
            idx = int(loc)
        elif isinstance(loc, np.ndarray) and loc.size > 0:
            idx = int(np.flatnonzero(loc)[0])
        else:
            raise KeyError(f"Feature '{gene}' could not be resolved to a unique index.")
        label = str(gene)

    vec = expr_matrix[:, idx]
    if sp.issparse(vec):
        expr = vec.toarray().ravel().astype(float)
    else:
        expr = np.asarray(vec).ravel().astype(float)
    return expr, label


def plot_rsp_polar(
    E_phi: np.ndarray,
    out_png: str,
    title: str | None = None,
    *,
    theta_zero: str = "E",
    theta_direction: int = 1,
    umap_aligned: bool = True,
    line_color: str = "#8B0000",
    fill_color: str = "#F08080",
    fill_alpha: float = 0.30,
    debug: bool = False,
) -> None:
    """Save a polar plot of the RSP profile with UMAP-aligned default orientation."""
    out_dir = Path(out_png).parent.as_posix() or "."
    ensure_dir(out_dir)
    fig, _ = plot_rsp(
        E_phi,
        title=title,
        theta_zero=theta_zero,
        theta_direction=theta_direction,
        umap_aligned=umap_aligned,
        line_color=line_color,
        fill_color=fill_color,
        fill_alpha=fill_alpha,
        debug=debug,
    )
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_rsp(
    E_phi: np.ndarray,
    title: str | None = None,
    *,
    ax: plt.Axes | None = None,
    theta_zero: str = "E",
    theta_direction: int = 1,
    umap_aligned: bool = True,
    line_color: str = "#8B0000",
    fill_color: str = "#F08080",
    fill_alpha: float = 0.30,
    debug: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """Create a polar RSP plot and return ``(figure, axes)``.

    Args:
        E_phi: RSP values per angular bin.
        title: Optional plot title.
        ax: Optional pre-existing polar axis. If None, a new figure/axis is created.
        theta_zero: Polar zero-location passed to ``set_theta_zero_location`` when
            ``umap_aligned=False``.
        theta_direction: Polar direction passed to ``set_theta_direction`` when
            ``umap_aligned=False``. Use ``1`` for CCW or ``-1`` for CW.
        umap_aligned: If True (default), enforce UMAP-aligned orientation:
            0° at East and CCW-increasing theta.
        line_color: Line color for the RSP curve.
        fill_color: Fill color for the RSP area.
        fill_alpha: Fill transparency.
        debug: If True, run orientation sanity assertions and raise on mismatch.
    """
    if debug:
        _assert_theta_orientation_debug()

    E_arr = np.asarray(E_phi, dtype=float).ravel()
    n_bins = E_arr.size
    bin_edges = np.linspace(0, 2 * np.pi, n_bins + 1, endpoint=True)
    theta = (bin_edges[:-1] + bin_edges[1:]) / 2
    theta_closed = np.concatenate([theta, theta[:1]])
    E_closed = np.concatenate([E_arr, E_arr[:1]])

    if ax is None:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="polar")
    else:
        fig = ax.figure
        if ax.name != "polar":
            raise ValueError("Provided ax must be a polar axis.")

    effective_theta_zero = "E" if umap_aligned else theta_zero
    effective_theta_direction = 1 if umap_aligned else theta_direction
    if effective_theta_direction not in (-1, 1):
        raise ValueError(f"theta_direction must be -1 or 1, got {effective_theta_direction}.")

    ax.plot(theta_closed, E_closed, lw=2, color=line_color)
    ax.fill_between(theta_closed, 0, E_closed, color=fill_color, alpha=fill_alpha)
    ax.set_theta_zero_location(effective_theta_zero)
    ax.set_theta_direction(effective_theta_direction)
    ax.set_thetagrids(np.arange(0, 360, 45))
    rmin = float(np.nanmin(E_closed))
    rmax = float(np.nanmax(E_closed))
    if np.isfinite(rmin) and np.isfinite(rmax):
        if np.isclose(rmin, rmax):
            ax.set_rticks([rmin])
        else:
            ax.set_rticks(np.linspace(rmin, rmax, num=5))
    ax.yaxis.set_tick_params(labelsize=8)
    ax.xaxis.set_tick_params(labelsize=9)
    ax.minorticks_off()
    ax.grid(True, alpha=0.4)
    ax.set_rlabel_position(135)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig, ax


def plot_umap_and_rsp_side_by_side(
    adata: Any,
    gene: str,
    basis: str = "umap",
    layer: str | None = None,
    use_raw: bool = False,
    outpath: str | None = None,
    *,
    var_index: int | None = None,
    n_bins: int = 72,
    point_size: float = 6.0,
    vantage_point: tuple[float, float] | None = None,
    theta_zero: str = "E",
    theta_direction: int = 1,
    umap_aligned: bool = True,
    debug: bool = False,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]] | None:
    """Plot UMAP expression (left) and UMAP-aligned RSP polar profile (right).

    Args:
        adata: AnnData-like object with ``.obsm`` and expression matrix.
        gene: Feature name for labeling and index resolution.
        basis: Embedding basis name, e.g. ``"umap"`` or ``"X_umap"``.
        layer: Optional expression layer.
        use_raw: Use ``adata.raw`` expression.
        outpath: Optional output path. If provided, figure is saved and closed.
        var_index: Optional feature index in selected expression namespace.
        n_bins: Number of angular bins for RSP computation.
        point_size: Point size for UMAP scatter.
        vantage_point: Optional explicit vantage point ``(x0, y0)``. If None,
            compute from embedding with ``compute_vantage``.
        theta_zero: Polar zero-location when ``umap_aligned=False``.
        theta_direction: Polar direction when ``umap_aligned=False``.
        umap_aligned: Default True; forces 0° at East and CCW theta.
        debug: Run orientation sanity assertion when True.

    Returns:
        ``None`` when outpath is provided; otherwise ``(fig, (ax_umap, ax_polar))``.
    """
    if debug:
        _assert_theta_orientation_debug()

    emb_key = basis if basis in adata.obsm else (basis if basis.startswith("X_") else f"X_{basis}")
    if emb_key not in adata.obsm:
        raise KeyError(f"Embedding '{basis}' not found in adata.obsm (checked '{emb_key}').")

    embedding = np.asarray(adata.obsm[emb_key], dtype=float)
    if embedding.ndim != 2 or embedding.shape[1] < 2:
        raise ValueError(f"Embedding '{emb_key}' must have shape (N, 2+), got {embedding.shape}.")
    xy = embedding[:, :2]

    expr_matrix, var_names = _resolve_expr_matrix(adata, layer=layer, use_raw=use_raw)
    expr, label = _extract_expr_vector(expr_matrix, var_names, gene=gene, var_index=var_index)

    if vantage_point is None:
        center = compute_vantage(xy)
    else:
        center_arr = np.asarray(vantage_point, dtype=float).ravel()
        if center_arr.size != 2 or not np.isfinite(center_arr).all():
            raise ValueError("vantage_point must be a finite pair (x0, y0).")
        center = center_arr
    theta = compute_theta_from_embedding(xy, center, debug=debug)
    E_phi, _, _ = compute_rsp_profile(expr, theta, n_bins=n_bins)

    fig = plt.figure(figsize=(12, 5))
    ax_umap = fig.add_subplot(1, 2, 1)
    ax_polar = fig.add_subplot(1, 2, 2, projection="polar")

    ax_umap.scatter(
        xy[:, 0],
        xy[:, 1],
        c="lightgray",
        s=point_size,
        alpha=0.35,
        linewidths=0,
        rasterized=True,
    )
    order = np.argsort(expr)
    pts = ax_umap.scatter(
        xy[order, 0],
        xy[order, 1],
        c=expr[order],
        s=point_size,
        linewidths=0,
        cmap="Reds",
        rasterized=True,
    )
    cbar = fig.colorbar(pts, ax=ax_umap, fraction=0.046, pad=0.04)
    cbar.set_label("expression")
    ax_umap.scatter(
        [float(center[0])],
        [float(center[1])],
        marker="X",
        s=110,
        color="black",
        edgecolors="white",
        linewidths=1.0,
        label="vantage point",
        zorder=12,
    )
    ax_umap.set_title(f"UMAP: {label}")
    ax_umap.set_xlabel("UMAP1")
    ax_umap.set_ylabel("UMAP2")
    ax_umap.set_xticks([])
    ax_umap.set_yticks([])
    ax_umap.legend(loc="best", frameon=True, fontsize=8)

    plot_rsp(
        E_phi,
        title=f"RSP: {label}",
        ax=ax_polar,
        theta_zero=theta_zero,
        theta_direction=theta_direction,
        umap_aligned=umap_aligned,
        line_color="#8B0000",
        fill_color="#F08080",
        fill_alpha=0.30,
        debug=debug,
    )
    fig.tight_layout()

    if outpath is not None:
        out_dir = Path(outpath).parent.as_posix() or "."
        ensure_dir(out_dir)
        fig.savefig(outpath, dpi=150)
        plt.close(fig)
        return None
    return fig, (ax_umap, ax_polar)
