"""
Preprocessing module for BioRSP.

Implements foreground selection and radial normalization:
- Binary and weighted foreground definition
- Robust radial normalization
"""

from typing import Optional, Tuple

import numpy as np
from scipy.stats import iqr


def normalize_radii(
    r: np.ndarray, eps: float = 1e-8, method: str = "robust_iqr"
) -> Tuple[np.ndarray, dict]:
    """
    Perform within-set robust radial normalization.

    r_hat_i = (r_i - median(r)) / (IQR(r) + eps)

    Parameters
    ----------
    r : np.ndarray
        (N,) array of radial distances.
    eps : float, optional
        Small constant to prevent division by zero, by default 1e-8.
    method : str, optional
        Normalization method (currently only "robust_iqr" supported), by default "robust_iqr".

    Returns
    -------
    r_hat : np.ndarray
        (N,) array of normalized radii.
    stats : dict
        Dictionary containing 'median_r', 'iqr_r', 'eps', and 'n_non_finite'.
    """
    if not np.all(np.isfinite(r)):
        n_non_finite = int(np.sum(~np.isfinite(r)))
        raise ValueError(f"Input radii contain {n_non_finite} non-finite values (NaN/inf).")

    median_r = np.median(r)
    iqr_r = iqr(r)

    # Fallback if IQR is too small
    if iqr_r < eps:
        mad = np.median(np.abs(r - median_r))
        iqr_r = 1.4826 * mad
        if iqr_r < eps:
            iqr_r = np.std(r)

    r_hat = (r - median_r) / (iqr_r + eps)

    stats = {
        "median_r": float(median_r),
        "iqr_r": float(iqr_r),
        "eps": float(eps),
        "n_non_finite": 0,
    }

    return r_hat, stats


def define_foreground(
    x: np.ndarray,
    mode: str = "quantile",
    q: float = 0.9,
    abs_threshold: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
    min_nonzero: int = 20,
    min_fg: int = 50,
    target_frac: Optional[float] = None,
    frac_bounds: Tuple[float, float] = (0.02, 0.20),
    overshoot_tol: float = 0.25,
    sharpness: float = 10.0,
    min_effective_fg: float = 10.0,
) -> Tuple[Optional[np.ndarray], dict]:
    """
    Define foreground indicator (binary or weighted) with robust tie-handling.

    Parameters
    ----------
    x : np.ndarray
        (N,) array of expression values (non-negative floats).
    mode : str, optional
        "quantile", "absolute", "auto", or "weights", by default "quantile".
    q : float, optional
        Quantile for threshold, by default 0.9.
    abs_threshold : float, optional
        Optional absolute threshold (expr >= T), by default None.
    rng : np.random.Generator, optional
        Random generator for deterministic tie-breaking, by default None.
    min_nonzero : int, optional
        Minimum number of non-zero values required, by default 20.
    min_fg : int, optional
        Minimum number of foreground cells required, by default 50.
    target_frac : float, optional
        Target foreground fraction, by default 1 - q.
    frac_bounds : Tuple[float, float], optional
        Acceptable foreground fraction range for auto mode, by default (0.02, 0.20).
    overshoot_tol : float, optional
        Tolerance for quantile overshoot before tie-breaking, by default 0.25.
    sharpness : float, optional
        Sharpness for "weights" mode, by default 10.0.
    min_effective_fg : float, optional
        Minimum effective foreground mass for "weights" mode, by default 10.0.

    Returns
    -------
    fg : np.ndarray or None
        (N,) boolean mask, float weights, or None if underpowered.
    info : dict
        Dictionary with detailed decision metadata.
    """
    N = len(x)
    if rng is None:
        rng = np.random.default_rng(0)

    if mode == "weights":
        return define_foreground_weights(
            x,
            q=q,
            abs_threshold=abs_threshold,
            sharpness=sharpness,
            min_effective_fg=min_effective_fg,
        )

    if target_frac is None:
        target_frac = 1.0 - q
    target_count = int(np.clip(round(target_frac * N), 1, N))

    info = {
        "mode": mode,
        "q": q,
        "abs_threshold": abs_threshold,
        "N": N,
        "target_frac": target_frac,
        "target_count": target_count,
        "status": "ok",
    }

    n_nonzero = int(np.sum(x > 0))
    info["n_nonzero"] = n_nonzero
    if n_nonzero < min_nonzero:
        info["status"] = "underpowered_nonzero"
        return None, info

    fg_mask = None
    rule = None

    if mode == "absolute":
        if abs_threshold is None:
            raise ValueError("abs_threshold must be provided for mode='absolute'")
        cand = x >= abs_threshold
        n_fg = int(np.sum(cand))
        if n_fg < min_fg:
            info["status"] = "underpowered_absolute_fg"
            return None, info
        fg_mask = cand
        rule = "absolute_ge"

    elif mode == "auto":
        if abs_threshold is None:
            mode = "quantile"
        else:
            cand = x >= abs_threshold
            n_fg = int(np.sum(cand))
            frac_fg = n_fg / N
            if n_fg >= min_fg and frac_bounds[0] <= frac_fg <= frac_bounds[1]:
                fg_mask = cand
                rule = "auto_absolute_ge"
            else:
                mode = "quantile"
                info["auto_fallback_quantile"] = True

    if mode == "quantile":
        tau = float(np.quantile(x, q))
        info["tau"] = tau

        if tau > 0:
            naive = x >= tau
            n_naive = int(np.sum(naive))
            frac_naive = n_naive / N

            if (
                target_frac * (1 - overshoot_tol) <= frac_naive <= target_frac * (1 + overshoot_tol)
            ) and n_naive >= min_fg:
                fg_mask = naive
                rule = "quantile_ge"
            else:
                s_high = x > tau
                s_tie = x == tau
                n_high = int(np.sum(s_high))
                n_tie = int(np.sum(s_tie))
                info["n_high"] = n_high
                info["n_tie"] = n_tie

                fg_mask = s_high.copy()
                remaining = target_count - n_high

                if remaining > 0:
                    tie_indices = np.where(s_tie)[0]
                    sampled_indices = rng.choice(
                        tie_indices, size=min(remaining, n_tie), replace=False
                    )
                    fg_mask[sampled_indices] = True
                    info["sampled_k"] = len(sampled_indices)
                else:
                    info["kept_only_s_high"] = True
                    info["sampled_k"] = 0

                rule = "quantile_tie_subsample"
                if info.get("auto_fallback_quantile"):
                    rule = "auto_fallback_" + rule

                if np.sum(fg_mask) < min_fg:
                    info["status"] = "underpowered_quantile_fg"
                    return None, info
        else:
            nz_idx = np.where(x > 0)[0]
            n_nz = len(nz_idx)
            if n_nz < min_fg:
                info["status"] = "underpowered_nonzero_fg"
                return None, info

            if n_nz <= target_count:
                fg_mask = x > 0
                rule = "all_nonzero"
            else:
                nz_values = x[nz_idx]
                sort_idx = np.argsort(-nz_values)
                cutoff_val = nz_values[sort_idx[target_count - 1]]

                s_high = x > cutoff_val
                s_tie = x == cutoff_val
                n_high = int(np.sum(s_high))
                n_tie = int(np.sum(s_tie))
                info["n_high"] = n_high
                info["n_tie"] = n_tie

                fg_mask = s_high.copy()
                remaining = target_count - n_high

                if remaining > 0:
                    tie_indices = np.where(s_tie)[0]
                    sampled_indices = rng.choice(
                        tie_indices, size=min(remaining, n_tie), replace=False
                    )
                    fg_mask[sampled_indices] = True
                    info["sampled_k"] = len(sampled_indices)
                else:
                    info["kept_only_s_high"] = True
                    info["sampled_k"] = 0
                rule = "zero_inflated_quantile_subsample"

    info["rule"] = rule
    info["n_fg"] = int(np.sum(fg_mask))
    info["realized_frac"] = info["n_fg"] / N

    return fg_mask, info


def define_foreground_weights(
    x: np.ndarray,
    method: str = "logistic",
    q: float = 0.9,
    abs_threshold: Optional[float] = None,
    tau: Optional[float] = None,
    scale: Optional[float] = None,
    sharpness: float = 10.0,
    min_effective_fg: float = 10.0,
) -> Tuple[np.ndarray, dict]:
    """
    Define continuous foreground weights w in [0, 1] based on expression.

    Parameters
    ----------
    x : np.ndarray
        (N,) array of expression values.
    method : str, optional
        "logistic" or "rank", by default "logistic".
    q : float, optional
        Quantile for threshold, by default 0.9.
    abs_threshold : float, optional
        Optional absolute threshold, by default None.
    tau : float, optional
        Center of the logistic function, by default None.
    scale : float, optional
        Slope/scale of the logistic function, by default None.
    sharpness : float, optional
        Sharpness parameter for logistic function, by default 10.0.
    min_effective_fg : float, optional
        Minimum effective foreground mass required, by default 10.0.

    Returns
    -------
    w : np.ndarray
        (N,) array of weights in [0, 1].
    info : dict
        Dictionary with detailed decision metadata.
    """
    N = len(x)
    info = {
        "mode": "weights",
        "method": method,
        "q": q,
        "abs_threshold": abs_threshold,
        "sharpness": sharpness,
        "N": N,
        "status": "ok",
    }

    if method == "logistic":
        if tau is None:
            if abs_threshold is not None:
                tau = abs_threshold
            else:
                tau = float(np.quantile(x, q))
                if tau == 0:
                    nz = x[x > 0]
                    if len(nz) > 0:
                        tau = float(np.quantile(nz, q))
                    else:
                        tau = 0.0
        info["tau"] = tau

        if scale is None:
            nz = x[x > 0]
            if len(nz) > 1:
                median_nz = np.median(nz)
                mad = np.median(np.abs(nz - median_nz))
                scale = mad if mad > 0 else (np.percentile(nz, 75) - np.percentile(nz, 25)) / 1.349
            if scale is None or scale == 0:
                scale = 1.0
        info["scale"] = scale

        eps = 1e-8
        z = (x - tau) / (scale + eps) * sharpness
        w = 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))

    elif method == "rank":
        ranks = np.argsort(np.argsort(x))
        w = ranks / max(1, N - 1)
    else:
        raise ValueError(f"Unknown method: {method}")

    eff_fg = np.sum(w)
    info["eff_fg_total"] = float(eff_fg)
    if eff_fg < min_effective_fg:
        info["status"] = "underpowered_effective_fg"

    return w, info


__all__ = [
    "normalize_radii",
    "define_foreground",
    "define_foreground_weights",
]
