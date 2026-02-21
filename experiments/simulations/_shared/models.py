"""Shared gene simulation models for null and directional alternatives."""

from __future__ import annotations

import numpy as np
from scipy.special import expit, logit


def _clip_pi(pi_target: float) -> float:
    return float(np.clip(float(pi_target), 1e-12, 1.0 - 1e-12))


def simulate_null_gene(
    pi_target: float,
    donor_eta_per_cell: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Strict-null Bernoulli detection with donor random effects only."""
    eta = np.asarray(donor_eta_per_cell, dtype=float).ravel()
    alpha = float(logit(_clip_pi(pi_target)))
    p = expit(alpha + eta)
    return (rng.random(eta.size) < p).astype(bool)


def simulate_unimodal_gene(
    pi_target: float,
    beta: float,
    theta: np.ndarray,
    theta0: float,
    donor_eta_per_cell: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Directional alternative with cosine modulation on logit scale."""
    eta = np.asarray(donor_eta_per_cell, dtype=float).ravel()
    ang = np.asarray(theta, dtype=float).ravel()
    if eta.size != ang.size:
        raise ValueError("theta and donor_eta_per_cell must have same length.")
    alpha = float(logit(_clip_pi(pi_target)))
    logits = alpha + float(beta) * np.cos(ang - float(theta0)) + eta
    p = expit(logits)
    return (rng.random(ang.size) < p).astype(bool)


def apply_dropout_noise(
    f: np.ndarray,
    dropout_rate: float,
    rng: np.random.Generator,
    symmetric: bool = False,
    p01_factor: float = 0.5,
) -> np.ndarray:
    """Apply dropout/noise by flipping labels with configurable asymmetry."""
    arr = np.asarray(f, dtype=bool).ravel().copy()
    p = float(dropout_rate)
    if p <= 0.0:
        return arr
    if p > 1.0:
        raise ValueError("dropout_rate must be <= 1.")

    ones = np.where(arr)[0]
    if ones.size > 0:
        drop_mask = rng.random(ones.size) < p
        arr[ones[drop_mask]] = False

    if symmetric:
        zeros = np.where(~arr)[0]
        if zeros.size > 0:
            p01 = min(1.0, p * float(p01_factor))
            rise_mask = rng.random(zeros.size) < p01
            arr[zeros[rise_mask]] = True
    return arr


def _zscore(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float).ravel()
    mu = float(np.mean(arr))
    sd = float(np.std(arr))
    if sd <= 1e-12:
        return np.zeros_like(arr, dtype=float)
    return (arr - mu) / sd


def _circular_distance_vec(theta: np.ndarray, center: float) -> np.ndarray:
    d = np.abs(np.asarray(theta, dtype=float) - float(center))
    d = np.mod(d, 2.0 * np.pi)
    return np.minimum(d, 2.0 * np.pi - d)


def _patchy_circle_signal(
    theta: np.ndarray, theta0: float, lobe_count: int, width: float
) -> np.ndarray:
    k = max(1, int(lobe_count))
    centers = np.linspace(0.0, 2.0 * np.pi, k, endpoint=False) + float(theta0)
    th = np.asarray(theta, dtype=float).ravel()
    mats = []
    for c in centers:
        d = _circular_distance_vec(th, float(c))
        mats.append(np.exp(-0.5 * (d / max(float(width), 1e-3)) ** 2))
    s = np.mean(np.vstack(mats), axis=0)
    return _zscore(s)


def simulate_foreground_boolean(
    model: str,
    X: np.ndarray,
    theta: np.ndarray,
    donor_ids: np.ndarray,
    eta_by_donor: np.ndarray,
    rng: np.random.Generator,
    **params: float | int | np.ndarray,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    """Unified foreground simulation API for standardized experiments."""
    th = np.asarray(theta, dtype=float).ravel()
    donor = np.asarray(donor_ids, dtype=int).ravel()
    eta_d = np.asarray(eta_by_donor, dtype=float).ravel()
    if donor.size != th.size:
        raise ValueError("donor_ids and theta must have equal length.")
    eta = eta_d[donor]

    pi_target = _clip_pi(float(params.get("pi_target", 0.1)))
    alpha = float(logit(pi_target))
    beta = float(params.get("beta", 0.0))
    theta0 = float(params.get("theta0", float(rng.uniform(0.0, 2.0 * np.pi))))
    model_name = str(model)
    logits = alpha + eta

    if model_name == "null_logit":
        pass
    elif model_name == "angular_cosine":
        k = int(params.get("k", 1))
        logits = logits + float(beta) * np.cos(float(k) * (th - float(theta0)))
    elif model_name == "patchy_mog_circle":
        lobes = int(params.get("lobes", 4))
        width = float(params.get("width", 0.35))
        sig = _patchy_circle_signal(th, float(theta0), int(lobes), float(width))
        logits = logits + float(beta) * sig
    elif model_name == "step_islands":
        labels = np.asarray(params.get("cluster_labels"), dtype=int).ravel()
        if labels.size != th.size:
            raise ValueError("step_islands requires cluster_labels with length N.")
        target_cluster = int(params.get("target_cluster", 1))
        logits = logits + float(beta) * np.where(labels == target_cluster, 1.0, -1.0)
    elif model_name == "qc_driven":
        qc = np.asarray(params.get("qc_values"), dtype=float).ravel()
        if qc.size != th.size:
            raise ValueError("qc_driven requires qc_values with length N.")
        gamma = float(params.get("gamma", 1.0))
        logits = logits + float(gamma) * _zscore(qc)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    p = expit(logits)
    f = (rng.random(th.size) < p).astype(bool)
    meta: dict[str, float | int | str] = {
        "model": model_name,
        "pi_target": float(pi_target),
        "beta": float(beta),
        "theta0": float(theta0),
    }
    if model_name == "angular_cosine":
        meta["k"] = int(params.get("k", 1))
    if model_name == "patchy_mog_circle":
        meta["lobes"] = int(params.get("lobes", 4))
        meta["width"] = float(params.get("width", 0.35))
    if model_name == "qc_driven":
        meta["gamma"] = float(params.get("gamma", 1.0))
    return f, meta


def apply_dropout(
    f: np.ndarray, rate: float, rng: np.random.Generator, symmetric: bool = False
) -> np.ndarray:
    """Unified dropout helper alias."""
    return apply_dropout_noise(
        f=f, dropout_rate=float(rate), rng=rng, symmetric=bool(symmetric)
    )
