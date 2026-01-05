import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# --- Shape Samplers ---


def _sample_disk(n: int, r_max: float = 1.0) -> np.ndarray:
    r = np.sqrt(np.random.uniform(0, r_max**2, n))
    theta = np.random.uniform(0, 2 * np.pi, n)
    return np.vstack([r * np.cos(theta), r * np.sin(theta)]).T


def _sample_annulus(n: int, r_in: float = 0.5, r_out: float = 1.0) -> np.ndarray:
    r = np.sqrt(np.random.uniform(r_in**2, r_out**2, n))
    theta = np.random.uniform(0, 2 * np.pi, n)
    return np.vstack([r * np.cos(theta), r * np.sin(theta)]).T


def _sample_crescent(n: int, R: float = 1.0, r: float = 0.7, d: float = 0.4) -> np.ndarray:
    # Rejection sampling for crescent (Disk R minus Disk r offset by d)
    coords = []
    while len(coords) < n:
        batch_size = max(n - len(coords), 100)
        # Sample from outer disk
        cand = _sample_disk(batch_size * 2, r_max=R)
        # Check if NOT in inner disk (centered at d, 0)
        dist_inner = np.sqrt((cand[:, 0] - d) ** 2 + cand[:, 1] ** 2)
        mask = dist_inner > r
        coords.extend(cand[mask][: n - len(coords)])
    return np.array(coords)


def _sample_two_lobe(n: int, sep: float = 1.5, r1: float = 1.0, r2: float = 0.8) -> np.ndarray:
    n1 = int(n * 0.5)
    n2 = n - n1
    lobe1 = _sample_disk(n1, r_max=r1) - np.array([sep / 2, 0])
    lobe2 = _sample_disk(n2, r_max=r2) + np.array([sep / 2, 0])
    return np.vstack([lobe1, lobe2])


def _sample_blob(
    n: int,
    R: float = 1.0,
    b1: float = 0.2,
    k1: int = 3,
    b2: float = 0.1,
    k2: int = 5,
    phi: float = 0.5,
) -> np.ndarray:
    theta = np.random.uniform(0, 2 * np.pi, n)
    r_max_theta = R * (1 + b1 * np.cos(k1 * theta) + b2 * np.cos(k2 * theta + phi))
    r = np.sqrt(np.random.uniform(0, 1, n)) * r_max_theta
    return np.vstack([r * np.cos(theta), r * np.sin(theta)]).T


def _apply_density(coords: np.ndarray, model: str = "uniform", strength: float = 1.0) -> np.ndarray:
    if model == "uniform":
        return coords

    n = len(coords)
    r = np.sqrt(np.sum(coords**2, axis=1))
    theta = np.arctan2(coords[:, 1], coords[:, 0])

    probs = np.ones(n)
    if model == "radial_center":
        probs = np.exp(-strength * r)
    elif model == "radial_rim":
        probs = np.exp(strength * (r - r.max()))
    elif model == "angular_bias":
        probs = np.exp(strength * np.cos(theta))
    elif model == "gmm":
        centers = [np.array([0.5, 0.5]), np.array([-0.5, -0.5])]
        for c in centers:
            dists = np.linalg.norm(coords - c, axis=1)
            probs += strength * np.exp(-(dists**2) / 0.1)

    probs /= probs.sum()
    idx = np.random.choice(n, size=n, replace=True, p=probs)
    return coords[idx]


def _apply_distortion(coords: np.ndarray, dist_type: str = "none", **kwargs) -> np.ndarray:
    if dist_type == "none":
        return coords

    x, y = coords[:, 0], coords[:, 1]
    if dist_type == "swirl":
        w = kwargs.get("swirl_strength", 1.0)
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x) + w * r
        return np.vstack([r * np.cos(theta), r * np.sin(theta)]).T

    elif dist_type == "anisotropic":
        a = kwargs.get("a", 0.2)
        b = kwargs.get("b", 0.1)
        x_new = x * (1 + a * x**2)
        y_new = y * (1 + b * y**2)
        return np.vstack([x_new, y_new]).T

    return coords


def simulate_points(
    n_points: int = 2000,
    shape: str = "disk",
    density_model: str = "uniform",
    density_strength: float = 1.0,
    distortion: str = "none",
    seed: Optional[int] = None,
    **kwargs,
) -> np.ndarray:
    """
    Generate 2D point coordinates for a given footprint shape and density model.
    """
    # Sample base shape
    s = shape.lower()
    if s == "disk":
        coords = _sample_disk(n_points)
    elif s == "ellipse":
        coords = _sample_disk(n_points)
        coords[:, 0] *= kwargs.get("aspect_ratio", 2.0)
    elif s == "annulus":
        coords = _sample_annulus(n_points, r_in=kwargs.get("r_in", 0.5))
    elif s == "crescent":
        coords = _sample_crescent(n_points, d=kwargs.get("d", 0.4))
    elif s == "two_lobe":
        coords = _sample_two_lobe(n_points, sep=kwargs.get("sep", 1.5))
    elif s == "blob":
        coords = _sample_blob(n_points)
    else:
        raise ValueError(f"Unknown shape: {shape}")

    # Apply density model
    coords = _apply_density(coords, model=density_model, strength=density_strength)

    # Apply rotation
    rot = kwargs.get("rotation_deg", 0.0)
    if rot != 0:
        rad = np.radians(rot)
        c, s = np.cos(rad), np.sin(rad)
        R = np.array(((c, -s), (s, c)))
        coords = coords @ R.T

    # Apply distortion
    coords = _apply_distortion(coords, dist_type=distortion, **kwargs)

    # Add jitter
    jitter = kwargs.get("noise_sigma", kwargs.get("jitter", 0.02))
    if jitter > 0:
        coords += np.random.normal(0, jitter, coords.shape)

    return coords


def simulate_foreground(
    coords: np.ndarray,
    enrichment_type: str = "null",
    quantile: float = 0.2,
    seed: Optional[int] = None,
    noise_sigma: float = 0.1,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Assign scores and binary labels based on enrichment geometry.
    """
    n_points = coords.shape[0]
    r = np.sqrt(np.sum(coords**2, axis=1))
    theta = np.arctan2(coords[:, 1], coords[:, 0])

    # Normalize r to [0, 1] for scoring
    r_norm = (r - r.min()) / (r.max() - r.min() + 1e-9)

    scores = np.zeros(n_points)
    e = enrichment_type.lower()

    if e == "null":
        scores = np.random.uniform(0, 1, n_points)

    elif e == "rim":
        scores = r_norm + np.random.normal(0, noise_sigma, n_points)

    elif e == "core":
        scores = (1.0 - r_norm) + np.random.normal(0, noise_sigma, n_points)

    elif e == "wedge":
        target = np.radians(kwargs.get("wedge_center_deg", 0.0))
        diff = np.abs(theta - target)
        diff = np.where(diff > np.pi, 2 * np.pi - diff, diff)
        scores = (1.0 - diff / np.pi) + np.random.normal(0, noise_sigma, n_points)

    elif e == "rim+wedge":
        target = np.radians(kwargs.get("wedge_center_deg", 0.0))
        diff = np.abs(theta - target)
        diff = np.where(diff > np.pi, 2 * np.pi - diff, diff)
        scores = (r_norm + (1.0 - diff / np.pi)) / 2.0 + np.random.normal(0, noise_sigma, n_points)

    elif e == "two_sector":
        mu = np.radians(kwargs.get("wedge_center_deg", 0.0))
        diff1 = np.abs(theta - mu)
        diff1 = np.where(diff1 > np.pi, 2 * np.pi - diff1, diff1)
        diff2 = np.abs(theta - (mu + np.pi))
        diff2 = np.where(diff2 > np.pi, 2 * np.pi - diff2, diff2)
        scores = (1.0 - np.minimum(diff1, diff2) / np.pi) + np.random.normal(
            0, noise_sigma, n_points
        )

    elif e == "patch":
        center = kwargs.get("patch_center", np.array([0.5, 0.5]))
        radius = kwargs.get("patch_radius", 0.3)
        dists = np.linalg.norm(coords - center, axis=1)
        scores = np.exp(-(dists**2) / (2 * radius**2)) + np.random.normal(0, noise_sigma, n_points)

    elif e == "boundary":
        # Approximate boundary distance via r_max(theta)
        # For simplicity, we'll use r_norm as a proxy for most shapes,
        # but for crescent/annulus it's more complex.
        # Let's just use r_norm for now as it's "distance from center".
        scores = r_norm + np.random.normal(0, noise_sigma, n_points)

    elif e == "confounded":
        # Correlated with density (proxy: local point count or just the density model)
        # Here we'll just use a simple radial bias as a proxy for "density" if model was radial
        scores = r_norm + np.random.normal(0, noise_sigma, n_points)

    else:
        raise ValueError(f"Unknown enrichment type: {enrichment_type}")

    threshold = np.quantile(scores, 1.0 - quantile)
    labels = (scores >= threshold).astype(int)

    return scores, labels


def simulate_dataset(
    n_points: int = 2000,
    shape: str = "disk",
    enrichment_type: str = "null",
    quantile: float = 0.2,
    seed: Optional[int] = None,
    noise_sigma: float = 0.1,
    **kwargs,
) -> Dict[str, Any]:
    """
    Generate a complete simulation dataset.
    """
    if seed is not None:
        np.random.seed(seed)

    coords = simulate_points(
        n_points=n_points,
        shape=shape,
        seed=None,  # Seed already set
        noise_sigma=noise_sigma,
        **kwargs,
    )
    scores, labels = simulate_foreground(
        coords,
        enrichment_type=enrichment_type,
        quantile=quantile,
        seed=None,  # Seed already set
        noise_sigma=noise_sigma,
        **kwargs,
    )

    return {
        "coords": coords,
        "scores": scores,
        "labels": labels,
        "metadata": {
            "n_points": n_points,
            "shape": shape,
            "enrichment_type": enrichment_type,
            "quantile": quantile,
            "seed": seed,
            "noise_sigma": noise_sigma,
            **kwargs,
        },
    }


def save_dataset(data: Dict[str, Any], output_dir: str, prefix: str):
    """
    Save simulation data to disk.

    Saves both binary numpy files and a human-readable CSV containing x,y,score,label.
    """
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, f"{prefix}_coords.npy"), data["coords"])
    np.save(os.path.join(output_dir, f"{prefix}_scores.npy"), data["scores"])
    np.save(os.path.join(output_dir, f"{prefix}_labels.npy"), data["labels"])
    meta_df = pd.DataFrame([data["metadata"]])
    meta_df.to_csv(os.path.join(output_dir, f"{prefix}_metadata.csv"), index=False)

    # Also save a combined CSV for easy inspection / downstream analysis
    coords = np.asarray(data["coords"])
    scores = np.asarray(data["scores"])
    labels = np.asarray(data["labels"])
    df_comb = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "score": scores, "label": labels})
    df_comb.to_csv(os.path.join(output_dir, f"{prefix}_points.csv"), index=False)


def generate_grid(
    shapes: List[str] = None,
    enrichments: List[str] = None,
    n_points: int = 2000,
    output_dir: str = "sim_results/inputs",
    seed: int = 42,
):
    """
    Generate a suite of simulations for benchmarking.
    """
    if enrichments is None:
        enrichments = ["null", "rim", "core", "wedge", "rim+wedge", "two_sector", "patch"]
    if shapes is None:
        shapes = ["disk", "ellipse", "annulus", "crescent", "two_lobe", "blob"]
    for shape in shapes:
        for enrichment in enrichments:
            prefix = f"{shape}_{enrichment}".replace("+", "_")
            print(f"Generating {prefix}...")
            data = simulate_dataset(
                n_points=n_points, shape=shape, enrichment_type=enrichment, seed=seed
            )
            save_dataset(data, output_dir, prefix)


def ground_truth_summary(enrichment_type: str) -> Dict[str, Any]:
    """
    Returns expected RSP properties for validation.
    """
    mapping = {
        "null": {"expected_sign": 0, "anisotropy": "low"},
        "rim": {"expected_sign": 1, "anisotropy": "low"},
        "core": {"expected_sign": -1, "anisotropy": "low"},
        "wedge": {"expected_sign": 0, "anisotropy": "high"},
        "rim+wedge": {"expected_sign": 1, "anisotropy": "high"},
        "two_sector": {"expected_sign": 0, "anisotropy": "high"},
        "patch": {"expected_sign": 0, "anisotropy": "high"},
    }
    return mapping.get(enrichment_type.lower(), {})
