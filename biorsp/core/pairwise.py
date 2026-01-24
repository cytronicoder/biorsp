"""Pairwise synergy/complementarity computations for BioRSP.

Key fix: gene-gene similarity is computed on SHARED support masks
with proper weighting.
"""

from __future__ import annotations

import numpy as np

from biorsp.core.geometry import wrapped_circular_distance
from biorsp.core.results import PairwiseResult
from biorsp.core.summaries import compute_scalar_summaries
from biorsp.core.typing import RadarResult


def _weighted_corr(a: np.ndarray, b: np.ndarray, w: np.ndarray, method: str = "pearson") -> float:
    """Compute weighted correlation with optional Spearman fallback."""
    mask = np.isfinite(a) & np.isfinite(b) & (w > 0)
    if np.sum(mask) < 2:
        return np.nan

    a_m = a[mask]
    b_m = b[mask]
    w_m = w[mask]

    sum_w = np.sum(w_m)
    if sum_w <= 0:
        return np.nan

    if method == "spearman":
        a_m = a_m.argsort().argsort().astype(float)
        b_m = b_m.argsort().argsort().astype(float)

    a_mean = np.sum(w_m * a_m) / sum_w
    b_mean = np.sum(w_m * b_m) / sum_w

    a_centered = a_m - a_mean
    b_centered = b_m - b_mean

    cov = np.sum(w_m * a_centered * b_centered) / sum_w
    var_a = np.sum(w_m * a_centered**2) / sum_w
    var_b = np.sum(w_m * b_centered**2) / sum_w

    if var_a < 1e-12 or var_b < 1e-12:
        return np.nan

    return float(cov / np.sqrt(var_a * var_b))


def _pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Compute unweighted Pearson correlation (backward compatible)."""
    mask = np.isfinite(a) & np.isfinite(b)
    if np.sum(mask) < 2:
        return np.nan
    a_masked = a[mask]
    b_masked = b[mask]
    if np.std(a_masked) == 0 or np.std(b_masked) == 0:
        return np.nan
    return float(np.corrcoef(a_masked, b_masked)[0, 1])


def _select_top_k(features: list[str], scores: dict[str, float], top_k: int | None) -> list[str]:
    if top_k is None or top_k <= 0 or top_k >= len(features):
        return features
    return sorted(features, key=lambda f: scores.get(f, -np.inf), reverse=True)[:top_k]


def compute_pairwise_relationships(
    radar_by_feature: dict[str, RadarResult],
    top_k: int | None = None,
    min_shared_mask_fraction: float = 0.5,
    corr_method: str = "pearson",
) -> tuple[list[PairwiseResult], list[PairwiseResult]]:
    """Compute pairwise synergy and complementarity from radar profiles.

    Key fix: uses SHARED geom_supported_mask and weighted correlation.

    Args:
        radar_by_feature: Mapping of feature name to RadarResult.
        top_k: Optional limit on number of features by anisotropy.
        min_shared_mask_fraction: Minimum shared support fraction.
        corr_method: Correlation method ("pearson" or "spearman").

    Returns:
        (synergy, complementarity) lists sorted by their respective scores.

    """
    if not radar_by_feature:
        return [], []

    features = list(radar_by_feature.keys())
    summaries = {f: compute_scalar_summaries(radar_by_feature[f]) for f in features}
    anisotropy_scores = {f: summaries[f].anisotropy for f in features}
    selected = _select_top_k(features, anisotropy_scores, top_k)

    synergy: list[PairwiseResult] = []
    complementarity: list[PairwiseResult] = []

    for i, f1 in enumerate(selected):
        radar1 = radar_by_feature[f1]
        r1 = radar1.rsp
        w1 = radar1.sector_weights
        mask1 = (
            radar1.geom_supported_mask
            if radar1.geom_supported_mask is not None
            else np.isfinite(r1)
        )

        for f2 in selected[i + 1 :]:
            radar2 = radar_by_feature[f2]
            r2 = radar2.rsp
            w2 = radar2.sector_weights
            mask2 = (
                radar2.geom_supported_mask
                if radar2.geom_supported_mask is not None
                else np.isfinite(r2)
            )

            shared_mask = mask1 & mask2
            shared_frac = float(np.mean(shared_mask))

            if shared_frac < min_shared_mask_fraction:
                synergy.append(
                    PairwiseResult(
                        feature_a=f1,
                        feature_b=f2,
                        correlation=np.nan,
                        complementarity=np.nan,
                        peak_distance=np.nan,
                    )
                )
                complementarity.append(
                    PairwiseResult(
                        feature_a=f1,
                        feature_b=f2,
                        correlation=np.nan,
                        complementarity=np.nan,
                        peak_distance=np.nan,
                    )
                )
                continue

            shared_weights = (
                np.sqrt(w1 * w2) if w1 is not None and w2 is not None else np.ones_like(r1)
            )
            shared_weights[~shared_mask] = 0.0

            corr = _weighted_corr(r1, r2, shared_weights, method=corr_method)
            comp = _weighted_corr(r1, -r2, shared_weights, method=corr_method)

            angle1 = summaries[f1].peak_extremal_angle
            angle2 = summaries[f2].peak_extremal_angle
            peak_distance = float(wrapped_circular_distance(np.array([angle1]), angle2)[0])

            synergy.append(
                PairwiseResult(
                    feature_a=f1,
                    feature_b=f2,
                    correlation=corr,
                    complementarity=comp,
                    peak_distance=peak_distance,
                )
            )
            complementarity.append(
                PairwiseResult(
                    feature_a=f1,
                    feature_b=f2,
                    correlation=corr,
                    complementarity=comp,
                    peak_distance=peak_distance,
                )
            )

    synergy_sorted = sorted(
        synergy, key=lambda x: (np.nan_to_num(x.correlation, nan=-np.inf)), reverse=True
    )
    complement_sorted = sorted(
        complementarity,
        key=lambda x: (np.nan_to_num(x.complementarity, nan=-np.inf)),
        reverse=True,
    )

    return synergy_sorted, complement_sorted


__all__ = ["compute_pairwise_relationships"]
