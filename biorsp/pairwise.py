"""
Pairwise synergy/complementarity computations for BioRSP.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .geometry import wrapped_circular_distance
from .radar import RadarResult
from .results import PairwiseResult
from .summaries import compute_scalar_summaries


def _pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if np.sum(mask) < 2:
        return np.nan
    a_masked = a[mask]
    b_masked = b[mask]
    if np.std(a_masked) == 0 or np.std(b_masked) == 0:
        return np.nan
    return float(np.corrcoef(a_masked, b_masked)[0, 1])


def _select_top_k(features: List[str], scores: Dict[str, float], top_k: Optional[int]) -> List[str]:
    if top_k is None or top_k <= 0 or top_k >= len(features):
        return features
    return sorted(features, key=lambda f: scores.get(f, -np.inf), reverse=True)[:top_k]


def compute_pairwise_relationships(
    radar_by_feature: Dict[str, RadarResult],
    top_k: Optional[int] = None,
) -> Tuple[List[PairwiseResult], List[PairwiseResult]]:
    """
    Compute pairwise synergy and complementarity from radar profiles.

    Args:
        radar_by_feature: Mapping of feature name to RadarResult.
        top_k: Optional limit on number of features by anisotropy.

    Returns:
        (synergy, complementarity) lists sorted by their respective scores.
    """
    if not radar_by_feature:
        return [], []

    features = list(radar_by_feature.keys())
    summaries = {f: compute_scalar_summaries(radar_by_feature[f]) for f in features}
    anisotropy_scores = {f: summaries[f].anisotropy for f in features}
    selected = _select_top_k(features, anisotropy_scores, top_k)

    synergy: List[PairwiseResult] = []
    complementarity: List[PairwiseResult] = []

    for i, f1 in enumerate(selected):
        r1 = radar_by_feature[f1].rsp
        for f2 in selected[i + 1 :]:
            r2 = radar_by_feature[f2].rsp
            corr = _pearson_corr(r1, r2)
            comp = _pearson_corr(r1, -r2)

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
