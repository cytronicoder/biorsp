import numpy as np

from biorsp.constants import EPS
from biorsp.core import assess_adequacy, compute_rsp_radar
from biorsp.pairwise import compute_pairwise_relationships
from biorsp.results import FeatureResult, assign_feature_types
from biorsp.summaries import ScalarSummaries, compute_scalar_summaries
from biorsp.typing import AdequacyReport, RadarResult


def test_adequacy_fraction_matches_manual():
    theta = np.array(
        [
            -np.pi + 0.01,
            -np.pi + 0.02,
            -np.pi / 2 + 0.01,
            -np.pi / 2 + 0.02,
            0.01,
            0.02,
            np.pi / 2 + 0.01,
            np.pi / 2 + 0.02,
        ]
    )
    y = np.array([True, False] * 4)
    report = assess_adequacy(
        y,
        theta,
        n_sectors=4,
        delta_deg=90.0,
        min_fg_sector=1,
        min_bg_sector=1,
        min_fg_total=1,
        min_adequacy_fraction=0.5,
    )
    assert np.isclose(report.adequacy_fraction, 1.0)


def test_rsp_sign_convention_distal_negative():
    r = np.array([5.0, 5.0, 6.0, 7.0])
    theta = np.array([0.0, 0.1, -0.1, 0.2])
    y = np.array([False, False, True, True])
    radar = compute_rsp_radar(r, theta, y, B=1, delta_deg=360.0, min_fg_sector=1, min_bg_sector=1)
    assert radar.rsp[0] < 0


def test_iqr_floor_hits_when_sector_iqr_zero():
    r = np.array([5.0, 5.0, 5.0, 6.0, 6.0])
    theta = np.zeros_like(r)
    y = np.array([False, False, False, True, True])
    radar = compute_rsp_radar(r, theta, y, B=1, delta_deg=360.0, min_fg_sector=1, min_bg_sector=1)
    assert np.isclose(radar.iqr_floor, EPS)
    assert radar.iqr_floor_hits[0]


def test_scalar_summaries_anisotropy_and_peaks():
    centers = np.array([-np.pi, -np.pi / 2, 0.0, np.pi / 2])
    rsp = np.array([1.0, -2.0, 0.5, -0.5])
    radar = RadarResult(
        rsp=rsp,
        counts_fg=np.ones(4, dtype=int),
        counts_bg=np.ones(4, dtype=int),
        centers=centers,
        iqr_floor=1.0,
        iqr_floor_hits=np.zeros(4, dtype=bool),
    )
    summary = compute_scalar_summaries(radar)
    expected_anisotropy = np.sqrt(np.mean(rsp**2))
    assert np.isclose(summary.anisotropy, expected_anisotropy)
    assert summary.peak_distal == -2.0
    assert summary.peak_distal_angle == centers[1]
    assert summary.peak_proximal == 1.0
    assert summary.peak_proximal_angle == centers[0]
    assert summary.peak_extremal == -2.0
    assert summary.peak_extremal_angle == centers[1]


def _make_feature(feature, coverage, anisotropy) -> FeatureResult:
    adequacy = AdequacyReport(
        is_adequate=True,
        reason="ok",
        counts_fg=np.array([1]),
        counts_bg=np.array([1]),
        sector_mask=np.array([True]),
        n_foreground=1,
        n_background=1,
        adequacy_fraction=1.0,
        sector_indices=None,
    )
    summary = ScalarSummaries(
        peak_distal=-1.0,
        peak_distal_angle=0.0,
        peak_proximal=1.0,
        peak_proximal_angle=0.0,
        peak_extremal=1.0,
        peak_extremal_angle=0.0,
        anisotropy=anisotropy,
        max_rsp=1.0,
        min_rsp=-1.0,
        integrated_rsp=0.0,
    )
    return FeatureResult(
        feature=feature,
        threshold_quantile=0.0,
        coverage_quantile=coverage,
        coverage_prevalence=coverage,
        adequacy=adequacy,
        summaries=summary,
    )


def test_assign_feature_types_quadrants():
    features = {
        "a": _make_feature("a", coverage=0.8, anisotropy=2.0),
        "b": _make_feature("b", coverage=0.2, anisotropy=2.0),
        "c": _make_feature("c", coverage=0.8, anisotropy=0.5),
        "d": _make_feature("d", coverage=0.2, anisotropy=0.5),
    }
    updated, thresholds = assign_feature_types(
        features, coverage_field="coverage_prevalence", method="user", c_hi=0.5, A_hi=1.0
    )
    assert thresholds.coverage_threshold == 0.5
    assert thresholds.anisotropy_threshold == 1.0
    assert updated["a"].feature_type == "Type I"
    assert updated["b"].feature_type == "Type II"
    assert updated["c"].feature_type == "Type III"
    assert updated["d"].feature_type == "Type IV"


def test_pairwise_common_valid_directions_only():
    centers = np.array([0.0, 1.0, 2.0])
    radar_by_feature = {
        "f1": RadarResult(
            rsp=np.array([1.0, 2.0, np.nan]),
            counts_fg=np.ones(3, dtype=int),
            counts_bg=np.ones(3, dtype=int),
            centers=centers,
            iqr_floor=1.0,
            iqr_floor_hits=np.zeros(3, dtype=bool),
        ),
        "f2": RadarResult(
            rsp=np.array([1.0, 2.0, 3.0]),
            counts_fg=np.ones(3, dtype=int),
            counts_bg=np.ones(3, dtype=int),
            centers=centers,
            iqr_floor=1.0,
            iqr_floor_hits=np.zeros(3, dtype=bool),
        ),
        "f3": RadarResult(
            rsp=np.array([np.nan, np.nan, 1.0]),
            counts_fg=np.ones(3, dtype=int),
            counts_bg=np.ones(3, dtype=int),
            centers=centers,
            iqr_floor=1.0,
            iqr_floor_hits=np.zeros(3, dtype=bool),
        ),
    }
    synergy, _ = compute_pairwise_relationships(radar_by_feature)
    pair_map = {(p.feature_a, p.feature_b): p for p in synergy}
    assert np.isclose(pair_map[("f1", "f2")].correlation, 1.0)
    assert np.isnan(pair_map[("f1", "f3")].correlation)
