import numpy as np

from biorsp.core.engine import assign_sectors
from biorsp.core.geometry import prepare_polar


def test_prepare_polar_determinism_and_translation_invariance():
    rng = np.random.default_rng(123)
    embedding = rng.normal(size=(200, 2))
    shift = np.array([3.4, -1.2])

    prep_a = prepare_polar(
        embedding,
        seed=0,
        vantage="median",
        fixed_vantage=None,
        radius_norm="quantile",
        radius_q=0.95,
    )
    prep_b = prepare_polar(
        embedding,
        seed=0,
        vantage="median",
        fixed_vantage=None,
        radius_norm="quantile",
        radius_q=0.95,
    )
    prep_shift = prepare_polar(
        embedding + shift,
        seed=0,
        vantage="median",
        fixed_vantage=None,
        radius_norm="quantile",
        radius_q=0.95,
    )

    assert np.allclose(prep_a.r_norm, prep_b.r_norm)
    assert np.allclose(prep_a.theta, prep_b.theta)
    assert np.allclose(prep_a.r_norm, prep_shift.r_norm)
    assert np.allclose(prep_a.theta, prep_shift.theta)


def test_prepare_polar_rotation_consistency_sector_ids():
    rng = np.random.default_rng(42)
    embedding = rng.normal(size=(300, 2))
    angle = 2 * np.pi / 12
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rotated = embedding @ rot.T

    prep_a = prepare_polar(
        embedding,
        seed=1,
        vantage="centroid",
        fixed_vantage=None,
        radius_norm="quantile",
        radius_q=0.9,
    )
    prep_b = prepare_polar(
        rotated,
        seed=1,
        vantage="centroid",
        fixed_vantage=None,
        radius_norm="quantile",
        radius_q=0.9,
    )

    sector_a = assign_sectors(
        prep_a.theta,
        prep_a.r_norm,
        B=12,
        n_radial=1,
        radial_rule="equal",
        seed=1,
    )
    sector_b = assign_sectors(
        prep_b.theta,
        prep_b.r_norm,
        B=12,
        n_radial=1,
        radial_rule="equal",
        seed=1,
    )

    counts_a = np.bincount(sector_a.angle_bin, minlength=12)
    counts_b = np.bincount(sector_b.angle_bin, minlength=12)
    shift_bins = int(np.round((angle / (2 * np.pi)) * 12)) % 12
    counts_b_rot = np.roll(counts_b, -shift_bins)

    assert np.allclose(counts_a, counts_b_rot)
