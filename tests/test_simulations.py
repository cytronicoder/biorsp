import numpy as np
import pytest

from biorsp.simulations.generator import simulate_dataset, simulate_foreground, simulate_points


def test_simulate_points_disk():
    n = 1000
    coords = simulate_points(n_points=n, shape="disk", seed=42)
    assert coords.shape == (n, 2)

    r = np.sqrt(np.sum(coords**2, axis=1))
    assert np.max(r) < 1.2


def test_simulate_points_ellipse():
    n = 1000
    aspect = 3.0
    coords = simulate_points(n_points=n, shape="ellipse", aspect_ratio=aspect, seed=42)
    assert coords.shape == (n, 2)

    x_range = coords[:, 0].max() - coords[:, 0].min()
    y_range = coords[:, 1].max() - coords[:, 1].min()
    assert x_range > y_range


def test_simulate_foreground_rim():
    n = 1000
    coords = simulate_points(n_points=n, shape="disk", seed=42)
    scores, labels = simulate_foreground(coords, enrichment_type="rim", quantile=0.1, seed=42)

    r = np.sqrt(np.sum(coords**2, axis=1))
    fg_r = r[labels == 1]
    bg_r = r[labels == 0]
    assert np.mean(fg_r) > np.mean(bg_r)


def test_simulate_foreground_core():
    n = 1000
    coords = simulate_points(n_points=n, shape="disk", seed=42)
    scores, labels = simulate_foreground(coords, enrichment_type="core", quantile=0.1, seed=42)

    r = np.sqrt(np.sum(coords**2, axis=1))
    fg_r = r[labels == 1]
    bg_r = r[labels == 0]
    assert np.mean(fg_r) < np.mean(bg_r)


def test_simulate_dataset():
    data = simulate_dataset(
        n_points=500, shape="disk", enrichment_type="null", quantile=0.1, seed=42
    )
    assert "coords" in data
    assert "scores" in data
    assert "labels" in data
    assert data["coords"].shape == (500, 2)
    assert len(data["labels"]) == 500
    assert np.sum(data["labels"]) == pytest.approx(50, abs=5)
