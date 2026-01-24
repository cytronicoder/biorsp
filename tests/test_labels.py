import pandas as pd
import pytest

from biorsp.utils.labels import (
    CANONICAL_ARCHETYPES,
    assert_archetype_labels,
    classify_from_thresholds,
    normalize_archetype,
    normalize_archetype_series,
)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("housekeeping", "Ubiquitous"),
        ("regional_program", "Gradient"),
        ("localized_marker", "Patchy"),
        ("basal", "Basal"),
        ("Gradient", "Gradient"),
    ],
)
def test_normalize_archetype_maps_legacy_names(raw, expected):
    assert normalize_archetype(raw) == expected


def test_normalize_archetype_series_raises_on_unknown():
    series = pd.Series(["housekeeping", "mystery_label"])
    with pytest.raises(ValueError):
        normalize_archetype_series(series)


def test_assert_archetype_labels_rejects_invalid():
    df = pd.DataFrame({"label": ["Ubiquitous", "not_a_label"]})
    with pytest.raises(ValueError):
        assert_archetype_labels(df, "label")


def test_classify_from_thresholds_quadrants():
    c_cut, s_cut = 0.3, 0.2
    assert classify_from_thresholds(0.8, 0.1, c_cut, s_cut) == "Ubiquitous"
    assert classify_from_thresholds(0.8, 0.5, c_cut, s_cut) == "Gradient"
    assert classify_from_thresholds(0.1, 0.5, c_cut, s_cut) == "Patchy"
    assert classify_from_thresholds(0.1, 0.1, c_cut, s_cut) == "Basal"


@pytest.mark.parametrize("label", CANONICAL_ARCHETYPES)
def test_normalize_accepts_canonical(label):
    assert normalize_archetype(label) == label
