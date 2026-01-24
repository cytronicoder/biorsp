"""Archetype label normalization and validation utilities.

Provides a single canonical vocabulary for archetype labels and helper
functions to normalize legacy names used across benchmarks and plots.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

CANONICAL_ARCHETYPES = ["Ubiquitous", "Gradient", "Patchy", "Basal"]
ABSTAIN_LABEL = "Abstain"

LEGACY_TO_CANONICAL = {
    "housekeeping": "Ubiquitous",
    "housekeeping_uniform": "Ubiquitous",
    "ubiquitous_housekeeping": "Ubiquitous",
    "ubiquitous": "Ubiquitous",
    "regional_program": "Gradient",
    "localized_program": "Gradient",
    "gradient": "Gradient",
    "niche_marker": "Patchy",
    "niche_biomarker": "Patchy",
    "localized_marker": "Patchy",
    "patchy": "Patchy",
    "sparse_noise": "Basal",
    "sparse_presence": "Basal",
    "basal": "Basal",
    "abstain": ABSTAIN_LABEL,
    "abstention": ABSTAIN_LABEL,
}

_CANONICAL_LOWER = {name.lower(): name for name in CANONICAL_ARCHETYPES}
_ALL_MAPPINGS = {**_CANONICAL_LOWER, **LEGACY_TO_CANONICAL, ABSTAIN_LABEL.lower(): ABSTAIN_LABEL}


def normalize_archetype(label: str) -> str:
    """Normalize a single archetype label to the canonical vocabulary.

    Parameters
    ----------
    label : str
        Input label (case-insensitive). Legacy aliases are supported.

    Returns
    -------
    str
        Canonical archetype name.

    Raises
    ------
    ValueError
        If the label is missing or not recognized.
    """

    if label is None or (isinstance(label, float) and np.isnan(label)):
        raise ValueError("Archetype label is missing or NaN")

    key = str(label).strip().lower()
    if key in _ALL_MAPPINGS:
        return _ALL_MAPPINGS[key]

    raise ValueError(
        f"Unknown archetype label: '{label}'. Expected one of: {CANONICAL_ARCHETYPES + [ABSTAIN_LABEL]}"
    )


def normalize_archetype_label(label: str) -> str:
    """Alias for ``normalize_archetype`` to disambiguate intent in callers."""

    return normalize_archetype(label)


def normalize_archetype_series(series: pd.Series, *, allow_abstain: bool = True) -> pd.Series:
    """Normalize all labels in a Series to canonical archetypes.

    Parameters
    ----------
    series : pd.Series
        Series containing archetype labels.

    Returns
    -------
    pd.Series
        Series with canonical labels.

    Raises
    ------
    ValueError
        If any label cannot be normalized.
    """

    def _map_val(value):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return np.nan
        key = str(value).strip().lower()
        return _ALL_MAPPINGS.get(key, np.nan)

    mapped = series.map(_map_val)

    # Detect invalid non-null original values that failed mapping
    invalid_mask = series.notna() & mapped.isna()
    if invalid_mask.any():
        bad_values = sorted({val for val in series.loc[invalid_mask].unique()})
        allowed = CANONICAL_ARCHETYPES + ([ABSTAIN_LABEL] if allow_abstain else [])
        raise ValueError(f"Found unsupported archetype labels: {bad_values}. Allowed: {allowed}")

    return pd.Series(mapped, index=series.index, dtype="object")


def assert_archetype_labels(df: pd.DataFrame, colname: str, *, allow_abstain: bool = False) -> None:
    """Assert that a DataFrame column contains only canonical archetype labels.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.
    colname : str
        Column name containing labels to check.

    Raises
    ------
    KeyError
        If the column is missing.
    ValueError
        If any label is not part of the canonical vocabulary.
    """

    if colname not in df.columns:
        raise KeyError(f"Column '{colname}' not found in DataFrame")

    unique_values = df[colname].dropna().unique()
    invalid: list[str] = []
    for value in unique_values:
        try:
            normed = normalize_archetype(value)
            if not allow_abstain and normed == ABSTAIN_LABEL:
                invalid.append(value)
                continue
        except ValueError:
            invalid.append(value)

    if invalid:
        raise ValueError(
            f"Column '{colname}' contains unsupported archetype labels: {sorted(invalid)}. "
            f"Allowed labels: {CANONICAL_ARCHETYPES + ([ABSTAIN_LABEL] if allow_abstain else [])}"
        )


def classify_from_thresholds(
    coverage: float, spatial_score: float, c_cut: float, s_cut: float
) -> str:
    """Classify a point into a canonical archetype using (C, S) thresholds.

    Coverage uses a high/low split at ``c_cut`` and spatial organization uses a
    high/low split at ``s_cut``. This function does *not* apply abstention logic;
    callers should separately gate on coverage/expressing cells if needed.
    """

    if np.isnan(coverage) or np.isnan(spatial_score):
        raise ValueError("Coverage and Spatial Score must be finite for classification")

    high_c = coverage >= c_cut
    high_s = spatial_score >= s_cut

    if high_c and not high_s:
        return "Ubiquitous"
    if high_c and high_s:
        return "Gradient"
    if (not high_c) and high_s:
        return "Patchy"
    return "Basal"


def canonicalize_labels(labels: Iterable[str]) -> list[str]:
    """Normalize an iterable of labels, raising on the first invalid entry."""

    return [normalize_archetype(label) for label in labels]


def label_order(include_abstain: bool = False) -> list[str]:
    """Return the canonical display order for archetype labels."""

    return CANONICAL_ARCHETYPES + ([ABSTAIN_LABEL] if include_abstain else [])


def label_palette(include_abstain: bool = False) -> dict[str, str]:
    """Stable palette keyed by canonical labels (optionally includes Abstain)."""

    palette = {
        "Ubiquitous": "#4CAF50",
        "Gradient": "#2196F3",
        "Patchy": "#FF7043",
        "Basal": "#9E9E9E",
    }
    if include_abstain:
        palette[ABSTAIN_LABEL] = "#000000"
    return palette
