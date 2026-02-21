import numpy as np

from biorsp.peaks import circular_peak_prominences, find_circular_peaks


def test_find_circular_peaks_basic() -> None:
    x = np.array([0.0, 1.0, 0.0, 0.5, 0.0], dtype=float)
    out = find_circular_peaks(x, prominence_threshold=0.0)
    assert out["indices"].tolist() == [1, 3]
    assert out["prominences"].shape == (2,)


def test_prominence_nonnegative() -> None:
    x = np.array([0.0, 2.0, 0.1, 1.5, 0.0], dtype=float)
    out = find_circular_peaks(x, prominence_threshold=0.0)
    assert np.all(out["prominences"] >= 0.0)
    p = circular_peak_prominences(x, out["indices"])
    assert np.allclose(p, out["prominences"])


def test_prominence_threshold_filters() -> None:
    x = np.array([0.0, 1.0, 0.0, 0.3, 0.0], dtype=float)
    out_lo = find_circular_peaks(x, prominence_threshold=0.0)
    out_hi = find_circular_peaks(x, prominence_threshold=0.5)
    assert out_lo["indices"].size >= out_hi["indices"].size
