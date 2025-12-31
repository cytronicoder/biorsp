import numpy as np

from biorsp.foreground import foreground_identifiable


def test_foreground_identifiable_rejects_zero_threshold():
    x = np.zeros(10)
    y = x > 0
    assert not foreground_identifiable(x, y, threshold=0.0)


def test_foreground_identifiable_rejects_tied_foreground():
    x = np.array([0, 0, 1, 1, 2, 2, 2, 2])
    y = x > 1
    assert not foreground_identifiable(x, y, threshold=1.0)


def test_foreground_identifiable_accepts_distinct_values():
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    y = x > 3
    assert foreground_identifiable(x, y, threshold=3.0)
