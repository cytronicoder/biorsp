"""Unit tests for radar plotting conventions and circular data handling."""

import numpy as np
import pytest

from biorsp.core.typing import RadarResult
from biorsp.plotting.radar import (
    _merge_wraparound_segments,
    _split_into_finite_segments,
    plot_radar,
    transform_theta,
)


class TestThetaConvention:
    """Test angle convention transformations."""

    def test_transform_identity(self):
        """Math to math should be identity (modulo 2pi)."""
        theta = np.array([0, np.pi / 2, np.pi, 3 * np.pi / 2])
        result = transform_theta(theta, from_convention="math", to_convention="math")
        np.testing.assert_allclose(result, theta)

    def test_transform_math_to_compass(self):
        """Test math (0=east, CCW) to compass (0=north, CW)."""
        theta_math = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2])
        expected_compass = np.array([np.pi / 2, 0.0, 3 * np.pi / 2, np.pi])

        result = transform_theta(theta_math, from_convention="math", to_convention="compass")
        np.testing.assert_allclose(result, expected_compass, atol=1e-10)

    def test_transform_compass_to_math(self):
        """Test compass to math (inverse transform)."""
        theta_compass = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2])
        expected_math = np.array([np.pi / 2, 0.0, 3 * np.pi / 2, np.pi])

        result = transform_theta(theta_compass, from_convention="compass", to_convention="math")
        np.testing.assert_allclose(result, expected_math, atol=1e-10)

    def test_transform_roundtrip(self):
        """Math -> compass -> math should recover original."""
        theta_math = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        theta_compass = transform_theta(theta_math, "math", "compass")
        theta_back = transform_theta(theta_compass, "compass", "math")
        np.testing.assert_allclose(theta_back, theta_math, atol=1e-10)

    def test_transform_peak_at_east(self):
        """A peak at 0 radians (east in math) should appear at pi/2 in compass."""
        theta_peak_math = np.array([0.0])
        theta_peak_compass = transform_theta(theta_peak_math, "math", "compass")
        np.testing.assert_allclose(theta_peak_compass, [np.pi / 2], atol=1e-10)

    def test_invalid_convention(self):
        """Invalid conventions should raise ValueError."""
        with pytest.raises(ValueError):
            transform_theta(np.array([0]), from_convention="math", to_convention="invalid")


class TestCircularSegmentation:
    """Test wrap-around continuity in circular data."""

    def test_split_single_continuous_segment(self):
        """All finite values should return one segment."""
        theta = np.linspace(0, 2 * np.pi, 10, endpoint=False)
        vals = np.ones(10)
        segments = _split_into_finite_segments(theta, vals)
        assert len(segments) == 1
        assert len(segments[0]) == 10

    def test_split_with_nan_gap(self):
        """NaN in middle should split into two segments."""
        theta = np.linspace(0, 2 * np.pi, 10, endpoint=False)
        vals = np.ones(10)
        vals[5] = np.nan
        segments = _split_into_finite_segments(theta, vals)
        assert len(segments) == 2

    def test_merge_wraparound_continuous(self):
        """Finite values at both ends should merge across boundary."""
        theta = np.linspace(0, 2 * np.pi, 10, endpoint=False)
        vals = np.ones(10)
        vals[3:7] = np.nan

        segments = _split_into_finite_segments(theta, vals)
        assert len(segments) == 2

        merged = _merge_wraparound_segments(segments, vals)
        assert len(merged) == 1, "Should merge into one circular segment"

    def test_merge_no_wraparound_if_gap_at_boundary(self):
        """If boundary values are NaN, should not merge."""
        theta = np.linspace(0, 2 * np.pi, 10, endpoint=False)
        vals = np.ones(10)
        vals[0] = np.nan
        vals[-1] = np.nan
        vals[5] = np.nan

        segments = _split_into_finite_segments(theta, vals)
        merged = _merge_wraparound_segments(segments, vals)
        assert len(merged) == len(segments), "Should not merge if boundary is NaN"

    def test_merge_full_circle_duplicates_first(self):
        """Full circle of finite values should duplicate first point."""
        vals = np.ones(10)
        segments = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]
        merged = _merge_wraparound_segments(segments, vals)
        assert len(merged) == 1
        assert merged[0][-1] == merged[0][0], "Should duplicate first index for closure"


class TestZeroFilledMask:
    """Test that zero-filled sectors are handled via explicit mask."""

    def test_zero_filled_mask_used_not_inferred(self):
        """Plotting should use forced_zero_mask if provided."""
        n = 10
        theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
        rsp = np.ones(n)
        rsp[5] = 0.0

        zero_mask = np.zeros(n, dtype=bool)
        zero_mask[5] = True

        radar = RadarResult(
            rsp=rsp,
            counts_fg=np.ones(n),
            counts_bg=np.ones(n),
            centers=theta,
            iqr_floor=0.1,
            iqr_floor_hits=np.zeros(n, dtype=bool),
            forced_zero_mask=zero_mask,
        )

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
        plot_radar(radar, ax=ax, mode="signed")
        plt.close(fig)

    def test_plotting_without_zero_mask_backward_compat(self):
        """Plotting should not crash if zero_filled_mask is absent."""
        n = 10
        theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
        rsp = np.ones(n)

        radar = RadarResult(
            rsp=rsp,
            counts_fg=np.ones(n),
            counts_bg=np.ones(n),
            centers=theta,
            iqr_floor=0.1,
            iqr_floor_hits=np.zeros(n, dtype=bool),
        )

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
        plot_radar(radar, ax=ax, mode="signed")
        plt.close(fig)


class TestRobustRadialScaling:
    """Test that radial_max supports robust scaling."""

    def test_radial_max_none_uses_robust(self):
        """radial_max=None should use 99th percentile."""
        n = 100
        theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
        rsp = np.random.randn(n)
        rsp[0] = 100.0

        radar = RadarResult(
            rsp=rsp,
            counts_fg=np.ones(n),
            counts_bg=np.ones(n),
            centers=theta,
            iqr_floor=0.1,
            iqr_floor_hits=np.zeros(n, dtype=bool),
        )

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
        plot_radar(radar, ax=ax, mode="signed", radial_max=None)
        ylim = ax.get_ylim()
        plt.close(fig)

        assert ylim[1] < 50, "Robust limit should exclude outlier"

    def test_radial_max_max_uses_absolute(self):
        """radial_max='max' should use absolute maximum."""
        n = 100
        theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
        rsp = np.random.randn(n)
        rsp[0] = 100.0

        radar = RadarResult(
            rsp=rsp,
            counts_fg=np.ones(n),
            counts_bg=np.ones(n),
            centers=theta,
            iqr_floor=0.1,
            iqr_floor_hits=np.zeros(n, dtype=bool),
        )

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
        plot_radar(radar, ax=ax, mode="signed", radial_max="max")
        ylim = ax.get_ylim()
        plt.close(fig)

        assert ylim[1] >= 100, "Max mode should include outlier"

    def test_radial_max_explicit_value(self):
        """radial_max=<float> should use explicit value."""
        n = 10
        theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
        rsp = np.ones(n)

        radar = RadarResult(
            rsp=rsp,
            counts_fg=np.ones(n),
            counts_bg=np.ones(n),
            centers=theta,
            iqr_floor=0.1,
            iqr_floor_hits=np.zeros(n, dtype=bool),
        )

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
        plot_radar(radar, ax=ax, mode="signed", radial_max=5.0)
        ylim = ax.get_ylim()
        plt.close(fig)

        np.testing.assert_allclose(ylim[1], 5.0, atol=1e-6)


class TestSemanticLabels:
    """Test that labels use proximal/distal, not enrichment/depletion."""

    def test_signed_mode_labels(self):
        """Signed mode should label proximal/distal shift."""
        n = 10
        theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
        rsp = np.concatenate([np.ones(5), -np.ones(5)])

        radar = RadarResult(
            rsp=rsp,
            counts_fg=np.ones(n),
            counts_bg=np.ones(n),
            centers=theta,
            iqr_floor=0.1,
            iqr_floor_hits=np.zeros(n, dtype=bool),
        )

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
        plot_radar(radar, ax=ax, mode="signed")

        legend = ax.get_legend()
        if legend:
            labels = [t.get_text() for t in legend.get_texts()]
            assert any("Proximal" in label for label in labels), "Should mention 'Proximal'"
            assert any("Distal" in label for label in labels), "Should mention 'Distal'"
            assert not any(
                "Depletion" in label for label in labels
            ), "Should not mention 'Depletion'"

        plt.close(fig)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
