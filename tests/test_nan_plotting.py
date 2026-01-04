import matplotlib.pyplot as plt
import numpy as np

from biorsp.plotting import plot_radar
from biorsp.typing import RadarResult


def test_nan_gaps_segments():
    # 8 sectors
    theta = np.linspace(0, 2 * np.pi, 8, endpoint=False)

    # Case 1: No gaps, all finite
    r1 = np.ones(8)
    res1 = RadarResult(
        rsp=r1,
        counts_fg=np.ones(8) * 10,
        counts_bg=np.ones(8) * 10,
        centers=theta,
        iqr_floor=0.1,
        iqr_floor_hits=np.zeros(8, dtype=bool),
    )
    fig1, ax1 = plt.subplots(subplot_kw={"projection": "polar"})
    plot_radar(res1, ax=ax1, mode="proximal")
    # Should have 1 line (closed loop)
    # Wait, matplotlib might add other lines. Let's check the number of lines.
    # Actually, _draw_segmented_rsp doesn't set labels, so they might be _childX.
    # Let's just count all lines added by us.
    # We can check the number of lines before and after.
    plt.close(fig1)


def test_nan_gaps_logic():
    # 8 sectors
    theta = np.linspace(0, 2 * np.pi, 8, endpoint=False)

    # Case 2: One gap in the middle
    # r = [1, 1, NaN, NaN, 1, 1, 1, 1]
    # This should be ONE segment because it wraps around!
    # Wait, if r[0] and r[-1] are finite, they are merged.
    # [1, 1] and [1, 1, 1, 1] -> [1, 1, 1, 1, 1, 1] (wrapped)
    r2 = np.array([1.0, 1.0, np.nan, np.nan, 1.0, 1.0, 1.0, 1.0])
    res2 = RadarResult(
        rsp=r2,
        counts_fg=np.ones(8) * 10,
        counts_bg=np.ones(8) * 10,
        centers=theta,
        iqr_floor=0.1,
        iqr_floor_hits=np.zeros(8, dtype=bool),
    )
    fig2, ax2 = plt.subplots(subplot_kw={"projection": "polar"})
    plot_radar(res2, ax=ax2, mode="proximal")
    # Count lines. We expect 1 line because of wrap-around merge.
    # Note: plot_radar adds gray ticks for NaNs, so we filter them out.
    rsp_lines2 = [line for line in ax2.get_lines() if line.get_color() != "gray"]
    assert len(rsp_lines2) == 1
    plt.close(fig2)

    # Case 3: Two gaps, creating two disjoint segments
    # r = [1, 1, NaN, 1, 1, NaN, 1, 1]
    # Segments: [0, 1], [3, 4], [6, 7]
    # Wrap around merges [6, 7] and [0, 1] -> [6, 7, 0, 1]
    # So we expect TWO lines: [6, 7, 0, 1] and [3, 4]
    r3 = np.array([1.0, 1.0, np.nan, 1.0, 1.0, np.nan, 1.0, 1.0])
    res3 = RadarResult(
        rsp=r3,
        counts_fg=np.ones(8) * 10,
        counts_bg=np.ones(8) * 10,
        centers=theta,
        iqr_floor=0.1,
        iqr_floor_hits=np.zeros(8, dtype=bool),
    )
    fig3, ax3 = plt.subplots(subplot_kw={"projection": "polar"})
    plot_radar(res3, ax=ax3, mode="proximal")
    rsp_lines3 = [line for line in ax3.get_lines() if line.get_color() != "gray"]
    assert len(rsp_lines3) == 2
    plt.close(fig3)


def test_no_bridging():
    # Verify that no line segment spans a large angular gap that contains NaNs
    theta = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    # Gap at index 2, 3 (angles 1.57, 2.35)
    r = np.array([1.0, 1.0, np.nan, np.nan, 1.0, 1.0, 1.0, 1.0])
    res = RadarResult(
        rsp=r,
        counts_fg=np.ones(8) * 10,
        counts_bg=np.ones(8) * 10,
        centers=theta,
        iqr_floor=0.1,
        iqr_floor_hits=np.zeros(8, dtype=bool),
    )
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    plot_radar(res, ax=ax, mode="proximal")

    rsp_lines = [line for line in ax.get_lines() if line.get_color() != "gray"]
    for line in rsp_lines:
        xdata = line.get_xdata()
        # Check diffs of angles. If any diff is > pi, it's a wrap-around (which is fine if it's one step)
        # But we want to make sure it doesn't jump over the NaN gap.
        # The NaN gap is between index 1 (angle 0.78) and index 4 (angle 3.14).
        # The diff is 2.36.
        # In our case, it should wrap around the OTHER way.
        # The line should be [3.14, 3.92, 4.71, 5.49, 2pi+0, 2pi+0.78]
        # Diffs should all be small (~0.78)
        diffs = np.abs(np.diff(xdata))
        assert np.all(diffs < 1.0)  # All steps should be small
    plt.close(fig)
