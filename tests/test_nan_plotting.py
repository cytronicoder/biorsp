import matplotlib.pyplot as plt
import numpy as np

from biorsp.core.typing import RadarResult
from biorsp.plotting.radar import plot_radar


def test_nan_gaps_segments():
    theta = np.linspace(0, 2 * np.pi, 8, endpoint=False)

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

    plt.close(fig1)


def test_nan_gaps_logic():
    theta = np.linspace(0, 2 * np.pi, 8, endpoint=False)

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

    rsp_lines2 = [line for line in ax2.get_lines() if line.get_color() != "gray"]
    assert len(rsp_lines2) == 1
    plt.close(fig2)

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
    theta = np.linspace(0, 2 * np.pi, 8, endpoint=False)

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

        diffs = np.abs(np.diff(xdata))
        assert np.all(diffs < 1.0)
    plt.close(fig)
