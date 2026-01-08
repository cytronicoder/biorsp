import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from biorsp.core.typing import RadarResult
from biorsp.plotting.radar import plot_radar, plot_radar_absolute
from biorsp.preprocess.geometry import angle_grid


def test_plot_radar_all_nan():
    B = 36
    centers = angle_grid(B)
    rsp = np.full(B, np.nan)
    counts_fg = np.zeros(B, dtype=int)
    counts_bg = np.zeros(B, dtype=int)
    res = RadarResult(
        rsp=rsp,
        counts_fg=counts_fg,
        counts_bg=counts_bg,
        centers=centers,
        iqr_floor=0.1,
        iqr_floor_hits=np.zeros(B, dtype=bool),
    )

    ax = plot_radar(res, mode="signed", title="Test NaN")

    assert ax is not None
    assert "No valid sectors" in ax.get_title() or "Test NaN" in ax.get_title()
    plt.close()


def test_plot_radar_basic_modes():
    B = 36
    centers = angle_grid(B)

    theta = centers
    rsp = 1.0 * np.sin(theta * 2)

    rsp[::10] = np.nan

    counts_fg = np.ones(B, dtype=int) * 10
    counts_bg = np.ones(B, dtype=int) * 20

    res = RadarResult(
        rsp=rsp,
        counts_fg=counts_fg,
        counts_bg=counts_bg,
        centers=centers,
        iqr_floor=0.1,
        iqr_floor_hits=np.zeros(B, dtype=bool),
    )

    ax = plot_radar(res, mode="signed", title="Signed")
    assert ax is not None
    plt.close()

    ax = plot_radar(res, mode="proximal", title="Proximal")
    assert ax is not None
    plt.close()

    ax = plot_radar(res, mode="distal", title="Distal")
    assert ax is not None
    plt.close()

    ax = plot_radar(res, mode="relative", title="Relative (alias)")
    assert ax is not None
    plt.close()


def test_plot_radar_absolute_helper():
    B = 36
    centers = angle_grid(B)
    rsp = 0.5 * np.cos(centers * 3)
    counts_fg = np.ones(B, dtype=int) * 6
    counts_bg = np.ones(B, dtype=int) * 10
    res = RadarResult(
        rsp=rsp,
        counts_fg=counts_fg,
        counts_bg=counts_bg,
        centers=centers,
        iqr_floor=0.1,
        iqr_floor_hits=np.zeros(B, dtype=bool),
    )

    fig = plot_radar_absolute(res, title="Abs")
    assert fig is not None

    ax1, ax2 = fig.axes[:2]
    assert ax1.get_ylim() == ax2.get_ylim()
    plt.close()


def test_plot_fills_no_gaps():

    B = 36
    centers = angle_grid(B)
    rsp = np.zeros(B)

    rsp[2:7] = 0.5
    rsp[12:16] = 0.7
    rsp[28:33] = 0.4

    rsp[0] = np.nan
    counts_fg = np.ones(B, dtype=int) * 6
    counts_bg = np.ones(B, dtype=int) * 10
    res = RadarResult(
        rsp=rsp,
        counts_fg=counts_fg,
        counts_bg=counts_bg,
        centers=centers,
        iqr_floor=0.1,
        iqr_floor_hits=np.zeros(B, dtype=bool),
    )

    ax = plot_radar(res, mode="proximal", title="Segments")

    n_patches = len(ax.patches)
    assert n_patches >= 3, f"Expected at least 3 filled segments, found patches={n_patches}"
    plt.close()
