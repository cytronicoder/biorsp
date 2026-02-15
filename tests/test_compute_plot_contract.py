import matplotlib

matplotlib.use("Agg")

import numpy as np

from biorsp.core.compute import compute_rsp
from biorsp.core.types import RSPConfig
from biorsp.plotting import rsp as rsp_plotting
from biorsp.plotting.rsp import plot_rsp


def test_compute_plot_contract(monkeypatch) -> None:
    rng = np.random.default_rng(0)
    n = 240
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    xy = np.column_stack([np.cos(theta), np.sin(theta)])

    # Directional bias toward east (+x).
    expr = (xy[:, 0] > 0.3).astype(float) + 0.05 * rng.normal(size=n)
    expr = np.clip(expr, 0.0, None)

    result = compute_rsp(
        expr=expr,
        embedding_xy=xy,
        config=RSPConfig(bins=72, center_method="median", threshold=0.0),
        feature_label="synthetic",
    )

    assert np.isfinite(result.theta).all()
    assert np.isfinite(result.R_theta).all()
    assert result.theta.shape == result.R_theta.shape
    assert result.theta.ndim == 1

    def _no_recompute(*_args, **_kwargs):
        raise AssertionError("plot_rsp should not recompute RSP")

    monkeypatch.setattr(rsp_plotting, "compute_rsp", _no_recompute)

    fig, ax = plot_rsp(result)
    assert fig is not None
    assert ax is not None
    assert ax.name == "polar"
    fig.clf()
