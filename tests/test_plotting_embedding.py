import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from biorsp.plotting import plot_embedding


def test_plot_embedding_binary_and_vantage():
    Z = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    c = np.array([0, 1, 0, 1])
    fig, ax = plt.subplots()
    ax = plot_embedding(Z, c=c, ax=ax, show_vantage=True, s=20)
    # Check that a 'v' text label was added
    texts = [t.get_text() for t in ax.texts]
    assert any(t == "v" for t in texts)
    # Check that legend contains Background, Foreground, and Vantage
    handles, labels = ax.get_legend_handles_labels()
    assert "Background" in labels
    assert "Foreground" in labels
    assert "Vantage" in labels
    plt.close(fig)
