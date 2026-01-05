from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from tqdm import tqdm

from biorsp import (
    compute_rsp_radar,
    compute_scalar_summaries,
    compute_vantage,
)
from biorsp.simulations import simulate_dataset


def compute_baselines(coords, y, v):
    """Compute simple geometric baselines."""
    rel = coords - v
    r = np.linalg.norm(rel, axis=1)
    theta = np.arctan2(rel[:, 1], rel[:, 0])

    fg_mask = y.astype(bool)
    bg_mask = ~fg_mask

    # Baseline 1: Angular concentration (Mean Resultant Length)
    if np.any(fg_mask):
        theta_fg = theta[fg_mask]
        r_bar = np.sqrt(np.mean(np.cos(theta_fg)) ** 2 + np.mean(np.sin(theta_fg)) ** 2)
    else:
        r_bar = np.nan

    # Baseline 2: Radial separation (Diff in mean normalized radius)
    r_norm = (r - r.min()) / (r.max() - r.min() + 1e-9)
    if np.any(fg_mask) and np.any(bg_mask):
        rad_sep = np.mean(r_norm[fg_mask]) - np.mean(r_norm[bg_mask])
    else:
        rad_sep = np.nan

    return {"angular_concentration": r_bar, "radial_separation": rad_sep}


def simulate_baseline_rep(i, shape, etype, n_points, config):
    seed = config.get("seed", 42) + i
    data = simulate_dataset(n_points=n_points, shape=shape, enrichment_type=etype, seed=seed)
    coords = data["coords"]
    y = data["labels"]
    v = compute_vantage(coords)

    # BioRSP
    rel = coords - v
    r_vals = np.linalg.norm(rel, axis=1)
    theta_vals = np.arctan2(rel[:, 1], rel[:, 0])
    radar = compute_rsp_radar(r_vals, theta_vals, y)
    summaries = compute_scalar_summaries(radar)

    # Baselines
    base = compute_baselines(coords, y, v)

    return {
        "shape": shape,
        "type": etype,
        "seed": seed,
        "biorsp_anisotropy": summaries.anisotropy,
        "biorsp_coverage": np.mean(~np.isnan(radar.rsp)),
        **base,
    }


def run(outdir: Path, config: dict) -> dict:
    """
    Lightweight baseline comparisons.
    """
    n_reps = config.get("n_reps", 30)
    n_workers = config.get("n_workers", -1)
    shapes = ["disk", "crescent"]
    types = ["null", "rim", "core", "wedge", "rim+wedge"]
    n_points = 2000

    print("Running baseline comparisons...")

    tasks = [(i, shape, etype) for shape in shapes for etype in types for i in range(n_reps)]

    results = Parallel(n_jobs=n_workers)(
        delayed(simulate_baseline_rep)(i, shape, etype, n_points, config)
        for i, shape, etype in tqdm(tasks, desc="Baselines")
    )

    df = pd.DataFrame(results)
    df.to_csv(outdir / "tables" / "baseline_comparison.csv", index=False)

    # Figure B1: Metric separation
    plt.figure(figsize=(15, 5))
    metrics = ["biorsp_anisotropy", "angular_concentration", "radial_separation"]
    for i, metric in enumerate(metrics):
        plt.subplot(1, 3, i + 1)
        sns.boxplot(data=df, x="type", y=metric)
        plt.title(metric.replace("_", " ").title())
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(outdir / "figures" / "figB1_metric_separation.png")
    plt.close()

    # Figure B2: Scatter BioRSP vs Baselines
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    sns.scatterplot(
        data=df, x="angular_concentration", y="biorsp_anisotropy", hue="type", alpha=0.6
    )
    plt.subplot(1, 2, 2)
    sns.scatterplot(data=df, x="radial_separation", y="biorsp_anisotropy", hue="type", alpha=0.6)
    plt.tight_layout()
    plt.savefig(outdir / "figures" / "figB2_scatter_biorsp_vs_baselines.png")
    plt.close()

    return {"baseline_summary": "Baseline comparison completed."}
