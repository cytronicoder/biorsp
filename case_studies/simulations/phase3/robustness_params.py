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


def interpolate_profile(theta, rsp, target_grid):
    """Interpolate RSP profile to a target grid, handling NaNs."""
    # Fill NaNs with 0 for interpolation or handle them?
    # Requirement says "RMSD between normalized radar profiles".
    # Let's use 0 for NaNs in the profile for comparison purposes if they are underpowered.
    rsp_filled = np.nan_to_num(rsp, nan=0.0)
    # Handle wrap around for interpolation
    theta_ext = np.concatenate([theta - 2 * np.pi, theta, theta + 2 * np.pi])
    rsp_ext = np.concatenate([rsp_filled, rsp_filled, rsp_filled])
    return np.interp(target_grid, theta_ext, rsp_ext)


def simulate_robustness_rep(
    i, shape, etype, n, config, ref_B, ref_delta, ref_min_fg, grid_sizes, sector_widths, common_grid
):
    seed = config.get("seed", 42) + i
    data = simulate_dataset(n_points=n, shape=shape, enrichment_type=etype, seed=seed)
    coords = data["coords"]
    y = data["labels"]
    v = compute_vantage(coords)
    rel = coords - v
    r = np.linalg.norm(rel, axis=1)
    theta = np.arctan2(rel[:, 1], rel[:, 0])

    rep_results = []

    # Reference run
    ref_radar = compute_rsp_radar(
        r, theta, y, B=ref_B, delta_deg=ref_delta, min_fg_sector=ref_min_fg
    )
    ref_prof = interpolate_profile(ref_radar.centers, ref_radar.rsp, common_grid)
    ref_norm = np.linalg.norm(ref_prof) if np.linalg.norm(ref_prof) > 0 else 1.0
    ref_prof /= ref_norm

    # Sweep B
    for B in grid_sizes:
        radar = compute_rsp_radar(r, theta, y, B=B, delta_deg=ref_delta, min_fg_sector=ref_min_fg)
        prof = interpolate_profile(radar.centers, radar.rsp, common_grid)
        prof /= np.linalg.norm(prof) if np.linalg.norm(prof) > 0 else 1.0
        sim = 1.0 - np.sqrt(np.mean((prof - ref_prof) ** 2))
        summ = compute_scalar_summaries(radar)
        rep_results.append(
            {
                "dataset_id": f"{shape}_{etype}_N{n}",
                "shape": shape,
                "type": etype,
                "N": n,
                "seed": seed,
                "param": "theta_grid_size",
                "value": B,
                "anisotropy": summ.anisotropy,
                "coverage": np.mean(~np.isnan(radar.rsp)),
                "similarity": sim,
                "abstain_flag": np.isnan(summ.anisotropy),
            }
        )

    # Sweep delta
    for delta in sector_widths:
        radar = compute_rsp_radar(r, theta, y, B=ref_B, delta_deg=delta, min_fg_sector=ref_min_fg)
        prof = interpolate_profile(radar.centers, radar.rsp, common_grid)
        prof /= np.linalg.norm(prof) if np.linalg.norm(prof) > 0 else 1.0
        sim = 1.0 - np.sqrt(np.mean((prof - ref_prof) ** 2))
        summ = compute_scalar_summaries(radar)
        rep_results.append(
            {
                "dataset_id": f"{shape}_{etype}_N{n}",
                "shape": shape,
                "type": etype,
                "N": n,
                "seed": seed,
                "param": "sector_width",
                "value": delta,
                "anisotropy": summ.anisotropy,
                "coverage": np.mean(~np.isnan(radar.rsp)),
                "similarity": sim,
                "abstain_flag": np.isnan(summ.anisotropy),
            }
        )
    return rep_results


def run(outdir: Path, config: dict) -> dict:
    """
    Parameter sensitivity and recommended defaults.
    """
    n_reps = config.get("n_reps", 10)
    n_workers = config.get("n_workers", -1)
    shapes = ["disk", "crescent"]
    types = ["rim", "wedge", "rim+wedge"]
    n_regimes = [2000, 10000]

    # Parameter sweeps
    grid_sizes = [60, 90, 180, 360]
    sector_widths = [15, 30, 45]

    # Reference settings
    ref_B = 360
    ref_delta = 20.0
    ref_min_fg = 10

    common_grid = np.linspace(-np.pi, np.pi, 360, endpoint=False)

    print("Running parameter sensitivity sweep...")

    tasks = []
    for shape in shapes:
        for etype in types:
            for n in n_regimes:
                for i in range(n_reps):
                    tasks.append((i, shape, etype, n))

    results_nested = Parallel(n_jobs=n_workers)(
        delayed(simulate_robustness_rep)(
            i,
            shape,
            etype,
            n,
            config,
            ref_B,
            ref_delta,
            ref_min_fg,
            grid_sizes,
            sector_widths,
            common_grid,
        )
        for i, shape, etype, n in tqdm(tasks, desc="Robustness")
    )

    results = [item for sublist in results_nested for item in sublist]

    df = pd.DataFrame(results)
    df.to_csv(outdir / "tables" / "param_sweep_runs.csv", index=False)

    # Figure R1: Parameter sensitivity heatmap (or line plot)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.lineplot(data=df[df["param"] == "theta_grid_size"], x="value", y="similarity", hue="type")
    plt.title("Sensitivity to Grid Size (B)")
    plt.subplot(1, 2, 2)
    sns.lineplot(data=df[df["param"] == "sector_width"], x="value", y="similarity", hue="type")
    plt.title("Sensitivity to Sector Width (delta)")
    plt.tight_layout()
    plt.savefig(outdir / "figures" / "figR1_param_sensitivity.png")
    plt.close()

    # Default constants table
    defaults = [
        {
            "Parameter": "theta_grid_size (B)",
            "Default": "360",
            "Rationale": "Provides high angular resolution while maintaining sector support.",
        },
        {
            "Parameter": "sector_width (delta)",
            "Default": "20.0",
            "Rationale": "Balances local specificity with statistical power per sector.",
        },
        {
            "Parameter": "min_per_sector_support",
            "Default": "10",
            "Rationale": "Ensures stable Wasserstein distance estimation.",
        },
        {
            "Parameter": "min_adequate_sector_fraction",
            "Default": "0.5",
            "Rationale": "Ensures the profile covers enough of the footprint to be representative.",
        },
    ]
    pd.DataFrame(defaults).to_csv(outdir / "tables" / "default_constants_table.csv", index=False)

    return {"robustness_summary": "Parameter sweep completed."}
