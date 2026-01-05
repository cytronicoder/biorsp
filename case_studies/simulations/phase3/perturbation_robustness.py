import argparse
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
)
from biorsp.simulations import simulate_dataset


def align_profiles(p1, p2):
    """Align p2 to p1 by circular shift to maximize correlation."""
    # Handle NaNs by filling with 0 for alignment purposes
    v1 = np.nan_to_num(p1)
    v2 = np.nan_to_num(p2)

    best_corr = -1
    best_shift = 0
    n = len(v1)
    for s in range(n):
        shifted = np.roll(v2, s)
        corr = np.corrcoef(v1, shifted)[0, 1]
        if corr > best_corr:
            best_corr = corr
            best_shift = s
    return best_corr, best_shift


def simulate_perturbation_rep(rep, seed):
    rep_results = []
    # Base dataset: Disk with Rim
    data_base = simulate_dataset(
        shape="disk", enrichment_type="rim", n_points=5000, quantile=0.1, seed=seed + rep
    )
    coords_base = data_base["coords"]
    y_base = data_base["labels"]
    r_base = np.linalg.norm(coords_base, axis=1)
    theta_base = np.arctan2(coords_base[:, 1], coords_base[:, 0])
    radar_base = compute_rsp_radar(r_base, theta_base, y_base)
    summ_base = compute_scalar_summaries(radar_base)

    # Rotations
    for angle_deg in [0, 90, 180, 270]:
        angle_rad = np.deg2rad(angle_deg)
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        R = np.array([[c, -s], [s, c]])
        coords_rot = coords_base @ R.T
        r_rot = np.linalg.norm(coords_rot, axis=1)
        theta_rot = np.arctan2(coords_rot[:, 1], coords_rot[:, 0])
        radar_rot = compute_rsp_radar(r_rot, theta_rot, y_base)
        summ_rot = compute_scalar_summaries(radar_rot)

        corr, _ = align_profiles(radar_base.rsp, radar_rot.rsp)
        rep_results.append(
            {
                "rep": rep,
                "type": "rotation",
                "param": angle_deg,
                "anisotropy": summ_rot.anisotropy,
                "similarity": corr,
                "aniso_diff": np.abs(summ_rot.anisotropy - summ_base.anisotropy),
            }
        )

    # Scales
    for scale in [0.5, 1.0, 2.0]:
        coords_scale = coords_base * scale
        r_scale = np.linalg.norm(coords_scale, axis=1)
        theta_scale = np.arctan2(coords_scale[:, 1], coords_scale[:, 0])
        radar_scale = compute_rsp_radar(r_scale, theta_scale, y_base)
        summ_scale = compute_scalar_summaries(radar_scale)

        corr, _ = align_profiles(radar_base.rsp, radar_scale.rsp)
        rep_results.append(
            {
                "rep": rep,
                "type": "scale",
                "param": scale,
                "anisotropy": summ_scale.anisotropy,
                "similarity": corr,
                "aniso_diff": np.abs(summ_scale.anisotropy - summ_base.anisotropy),
            }
        )

    # Distortions (Warping)
    for dist in ["none", "swirl", "anisotropic"]:
        data_dist = simulate_dataset(
            shape="disk",
            enrichment_type="rim",
            n_points=5000,
            quantile=0.1,
            distortion=dist,
            seed=seed + rep,
        )
        coords_dist = data_dist["coords"]
        y_dist = data_dist["labels"]
        r_dist = np.linalg.norm(coords_dist, axis=1)
        theta_dist = np.arctan2(coords_dist[:, 1], coords_dist[:, 0])
        radar_dist = compute_rsp_radar(r_dist, theta_dist, y_dist)
        summ_dist = compute_scalar_summaries(radar_dist)

        corr, _ = align_profiles(radar_base.rsp, radar_dist.rsp)
        rep_results.append(
            {
                "rep": rep,
                "type": "distortion",
                "param": dist,
                "anisotropy": summ_dist.anisotropy,
                "similarity": corr,
                "aniso_diff": np.abs(summ_dist.anisotropy - summ_base.anisotropy),
            }
        )
    return rep_results


def run(outdir, config):
    outdir = Path(outdir)
    tables_dir = outdir / "tables"
    figs_dir = outdir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    n_reps = config.get("n_reps", 5)
    n_workers = config.get("n_workers", -1)
    seed = config.get("seed", 42)

    print("Running Perturbation Robustness sweep...")

    results_nested = Parallel(n_jobs=n_workers)(
        delayed(simulate_perturbation_rep)(rep, seed)
        for rep in tqdm(range(n_reps), desc="Perturbation Robustness")
    )

    results = [item for sublist in results_nested for item in sublist]

    df = pd.DataFrame(results)
    df.to_csv(tables_dir / "perturbation_robustness.csv", index=False)

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Similarity (Profile Invariance)
    sns.boxplot(data=df, x="type", y="similarity", ax=axes[0])
    axes[0].set_title("Profile Similarity after Alignment")
    axes[0].set_ylim(0.8, 1.05)

    # Anisotropy Stability
    sns.boxplot(data=df, x="type", y="anisotropy", ax=axes[1])
    axes[1].set_title("Anisotropy Score Stability")

    plt.tight_layout()
    plt.savefig(figs_dir / "figR2_perturbation_robustness.png")
    plt.close()

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="results/simulations_phase3")
    parser.add_argument("--n_reps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.outdir, {"n_reps": args.n_reps, "seed": args.seed})
