from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from biorsp import (
    compute_p_value,
    compute_rsp_radar,
    compute_scalar_summaries,
    compute_vantage,
)
from biorsp.simulations import simulate_dataset


def simulate_calibration_rep(i, shape, density, distortion, n_points, q, n_perm, base_seed):
    seed = base_seed + i
    # Generate null dataset
    data = simulate_dataset(
        n_points=n_points,
        shape=shape,
        density_model=density,
        distortion=distortion,
        enrichment_type="null",
        quantile=1.0 - q,  # simulate_dataset uses quantile as "fraction of foreground"
        seed=seed,
    )

    coords = data["coords"]
    y = data["labels"]

    # Compute vantage and polar coords
    v = compute_vantage(coords)
    rel = coords - v
    r = np.linalg.norm(rel, axis=1)
    theta = np.arctan2(rel[:, 1], rel[:, 0])

    # Run BioRSP
    radar = compute_rsp_radar(r, theta, y)
    summaries = compute_scalar_summaries(radar)

    # Compute p-value
    p_val, _, _, _ = compute_p_value(r, theta, y, n_perm=n_perm, seed=seed)

    abstain = np.isnan(summaries.anisotropy)

    return {
        "shape": shape,
        "density_model": density,
        "distortion": distortion,
        "N": n_points,
        "q": q,
        "seed": seed,
        "anisotropy": summaries.anisotropy,
        "coverage": np.mean(~np.isnan(radar.rsp)),
        "peak_angle": summaries.peak_extremal_angle,
        "abstain_flag": abstain,
        "abstain_reason": "low_coverage" if abstain else "none",
        "p_value": p_val,
        "n_perm": n_perm,
    }


def run(outdir: Path, config: dict) -> dict:
    """
    Null calibration and false positive control.
    """
    n_reps = config.get("n_reps", 50)
    shapes = config.get("shapes", ["disk", "ellipse", "crescent", "annulus"])
    density_models = config.get("density_models", ["uniform", "radial_rim"])
    distortions = config.get("distortions", ["none", "swirl"])
    n_points = config.get("n_points", 2000)
    q = config.get("q", 0.9)
    n_perm = config.get("n_perm", 100)
    base_seed = config.get("seed", 42)
    n_workers = config.get("n_workers", -1)

    results = []

    print(f"Running calibration with {n_reps} replicates per condition...")

    tasks = [
        (i, shape, density, distortion)
        for shape in shapes
        for density in density_models
        for distortion in distortions
        for i in range(n_reps)
    ]

    results = Parallel(n_jobs=n_workers)(
        delayed(simulate_calibration_rep)(
            i, shape, density, distortion, n_points, q, n_perm, base_seed
        )
        for i, shape, density, distortion in tqdm(tasks, desc="Calibration")
    )

    df = pd.DataFrame(results)
    df.to_csv(outdir / "tables" / "calibration_runs.csv", index=False)

    # Summary table
    summary_rows = []
    for (shape, density, distortion), group in df.groupby(["shape", "density_model", "distortion"]):
        non_abstained = group[~group["abstain_flag"]]
        if len(non_abstained) > 0:
            fpr_05 = np.mean(non_abstained["p_value"] <= 0.05)
            fpr_01 = np.mean(non_abstained["p_value"] <= 0.01)
        else:
            fpr_05 = fpr_01 = np.nan

        summary_rows.append(
            {
                "shape": shape,
                "density_model": density,
                "distortion": distortion,
                "fpr_05": fpr_05,
                "fpr_01": fpr_01,
                "abstention_fraction": group["abstain_flag"].mean(),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(outdir / "tables" / "calibration_summary.csv", index=False)

    # Figure C1: Null anisotropy distribution
    plt.figure(figsize=(10, 6))
    for shape in ["disk", "crescent"]:
        if shape in df["shape"].unique():
            subset = df[df["shape"] == shape]["anisotropy"].dropna()
            plt.hist(subset, bins=20, alpha=0.5, label=f"Shape: {shape}")
    plt.xlabel("RMS Anisotropy")
    plt.ylabel("Frequency")
    plt.title("Null Anisotropy Distribution")
    plt.legend()
    plt.savefig(outdir / "figures" / "figC1_null_anisotropy_distribution.png")
    plt.close()

    # Figure C2: P-value QQ plot
    plt.figure(figsize=(6, 6))
    p_vals = df["p_value"].dropna().sort_values()
    if len(p_vals) > 0:
        expected = np.linspace(0, 1, len(p_vals))
        plt.scatter(expected, p_vals, s=10)
        plt.plot([0, 1], [0, 1], "r--")
        plt.xlabel("Expected (Uniform)")
        plt.ylabel("Observed P-value")
        plt.title("P-value Calibration (Null)")
        plt.savefig(outdir / "figures" / "figC2_null_pvalue_qq.png")
    plt.close()

    return {"calibration_summary": summary_df.to_dict()}
