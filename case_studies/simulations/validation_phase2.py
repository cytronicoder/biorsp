import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from biorsp import (
    compute_rsp_radar,
    compute_vantage,
    plot_embedding,
    plot_radar,
    polar_coordinates,
)
from biorsp.simulations.generator import simulate_dataset

# Set plotting defaults
plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42})


def compute_rsp(coords, labels):
    """Helper to convert coords to polar and compute RSP."""
    v = compute_vantage(coords)
    r, theta = polar_coordinates(coords, v)
    res = compute_rsp_radar(r, theta, labels)
    # Calculate anisotropy (RMS of non-NaN RSP)
    valid = ~np.isnan(res.rsp)
    anisotropy = np.sqrt(np.mean(res.rsp[valid] ** 2)) if valid.any() else 0.0
    # Return a dict for easy access
    return {
        "rsp": np.nanmean(res.rsp),  # Mean RSP as a summary
        "anisotropy": anisotropy,
        "raw_rsp": res.rsp,
        "radar_res": res,
    }


def run_null_calibration(
    n_sims=100, n_points=1000, output_dir="results/simulations_phase2/null_calibration"
):
    """Verify RSP is centered at 0 for null distributions across shapes."""
    os.makedirs(output_dir, exist_ok=True)
    shapes = ["disk", "ellipse", "annulus", "crescent", "two_lobe", "blob"]
    results = []

    print("Running Null Calibration...")
    raw_dir = os.path.join(output_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    for shape in shapes:
        for i in tqdm(range(n_sims), desc=f"Shape: {shape}"):
            data = simulate_dataset(n_points=n_points, shape=shape, enrichment_type="null", seed=i)
            rsp_res = compute_rsp(data["coords"], data["labels"])
            # Save per-run raw RSP as CSV
            # Use radar centers from radar_res if available
            radar = rsp_res.get("radar_res", None)
            if radar is not None:
                centers_use = radar.centers
            else:
                centers_use = np.arange(len(rsp_res["raw_rsp"]))
            df_r = pd.DataFrame(
                {
                    "center_rad": centers_use,
                    "center_deg": np.degrees(centers_use),
                    "rsp": rsp_res["raw_rsp"],
                }
            )
            df_r.to_csv(os.path.join(raw_dir, f"null_{shape}_{i}_radar.csv"), index=False)
            results.append(
                {
                    "shape": shape,
                    "seed": i,
                    "rsp": rsp_res["rsp"],
                    "anisotropy": rsp_res["anisotropy"],
                    "radar_csv": os.path.relpath(
                        os.path.join(raw_dir, f"null_{shape}_{i}_radar.csv"), start=output_dir
                    ),
                }
            )

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "null_results.csv"), index=False)

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df, x="shape", y="rsp", inner="quart")
    plt.axhline(0, color="red", linestyle="--")
    plt.title("RSP Null Calibration (Expected: 0)")
    out_pdf = os.path.join(output_dir, "null_calibration_violin.pdf")
    out_png = os.path.join(output_dir, "null_calibration_violin.png")
    plt.savefig(out_pdf)
    plt.savefig(out_png)
    plt.close()


def run_power_analysis(n_sims=50, n_points=1000, output_dir="results/simulations_phase2/power"):
    """Sensitivity to enrichment strength (noise_sigma)."""
    os.makedirs(output_dir, exist_ok=True)
    noise_levels = [0.01, 0.05, 0.1, 0.2, 0.4, 0.8]
    enrichments = ["rim", "core", "wedge"]
    results = []

    print("Running Power Analysis...")
    raw_dir = os.path.join(output_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    for e in enrichments:
        for noise in noise_levels:
            for i in range(n_sims):
                data = simulate_dataset(
                    n_points=n_points, shape="disk", enrichment_type=e, noise_sigma=noise, seed=i
                )
                rsp_res = compute_rsp(data["coords"], data["labels"])
                # Save raw radar
                radar = rsp_res.get("radar_res", None)
                if radar is not None:
                    centers = radar.centers
                else:
                    centers = np.arange(len(rsp_res["raw_rsp"]))
                df_r = pd.DataFrame(
                    {
                        "center_rad": centers,
                        "center_deg": np.degrees(centers),
                        "rsp": rsp_res["raw_rsp"],
                    }
                )
                df_r.to_csv(
                    os.path.join(raw_dir, f"power_{e}_noise{noise}_{i}_radar.csv"), index=False
                )
                results.append(
                    {
                        "enrichment": e,
                        "noise": noise,
                        "seed": i,
                        "rsp": rsp_res["rsp"],
                        "anisotropy": rsp_res["anisotropy"],
                        "radar_csv": os.path.relpath(
                            os.path.join(raw_dir, f"power_{e}_noise{noise}_{i}_radar.csv"),
                            start=output_dir,
                        ),
                    }
                )

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "power_results.csv"), index=False)

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.lineplot(
        data=df[df["enrichment"].isin(["rim", "core"])],
        x="noise",
        y="rsp",
        hue="enrichment",
        ax=axes[0],
    )
    axes[0].set_title("RSP Sensitivity (Radial)")
    sns.lineplot(data=df[df["enrichment"] == "wedge"], x="noise", y="anisotropy", ax=axes[1])
    axes[1].set_title("Anisotropy Sensitivity (Angular)")
    plt.tight_layout()
    out_pdf = os.path.join(output_dir, "power_analysis.pdf")
    out_png = os.path.join(output_dir, "power_analysis.png")
    plt.savefig(out_pdf)
    plt.savefig(out_png)
    plt.close()


def run_robustness_test(
    n_sims=30, n_points=1000, output_dir="results/simulations_phase2/robustness"
):
    """Robustness to density gradients and distortions."""
    os.makedirs(output_dir, exist_ok=True)
    models = ["uniform", "radial_center", "radial_rim", "angular_bias"]
    distortions = ["none", "swirl", "anisotropic"]
    results = []

    print("Running Robustness Tests...")
    raw_dir = os.path.join(output_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    for model in models:
        for dist in distortions:
            for i in range(n_sims):
                # Test on a 'rim' enrichment - should stay positive
                data = simulate_dataset(
                    n_points=n_points,
                    shape="disk",
                    enrichment_type="rim",
                    density_model=model,
                    distortion=dist,
                    seed=i,
                )
                rsp_res = compute_rsp(data["coords"], data["labels"])
                # Save raw radar
                radar = rsp_res.get("radar_res", None)
                if radar is not None:
                    centers = radar.centers
                else:
                    centers = np.arange(len(rsp_res["raw_rsp"]))
                df_r = pd.DataFrame(
                    {
                        "center_rad": centers,
                        "center_deg": np.degrees(centers),
                        "rsp": rsp_res["raw_rsp"],
                    }
                )
                df_r.to_csv(
                    os.path.join(raw_dir, f"robust_{model}_{dist}_{i}_radar.csv"), index=False
                )
                results.append(
                    {
                        "density": model,
                        "distortion": dist,
                        "seed": i,
                        "rsp": rsp_res["rsp"],
                        "radar_csv": os.path.relpath(
                            os.path.join(raw_dir, f"robust_{model}_{dist}_{i}_radar.csv"),
                            start=output_dir,
                        ),
                    }
                )

    df = pd.DataFrame(results)

    summary_rows = []
    outlier_rows = []
    grouped = df.groupby(["density", "distortion"])
    for (density, distortion), grp in grouped:
        vals = grp["rsp"].dropna()
        n = int(len(vals))
        if n == 0:
            continue
        q1 = float(vals.quantile(0.25))
        median = float(vals.median())
        q3 = float(vals.quantile(0.75))
        iqr = q3 - q1
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr
        min_val = float(vals.min())
        max_val = float(vals.max())
        outliers = grp[(grp["rsp"] < lower_fence) | (grp["rsp"] > upper_fence)]
        out_vals = outliers["rsp"].tolist()
        summary_rows.append(
            {
                "density": density,
                "distortion": distortion,
                "n": n,
                "min": min_val,
                "q1": q1,
                "median": median,
                "q3": q3,
                "max": max_val,
                "iqr": iqr,
                "lower_fence": lower_fence,
                "upper_fence": upper_fence,
                "n_outliers": int(len(out_vals)),
                "outliers": ";".join(map(str, out_vals)),
            }
        )
        for _, r in outliers.iterrows():
            outlier_rows.append(
                {
                    "density": density,
                    "distortion": distortion,
                    "seed": r.get("seed", None),
                    "rsp": r["rsp"],
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(output_dir, "robustness_summary.csv"), index=False)
    outliers_df = pd.DataFrame(outlier_rows)
    outliers_df.to_csv(os.path.join(output_dir, "robustness_outliers.csv"), index=False)

    # Plotting
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x="density", y="rsp", hue="distortion")
    plt.axhline(0, color="black", alpha=0.3)
    plt.title("Robustness of 'Rim' Enrichment (Expected: >0)")
    out_pdf = os.path.join(output_dir, "robustness_results.pdf")
    out_png = os.path.join(output_dir, "robustness_results.png")
    plt.savefig(out_pdf)
    plt.savefig(out_png)
    plt.close()


def run_failure_modes(output_dir="results/simulations_phase2/failure_modes"):
    """Visualizing failure modes in non-convex shapes."""
    os.makedirs(output_dir, exist_ok=True)

    # Case 1: Two-lobe with enrichment in one lobe
    data = simulate_dataset(
        n_points=2000,
        shape="two_lobe",
        enrichment_type="patch",
        patch_center=np.array([-0.75, 0]),
        patch_radius=0.4,
        seed=42,
    )

    rsp_res = compute_rsp(data["coords"], data["labels"])

    fig = plt.figure(figsize=(12, 5))
    ax_embed = fig.add_subplot(121)
    # Standardized embedding: background grey, foreground red; show vantage
    plot_embedding(
        data["coords"],
        c=data["labels"],
        ax=ax_embed,
        title="Two-Lobe Patch Embedding",
        fg_color="red",
        bg_color="lightgrey",
        show_vantage=True,
    )

    # Create a polar axis for plot_radar (no extra cartesian axis)
    ax_polar = fig.add_subplot(122, projection="polar")
    plot_radar(rsp_res["radar_res"], ax=ax_polar, title="Two-Lobe Patch Radar")

    plt.tight_layout()
    out_pdf = os.path.join(output_dir, "failure_two_lobe.pdf")
    out_png = os.path.join(output_dir, "failure_two_lobe.png")
    plt.savefig(out_pdf)
    plt.savefig(out_png)
    # Save coordinates + labels + scores for debugging
    coords_df = pd.DataFrame(
        {
            "x": data["coords"][..., 0],
            "y": data["coords"][..., 1],
            "label": data["labels"],
            "score": data["scores"],
        }
    )
    coords_df.to_csv(os.path.join(output_dir, "failure_two_lobe_points.csv"), index=False)
    # Save radar (centers + rsp)
    radar = rsp_res["radar_res"]
    df_r = pd.DataFrame(
        {"center_rad": radar.centers, "center_deg": np.degrees(radar.centers), "rsp": radar.rsp}
    )
    df_r.to_csv(os.path.join(output_dir, "failure_two_lobe_radar.csv"), index=False)
    plt.close()

    # Case 2: Crescent with 'core' enrichment
    data = simulate_dataset(n_points=2000, shape="crescent", enrichment_type="core", seed=42)
    rsp_res = compute_rsp(data["coords"], data["labels"])

    fig = plt.figure(figsize=(12, 5))
    ax_embed = fig.add_subplot(121)
    plot_embedding(
        data["coords"],
        c=data["labels"],
        ax=ax_embed,
        title="Crescent Core Embedding",
        fg_color="red",
        bg_color="lightgrey",
        show_vantage=True,
    )

    ax_polar = fig.add_subplot(122, projection="polar")
    plot_radar(rsp_res["radar_res"], ax=ax_polar, title="Crescent Core Radar")

    plt.tight_layout()
    out_pdf = os.path.join(output_dir, "failure_crescent.pdf")
    out_png = os.path.join(output_dir, "failure_crescent.png")
    plt.savefig(out_pdf)
    plt.savefig(out_png)
    coords_df = pd.DataFrame(
        {
            "x": data["coords"][..., 0],
            "y": data["coords"][..., 1],
            "label": data["labels"],
            "score": data["scores"],
        }
    )
    coords_df.to_csv(os.path.join(output_dir, "failure_crescent_points.csv"), index=False)
    radar = rsp_res["radar_res"]
    df_r = pd.DataFrame(
        {"center_rad": radar.centers, "center_deg": np.degrees(radar.centers), "rsp": radar.rsp}
    )
    df_r.to_csv(os.path.join(output_dir, "failure_crescent_radar.csv"), index=False)
    plt.close()


if __name__ == "__main__":
    run_null_calibration()
    run_power_analysis()
    run_robustness_test()
    run_failure_modes()
    print("Phase 2 Validation Complete. Results in results/simulations_phase2/")
