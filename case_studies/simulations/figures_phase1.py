import json
import os
import subprocess
import time
from typing import Any, Dict, List

import fig_utils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from biorsp import (
    compute_rsp_radar,
    compute_scalar_summaries,
    compute_vantage,
    plot_embedding,
    plot_radar,
    polar_coordinates,
)
from biorsp.simulations.generator import simulate_dataset

# Constants
RESULTS_DIR = "results/simulations_phase1"
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
TABLE_DIR = os.path.join(RESULTS_DIR, "tables")


def run_biorsp_on_sim(
    coords: np.ndarray, scores: np.ndarray, q: float = 0.1, biorsp_params: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Wrapper to run BioRSP on simulated data.
    q is the fraction of points to label as foreground (e.g., 0.1 for top decile).
    """
    if biorsp_params is None:
        biorsp_params = {"B": 120, "delta_deg": 40.0}

    # Vantage point
    v = compute_vantage(coords)
    r, theta = polar_coordinates(coords, v)

    # Foreground labels (top q)
    threshold = np.quantile(scores, 1.0 - q)
    y = (scores >= threshold).astype(int)

    # Compute radar
    radar = compute_rsp_radar(r, theta, y, **biorsp_params)

    # Compute summaries
    summaries = compute_scalar_summaries(radar)

    # Coverage: fraction of non-NaN sectors
    coverage = np.mean(~np.isnan(radar.rsp))

    return {
        "radar": radar,
        "summaries": summaries,
        "coverage": coverage,
        "params": biorsp_params,
        "y": y,
        "vantage": v,
    }


def generate_fig_s1_s2(examples: List[Dict[str, Any]]):
    """Generate FIG S1 (Cartoons) and FIG S2 (Profiles)."""
    fig_s1, axes_s1 = plt.subplots(2, 3, figsize=(15, 10))
    fig_s2, axes_s2 = plt.subplots(2, 3, figsize=(15, 10), subplot_kw={"projection": "polar"})

    for i, ex in enumerate(examples):
        row, col = i // 3, i % 3
        ax_s1 = axes_s1[row, col]
        ax_s2 = axes_s2[row, col]

        # Run BioRSP
        res = run_biorsp_on_sim(ex["coords"], ex["scores"], q=ex["metadata"]["quantile"])

        # FIG S1: Scatter (standardized colors; show vantage)
        coords = ex["coords"]
        y = res["y"]
        plot_embedding(
            coords,
            c=y,
            ax=ax_s1,
            title=None,
            fg_color="red",
            bg_color="lightgrey",
            show_vantage=True,
        )

        title = f"{ex['metadata']['shape'].capitalize()} - {ex['metadata']['enrichment_type'].capitalize()}"
        ax_s1.set_title(title)
        subtitle = f"N={ex['metadata']['n_points']}, q={ex['metadata']['quantile']}, noise={ex['metadata'].get('noise_sigma', 0.3)}"
        ax_s1.text(0.5, -0.1, subtitle, transform=ax_s1.transAxes, ha="center", fontsize=8)
        fig_utils.clean_axis(ax_s1)

        # FIG S2: Radar Profile
        radar = res["radar"]
        summaries = res["summaries"]

        # Use BioRSP's plot_radar for consistent styling
        plot_radar(radar, ax=ax_s2, title=title, summaries=summaries, show_anchors=True)

        # Add coverage text
        ax_s2.text(1.1, 0.1, f"C={res['coverage']:.2f}", transform=ax_s2.transAxes, fontsize=10)

        # Save per-example CSVs for reproducibility and inspection
        csv_dir = os.path.join(FIG_DIR, "csv")
        os.makedirs(csv_dir, exist_ok=True)
        # Points CSV
        df_pts = pd.DataFrame(
            {"x": coords[:, 0], "y": coords[:, 1], "score": ex["scores"], "label": res["y"]}
        )
        pts_fname = os.path.join(
            csv_dir,
            f"example_{ex['metadata']['shape']}_{ex['metadata']['enrichment_type']}_seed{ex['metadata']['seed']}_points.csv",
        )
        df_pts.to_csv(pts_fname, index=False)
        # Radar CSV
        df_rad = pd.DataFrame(
            {"center_rad": radar.centers, "center_deg": np.degrees(radar.centers), "rsp": radar.rsp}
        )
        rad_fname = os.path.join(
            csv_dir,
            f"example_{ex['metadata']['shape']}_{ex['metadata']['enrichment_type']}_seed{ex['metadata']['seed']}_radar.csv",
        )
        df_rad.to_csv(rad_fname, index=False)
        # Append references to the example results table
        ex["radar_csv"] = os.path.relpath(rad_fname, start=FIG_DIR)
        ex["points_csv"] = os.path.relpath(pts_fname, start=FIG_DIR)

    fig_s1.tight_layout()
    fig_s2.tight_layout()

    fig_utils.save_fig(fig_s1, FIG_DIR, "figS1_simulation_cartoons")
    fig_utils.save_fig(fig_s2, FIG_DIR, "figS2_biorsp_profiles")

    # Save an index CSV linking examples to their saved CSVs
    csv_dir = os.path.join(FIG_DIR, "csv")
    os.makedirs(csv_dir, exist_ok=True)
    rows = []
    for ex in examples:
        meta = ex.get("metadata", {})
        rows.append(
            {
                "shape": meta.get("shape"),
                "enrichment": meta.get("enrichment_type"),
                "seed": meta.get("seed"),
                "points_csv": ex.get("points_csv", ""),
                "radar_csv": ex.get("radar_csv", ""),
            }
        )
    pd.DataFrame(rows).to_csv(os.path.join(csv_dir, "phase1_example_index.csv"), index=False)


def generate_fig_s3():
    """Generate FIG S3 (Anisotropy distributions)."""
    types = ["null", "rim", "wedge"]
    n_reps = 30
    results = []

    for t in types:
        for seed in range(n_reps):
            data = simulate_dataset(
                n_points=4000,
                shape="disk",
                enrichment_type=t,
                quantile=0.1,
                seed=seed,
                noise_sigma=0.3,  # Passing as kwarg to metadata
            )
            res = run_biorsp_on_sim(data["coords"], data["scores"], q=0.1)

            results.append(
                {
                    "shape": "disk",
                    "type": t,
                    "seed": seed,
                    "N": 4000,
                    "noise_sigma": 0.3,
                    "q": 0.1,
                    "anisotropy": res["summaries"].anisotropy,
                    "coverage": res["coverage"],
                    "peak_angle": res["summaries"].peak_extremal_angle,
                }
            )

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(TABLE_DIR, "aniso_summary_phase1.csv"), index=False)

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 6))

    data_to_plot = [df[df["type"] == t]["anisotropy"].dropna() for t in types]
    ax.boxplot(
        data_to_plot,
        tick_labels=[t.capitalize() for t in types],
        patch_artist=True,
        boxprops=dict(facecolor="white", color="black"),
        medianprops=dict(color="red"),
    )

    # Overlay points
    for i, t in enumerate(types):
        y = df[df["type"] == t]["anisotropy"].dropna()
        x = np.random.normal(i + 1, 0.04, size=len(y))
        ax.scatter(x, y, alpha=0.5, s=10, color="blue")

    ax.set_ylabel("Anisotropy (A)")
    ax.set_title("Anisotropy Distribution by Enrichment Type")

    fig_utils.save_fig(fig, FIG_DIR, "figS3_anisotropy_distributions")


def main():
    fig_utils.setup_style()
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TABLE_DIR, exist_ok=True)

    # Define examples for S1/S2
    # Row 1: Disk (Null, Rim, Wedge)
    # Row 2: Ellipse (Core, Rim+Wedge, Null)
    example_configs = [
        {"shape": "disk", "type": "null", "seed": 0},
        {"shape": "disk", "type": "rim", "seed": 1},
        {"shape": "disk", "type": "wedge", "seed": 2},
        {"shape": "ellipse", "type": "core", "seed": 0},
        {"shape": "ellipse", "type": "rim+wedge", "seed": 1},
        {"shape": "ellipse", "type": "null", "seed": 2},
    ]

    examples = []
    example_results = []
    for config in example_configs:
        data = simulate_dataset(
            n_points=2000,
            shape=config["shape"],
            enrichment_type=config["type"],
            quantile=0.1,
            seed=config["seed"],
            noise_sigma=0.3,
        )
        examples.append(data)

        # Run BioRSP for the table
        res = run_biorsp_on_sim(data["coords"], data["scores"], q=0.1)
        example_results.append(
            {
                "shape": config["shape"],
                "type": config["type"],
                "seed": config["seed"],
                "N": 2000,
                "noise_sigma": 0.3,
                "q": 0.1,
                "anisotropy": res["summaries"].anisotropy,
                "coverage": res["coverage"],
                "peak_angle": res["summaries"].peak_extremal_angle,
            }
        )

    # Save example table
    pd.DataFrame(example_results).to_csv(
        os.path.join(TABLE_DIR, "phase1_examples_table.csv"), index=False
    )

    # Generate FIG S1 and S2
    generate_fig_s1_s2(examples)

    # Generate FIG S3
    generate_fig_s3()

    # Write Captions
    captions = """FIG S1: Simulation cartoons showing footprint shapes and foreground enrichment patterns.
Top row shows disk footprints with Null, Rim, and Wedge enrichments.
Bottom row shows ellipse footprints with Core, Rim+Wedge, and Null enrichments.
Foreground (red) is defined as the top decile (q=0.1) of simulated scores.
Embeddings are synthetic and coordinates are held fixed.

FIG S2: BioRSP radar profiles for the representative patterns shown in FIG S1.
Radar profiles R(theta) are plotted in polar coordinates. Angles measured with 0° at right (+x axis), degrees increase counter-clockwise.
Anisotropy (A) and coverage (C) metrics are shown for each pattern.
Rim patterns show negative RSP (distal bias), while Core patterns show positive RSP (proximal bias).
Wedge patterns exhibit high anisotropy due to angular localization.

FIG S3: Distribution of anisotropy (A) across 30 replicates for Null, Rim, and Wedge patterns.
Null patterns consistently show low anisotropy, while structured patterns (Rim, Wedge) show significantly higher values.
Wedge patterns exhibit the highest anisotropy due to their strong angular bias.
N=2000 points, q=0.1 foreground threshold, noise_sigma=0.3.
"""
    with open(os.path.join(FIG_DIR, "captions_phase1.txt"), "w") as f:
        f.write(captions)

    # 5. Write Metadata
    git_hash = "unknown"
    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    except Exception:
        pass

    metadata = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": git_hash,
        "generator_params": {"n_points": 2000, "noise_sigma": 0.3, "quantile": 0.1},
        "biorsp_params": {"B": 120, "delta_deg": 40.0, "vantage": "geometric_median"},
        "seeds": {"examples": [c["seed"] for c in example_configs], "replicates": list(range(30))},
    }
    with open(os.path.join(RESULTS_DIR, "run_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)


if __name__ == "__main__":
    main()
