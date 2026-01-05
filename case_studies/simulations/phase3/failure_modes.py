from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from tqdm import tqdm

import biorsp
from biorsp import (
    compute_vantage,
    plot_radar,
)
from biorsp.simulations import simulate_dataset


def simulate_failure_rep(i, case, base_seed, outdir):
    seed = base_seed + i
    q_val = 1.0 - case["q"]

    data = simulate_dataset(
        n_points=case["n"],
        shape=case["shape"],
        enrichment_type=case["etype"],
        quantile=q_val,
        seed=seed,
        **{k: v for k, v in case.items() if k not in ["name", "n", "q", "shape", "etype"]},
    )

    coords = data["coords"]
    y = data["labels"]
    v = compute_vantage(coords)
    rel = coords - v
    r = np.linalg.norm(rel, axis=1)
    theta = np.arctan2(rel[:, 1], rel[:, 0])

    # Use formal adequacy check
    config = biorsp.BioRSPConfig(B=360, delta_deg=20)
    report = biorsp.assess_adequacy(r, theta, y, config=config)
    radar = biorsp.compute_rsp_radar(r, theta, y, config=config, adequacy=report)
    summaries = biorsp.compute_scalar_summaries(radar)

    abstain = not report.is_adequate

    # Save one representative example for Figure F2
    if i == 0:
        case_dir = outdir / "figures" / "failure_examples"
        case_dir.mkdir(parents=True, exist_ok=True)

        fig = plt.figure(figsize=(12, 5))
        # Scatter
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.scatter(coords[:, 0], coords[:, 1], c=y, s=1, cmap="coolwarm", alpha=0.5)
        ax1.set_title(f"Scatter: {case['name']}\n(Adequate: {report.is_adequate})")
        ax1.axis("equal")

        # Radar
        ax2 = fig.add_subplot(1, 2, 2, projection="polar")
        plot_radar(radar, ax=ax2, title=f"Radar: {case['name']}\nReason: {report.reason}")

        plt.tight_layout()
        plt.savefig(case_dir / f"example_{case['name']}.png")
        plt.close()

    return {
        "case": case["name"],
        "seed": seed,
        "anisotropy": summaries.anisotropy if not abstain else np.nan,
        "coverage": report.adequacy_fraction,
        "abstain_flag": abstain,
        "abstain_reason": report.reason,
    }


def run(outdir: Path, config: dict) -> dict:
    """
    Formal abstention / adequacy behavior evaluation.
    """
    n_reps = config.get("n_reps", 50)
    n_workers = config.get("n_workers", -1)
    base_seed = config.get("seed", 42)

    stress_cases = [
        {"name": "sparse_fg", "n": 2000, "q": 0.98, "shape": "disk", "etype": "wedge"},
        {"name": "tiny_n", "n": 300, "q": 0.9, "shape": "disk", "etype": "wedge"},
        {"name": "disconnected", "n": 2000, "q": 0.9, "shape": "two_lobe", "etype": "wedge"},
        {"name": "hole", "n": 2000, "q": 0.9, "shape": "annulus", "etype": "wedge", "r_in": 0.8},
        {
            "name": "density_bias",
            "n": 2000,
            "q": 0.9,
            "shape": "disk",
            "etype": "wedge",
            "density_model": "angular_bias",
            "density_strength": 2.0,
        },
    ]

    print("Running failure mode stress tests...")

    tasks = []
    for case in stress_cases:
        for i in range(n_reps):
            tasks.append((i, case))

    results = Parallel(n_jobs=n_workers)(
        delayed(simulate_failure_rep)(i, case, base_seed, outdir)
        for i, case in tqdm(tasks, desc="Failure Modes")
    )

    df = pd.DataFrame(results)
    df.to_csv(outdir / "tables" / "failure_modes_runs.csv", index=False)

    # Figure F1: Abstention rates
    plt.figure(figsize=(10, 6))
    abstain_rates = df.groupby("case")["abstain_flag"].mean().reset_index()
    sns.barplot(data=abstain_rates, x="case", y="abstain_flag")
    plt.ylabel("Abstention Rate")
    plt.title("Abstention Rates by Stress Condition")
    plt.savefig(outdir / "figures" / "figF1_abstention_rates.png")
    plt.close()

    # Figure F2: Representative failure examples (Composite)
    plt.figure(figsize=(15, 5))
    example_cases = ["sparse_fg", "tiny_n", "density_bias"]
    for i, case_name in enumerate(example_cases):
        img_path = outdir / "figures" / "failure_examples" / f"example_{case_name}.png"
        if img_path.exists():
            plt.subplot(1, 3, i + 1)
            img = plt.imread(img_path)
            plt.imshow(img)
            plt.axis("off")
            plt.title(case_name)

    plt.tight_layout()
    plt.savefig(outdir / "figures" / "figF2_failure_examples.png")
    plt.close()

    return {"failure_summary": "Failure mode analysis completed."}
