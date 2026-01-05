import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from tqdm import tqdm

import biorsp
from biorsp.simulations import simulate_dataset


def simulate_power_rep(rep, N, q, seed, n_perm, alpha):
    # Generate a wedge enrichment (standard signal)
    data = simulate_dataset(
        shape="disk",
        enrichment_type="wedge",
        n_points=N,
        quantile=q,
        noise_sigma=0.3,
        seed=seed + rep + int(N) + int(q * 100),
    )
    coords = data["coords"]
    y = data["labels"]

    r = np.linalg.norm(coords, axis=1)
    theta = np.arctan2(coords[:, 1], coords[:, 0])

    config = biorsp.BioRSPConfig(
        B=360,
        delta_deg=20,
        min_fg_total=10,
        min_adequacy_fraction=0.01,
        n_permutations=n_perm,
        seed=seed,
    )

    # Adequacy check - use permissive settings for power simulation to see where signal emerges
    report = biorsp.assess_adequacy(r, theta, y, config=config)
    abstain = not report.is_adequate

    anisotropy = np.nan
    p_val = np.nan
    detected = False

    if not abstain:
        radar = biorsp.compute_rsp_radar(r, theta, y, config=config, adequacy=report)
        summ = biorsp.compute_scalar_summaries(radar)
        anisotropy = summ.anisotropy
        # Fast p-value - pass the same adequacy report to ensure consistent masking
        p_val_res = biorsp.compute_p_value(r, theta, y, config=config, adequacy=report)
        p_val = p_val_res.p_value
        detected = p_val <= alpha

    return {
        "N": N,
        "q": q,
        "rep": rep,
        "anisotropy": anisotropy,
        "p_value": p_val,
        "abstain": abstain,
        "detected": detected,
        "reason": report.reason if abstain else "none",
    }


def run(outdir, config):
    outdir = Path(outdir)
    tables_dir = outdir / "tables"
    figs_dir = outdir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    n_reps = config.get("n_reps", 10)
    n_workers = config.get("n_workers", -1)
    seed = config.get("seed", 42)

    # Grids
    N_grid = config.get("N_grid", [300, 600, 1200, 2500, 5000, 10000, 20000])
    q_grid = config.get("q_grid", [0.02, 0.05, 0.1, 0.2, 0.4])
    alpha = 0.05
    n_perm = 100

    print("Running Power vs N/q sweep...")

    tasks = [(rep, N, q) for N in N_grid for q in q_grid for rep in range(n_reps)]

    results = Parallel(n_jobs=n_workers)(
        delayed(simulate_power_rep)(rep, N, q, seed, n_perm, alpha)
        for rep, N, q in tqdm(tasks, desc="Power vs N/q")
    )

    df = pd.DataFrame(results)
    df.to_csv(tables_dir / "power_vs_N.csv", index=False)

    # Power Curve (Detection Rate)
    plt.figure(figsize=(10, 6))
    power_df = df.groupby(["N", "q"])["detected"].mean().reset_index()
    sns.lineplot(data=power_df, x="N", y="detected", hue="q", marker="o")
    plt.xscale("log")
    plt.ylim(-0.05, 1.05)
    plt.axhline(alpha, ls="--", color="gray", alpha=0.5, label=f"alpha={alpha}")
    plt.title("Detection Power vs N and Foreground Fraction (q)")
    plt.ylabel("Power (P(p <= alpha))")
    plt.legend(title="q")
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig(figs_dir / "figP1_power_curves.png")
    plt.close()

    # Abstention Rate Overlay
    plt.figure(figsize=(10, 6))
    abstain_df = df.groupby(["N", "q"])["abstain"].mean().reset_index()
    sns.lineplot(data=abstain_df, x="N", y="abstain", hue="q", marker="s", ls="--")
    plt.xscale("log")
    plt.ylim(-0.05, 1.05)
    plt.title("Abstention Rate vs N and Foreground Fraction (q)")
    plt.ylabel("Abstention Rate")
    plt.legend(title="q")
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig(figs_dir / "figP2_abstention_rates.png")
    plt.close()

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="results/simulations_phase3")
    parser.add_argument("--n_reps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run(args.outdir, {"n_reps": args.n_reps, "seed": args.seed})
