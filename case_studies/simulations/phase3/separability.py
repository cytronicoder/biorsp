import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from biorsp import (
    compute_rsp_radar,
    compute_scalar_summaries,
)
from biorsp.simulations import simulate_dataset


def extract_features(radar, summ):
    """Extract simple features from RSP profile."""
    rsp = np.nan_to_num(radar.rsp)
    return {
        "anisotropy": summ.anisotropy,
        "max_rsp": np.max(rsp),
        "mean_rsp": np.mean(rsp),
        "std_rsp": np.std(rsp),
        "coverage": np.mean(np.isfinite(radar.rsp)),
    }


def simulate_separability_rep(rep, case, seed):
    sim_data = simulate_dataset(
        shape=case["shape"],
        enrichment_type=case["type"],
        n_points=5000,
        quantile=0.1,
        seed=seed + rep,
    )
    coords = sim_data["coords"]
    y = sim_data["labels"]
    r = np.linalg.norm(coords, axis=1)
    theta = np.arctan2(coords[:, 1], coords[:, 0])
    radar = compute_rsp_radar(r, theta, y)
    summ = compute_scalar_summaries(radar)

    features = extract_features(radar, summ)
    features["label"] = case["name"]
    features["is_anisotropic"] = case["name"] != "Null"
    return features


def run(outdir, config):
    outdir = Path(outdir)
    tables_dir = outdir / "tables"
    figs_dir = outdir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    n_reps = config.get("n_reps", 20)
    n_workers = config.get("n_workers", -1)
    seed = config.get("seed", 42)

    # Pattern types to distinguish
    cases = [
        {"name": "Null", "shape": "disk", "type": "null"},
        {"name": "Type I (Rim)", "shape": "disk", "type": "rim"},
        {"name": "Type II (Core)", "shape": "disk", "type": "core"},
        {"name": "Type III (Wedge)", "shape": "disk", "type": "wedge"},
        {"name": "Type IV (Mixed)", "shape": "disk", "type": "rim+wedge"},
    ]

    print("Running Separability Simulations...")

    tasks = []
    for case in cases:
        for rep in range(n_reps):
            tasks.append((rep, case))

    data = Parallel(n_jobs=n_workers)(
        delayed(simulate_separability_rep)(rep, case, seed)
        for rep, case in tqdm(tasks, desc="Separability")
    )

    df = pd.DataFrame(data)
    df.to_csv(tables_dir / "type_separability.csv", index=False)

    # 1. ROC/AUC for Anisotropic vs Null
    plt.figure(figsize=(8, 6))
    y_true = df["is_anisotropic"]
    y_score = df["anisotropy"]
    auc = roc_auc_score(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.plot(fpr, tpr, label=f"BioRSP Anisotropy (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC: Anisotropic vs Null Detection")
    plt.legend()
    plt.savefig(figs_dir / "figS1_roc_auc.png")
    plt.close()

    # 2. Confusion Matrix for Type Classification
    X = df.drop(columns=["label", "is_anisotropic"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)
    clf = RandomForestClassifier(n_estimators=100, random_state=seed)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", xticklabels=clf.classes_, yticklabels=clf.classes_, cmap="Blues"
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix: Pattern Type Classification")
    plt.savefig(figs_dir / "figS2_confusion_matrix.png")
    plt.close()

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="results/simulations_phase3")
    parser.add_argument("--n_reps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.outdir, {"n_reps": args.n_reps, "seed": args.seed})
