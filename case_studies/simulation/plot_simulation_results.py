"""
Plotting utilities for BioRSP simulation outputs.

Usage:
    python examples/plot_simulation_results.py --indir sim_results --outdir sim_plots

Generates publication-style figures:
 - QQ plots comparing naive vs stratified under Null A and structured Null B'
 - Power curves (power vs beta) split by variant and sigma
 - Angle recovery boxplots (circular error)
 - Distortion sensitivity scatter (A_g on latent vs distorted)
 - Adequacy / abstention heatmaps and summary tables

"""

import argparse
import importlib.util
import os
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# seaborn is optional; fall back to matplotlib when unavailable
try:
    import seaborn as sns

    sns.set(style="whitegrid", context="talk")
    HAS_SEABORN = True
except Exception:
    sns = None
    HAS_SEABORN = False
    plt.style.use("seaborn-whitegrid")


def prettify_name(s: str) -> str:
    """Make human-friendly labels from variable or scenario names."""
    if s is None:
        return ""
    try:
        s = str(s)
    except Exception:
        return str(s)

    mapping = {
        "beta": "Signal strength (β)",
        "sigma_deg": "σ (deg)",
        "theta_0": "Signal angle (rad)",
        "theta_g": "True angle (rad)",
        "A_g": "A_g",
        "variant": "Variant",
    }
    if s in mapping:
        return mapping[s]

    s2 = s.replace("_", " ")
    s2 = s2.replace("NullB", "Null B").replace("NullA", "Null A").replace("NullC", "Null C")
    s3 = " ".join([w.capitalize() for w in s2.split()])
    s3 = s3.replace("Beta", "β").replace("Sigma Deg", "σ (deg)")
    return s3


def compute_ks_stat(pvals: np.ndarray) -> float:
    """Compute simple KS statistic vs Uniform(0,1) for vector of p-values."""
    if len(pvals) == 0:
        return np.nan
    p_sorted = np.sort(pvals)
    expected = np.linspace(0, 1, len(p_sorted), endpoint=False) + 1.0 / (2 * len(p_sorted))
    return float(np.max(np.abs(p_sorted - expected)))


def plot_pvalue_hist_and_ks(
    p_naive: np.ndarray, p_strat: Optional[np.ndarray], outpath: str, title: str
):
    """Plot histograms of naive and stratified p-values and annotate KS stats."""
    from pathlib import Path

    import matplotlib.pyplot as plt

    p_naive = p_naive[~np.isnan(p_naive)]
    p_strat = p_strat[~np.isnan(p_strat)] if p_strat is not None else np.array([])

    ks_naive = compute_ks_stat(p_naive)
    ks_strat = compute_ks_stat(p_strat) if len(p_strat) > 0 else np.nan

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    ax.hist(p_naive, bins=50, alpha=0.6, label="Naive", color="red", density=True)
    if len(p_strat) > 0:
        ax.hist(p_strat, bins=50, alpha=0.6, label="Stratified", color="blue", density=True)
    ax.plot([0, 1], [1, 1], "k--", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_xlabel("p-value")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()

    ax = axes[1]
    bars = [ks_naive, ks_strat]
    labels = ["Naive", "Stratified"]
    ax.bar(labels, bars, color=["red", "blue"])
    ax.set_ylabel("KS stat (vs Uniform)")
    ax.set_ylim(0, max(0.05, np.nanmax(bars) * 1.2))
    for i, v in enumerate(bars):
        ax.text(
            i,
            (v if not np.isnan(v) else 0) + 0.005,
            f"{v:.3f}" if not np.isnan(v) else "n/a",
            ha="center",
        )

    fig.tight_layout()
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath)
    plt.close(fig)


def plot_null_calibration(indir: str, outdir: str):
    df = pd.read_csv(os.path.join(indir, "family_1_null_calibration.csv"))

    # Null A: exchangeable -- naive vs stratified should both be uniform
    df_a = df[df["null_type"] == "A"]
    if not df_a.empty:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        for geom in df_a["geometry"].unique():
            sub = df_a[df_a["geometry"] == geom]
            p_naive = sub["p_naive"].dropna().values
            p_strat = sub["p_strat"].dropna().values
            if len(p_naive) > 0:
                expected = np.linspace(0, 1, len(p_naive))
                ax.plot(
                    expected,
                    np.sort(p_naive),
                    label=f"{prettify_name(geom)} — Naive",
                    alpha=0.6,
                    linestyle="--",
                )
            if len(p_strat) > 0:
                expected = np.linspace(0, 1, len(p_strat))
                ax.plot(
                    expected,
                    np.sort(p_strat),
                    label=f"{prettify_name(geom)} — Stratified",
                    alpha=0.9,
                )
        ax.plot([0, 1], [0, 1], "k--")
        ax.set_xlabel("Expected P-value")
        ax.set_ylabel("Observed P-value")
        ax.set_title("QQ Plot: Null A (Exchangeable)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "qq_nullA_naive_vs_strat.png"))

    # Null B structured (depth confounding) -- show naive vs stratified overlay
    df_b = df[df["geometry"].str.contains("NullB", na=False)]
    if not df_b.empty:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

        p_naive = df_b["p_naive"].dropna().values
        p_strat = df_b["p_strat"].dropna().values

        if len(p_naive) > 0:
            expected = np.linspace(0, 1, len(p_naive))
            ax.plot(
                expected,
                np.sort(p_naive),
                label="Naive — Confounded",
                color="red",
                linestyle="--",
            )
        if len(p_strat) > 0:
            expected = np.linspace(0, 1, len(p_strat))
            ax.plot(expected, np.sort(p_strat), label="Stratified — Corrected", color="blue")

        ax.plot([0, 1], [0, 1], "k--")
        ax.set_xlabel("Expected P-value")
        ax.set_ylabel("Observed P-value")
        ax.set_title("QQ Plot: Null B' Structured Confounding")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "qq_nullB_structured_naive_vs_strat.png"))


def plot_power(indir: str, outdir: str):
    df = pd.read_csv(os.path.join(indir, "family_2_power.csv"))

    if df.empty:
        return

    # Ensure sigma_deg exists
    if "sigma_deg" not in df.columns:
        df["sigma_deg"] = np.nan

    # Compute power (fraction p_strat < 0.05) grouped by variant, beta, sigma_deg
    df["sig_detect"] = df["p_strat"] < 0.05
    grouped = (
        df.groupby(["variant", "beta", "sigma_deg"])["sig_detect"]
        .agg(["mean", "count"])
        .reset_index()
    )
    grouped = grouped.rename(columns={"mean": "power", "count": "n"})

    # Plot power vs beta for each variant, with one line per sigma
    variants = grouped["variant"].unique()
    for variant in variants:
        sub = grouped[grouped["variant"] == variant]
        plt.figure(figsize=(6, 4))
        for sigma in sorted(sub["sigma_deg"].dropna().unique()):
            ssub = sub[sub["sigma_deg"] == sigma]
            plt.plot(ssub["beta"], ssub["power"], marker="o", label=f"sigma={int(sigma)}°")
        if sub["sigma_deg"].isna().all():
            # No sigma info, aggregate across sigma
            agg = sub.groupby("beta")["power"].mean()
            plt.plot(agg.index, agg.values, marker="o", label="all sigma")
        plt.xlabel("Signal strength (β)")
        plt.ylabel("Power (p_strat < 0.05)")
        plt.title(f"Power Curve ({prettify_name(variant)})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"power_curve_{variant}.png"))
        plt.close()

    # Angle recovery: circular error distributions
    df["error"] = np.minimum(
        np.abs(df["theta_g"] - df["theta_0"]),
        2 * np.pi - np.abs(df["theta_g"] - df["theta_0"]),
    )
    plt.figure(figsize=(8, 6))
    if HAS_SEABORN:
        sns.boxplot(x="beta", y="error", hue="variant", data=df)
        plt.xticks(rotation=45, fontsize=8)
    else:
        # Fallback: simple grouped boxplot using pandas
        df.boxplot(column="error", by=["beta", "variant"], rot=45)
        plt.suptitle("")
        plt.xticks(rotation=45, fontsize=8)
    plt.xlabel("Signal strength (β)")
    plt.ylabel("Circular Error (radians)")
    plt.title("Angle Recovery Error")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "angle_recovery_boxplot.png"))
    plt.close()

    # Adequacy vs A_g scatter and Spearman correlation
    plt.figure(figsize=(6, 5))
    df_clean = df.dropna(subset=["A_g", "adequacy_fraction"])
    if not df_clean.empty:
        rho = df_clean["A_g"].corr(df_clean["adequacy_fraction"], method="spearman")
        plt.scatter(df_clean["adequacy_fraction"], df_clean["A_g"], s=1)
        plt.xlabel("Adequacy fraction")
        plt.ylabel("A_g (mean |RSP|)")
        plt.title(f"Adequacy vs A_g (Spearman ρ={rho:.2f})")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "adequacy_vs_Ag_scatter.png"))
    else:
        plt.close()


def plot_distortion_sensitivity(indir: str, outdir: str):
    df = pd.read_csv(os.path.join(indir, "family_3_robustness.csv"))
    if df.empty:
        return

    # Pivot so each gene has columns for each transform
    pivot = df.pivot(index="gene_id", columns="transform", values="A_g")

    # For each distorted transform vs Original, scatter and compute rank correlation
    for col in pivot.columns:
        if col == "Original":
            continue
        # Align on gene ids
        both = pivot[["Original", col]].dropna()
        if both.empty:
            continue
        rho = both["Original"].rank().corr(both[col].rank(), method="spearman")
        plt.figure(figsize=(5, 5))
        plt.scatter(both["Original"], both[col], s=1)
        plt.xlabel("A_g (Original)")
        plt.ylabel(f"A_g ({prettify_name(col)})")
        plt.title(f"Distortion sensitivity: {prettify_name(col)}\nSpearman ρ={rho:.2f}")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"distortion_scatter_{col}.png"))
        plt.close()

    # Violin/boxplot of A_g distributions across transforms
    df_wide = df.pivot(index="gene_id", columns="transform", values="A_g").reset_index()
    value_vars = [c for c in df_wide.columns if c != "gene_id"]
    df_m = df_wide.melt(
        id_vars=["gene_id"],
        value_vars=value_vars,
        var_name="transform",
        value_name="A_g_val",
    )
    plt.figure(figsize=(8, 5))
    if HAS_SEABORN:
        sns.violinplot(x="transform", y="A_g_val", data=df_m, inner="quartile")
        plt.xticks(rotation=45, fontsize=8)
    else:
        df_m.boxplot(column="A_g_val", by="transform", rot=45)
        plt.xticks(rotation=45, fontsize=8)
    plt.xlabel("Transform")
    plt.title("A_g distribution across transforms")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "distortion_Ag_violin.png"))
    plt.close()


def plot_adequacy_heatmap(indir: str, outdir: str):
    df_power = pd.read_csv(os.path.join(indir, "family_2_power.csv"))
    if df_power.empty:
        return

    # Fraction of genes that are adequate (is_adequate True) per beta/variant
    summary = df_power.groupby(["variant", "beta"])["is_adequate"].mean().reset_index()
    heat = summary.pivot(index="variant", columns="beta", values="is_adequate")

    plt.figure(figsize=(6, 4))
    if HAS_SEABORN:
        sns.heatmap(heat, annot=True, fmt=".2f", cmap="viridis")
    else:
        plt.imshow(heat, aspect="auto", cmap="viridis")
        plt.colorbar()
        # annotate
        for (i, j), val in np.ndenumerate(heat.values):
            plt.text(j, i, f"{val:.2f}", ha="center", va="center", color="white")
        plt.yticks(np.arange(len(heat.index)), [prettify_name(v) for v in heat.index])
        plt.xticks(np.arange(len(heat.columns)), heat.columns)
    plt.xlabel("Beta")
    plt.ylabel("Variant")
    plt.xticks(rotation=45, fontsize=8)
    plt.title("Fraction of adequate genes (by variant & beta)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "adequacy_heatmap.png"))
    plt.close()


# ---------------------- Additional (advanced) panels ----------------------


def plot_embedding_with_rsp(indir: str, outdir: str, top_frac: float = 0.1):
    """Produce side-by-side embedding + foreground RSP plots per geometry and save PNGs.

    - loads saved inputs from `indir/inputs/{geometry}_z.npy` and `_umis.npy`
    - re-generates expression using saved `seed` when present and available metadata
    - selects top `top_frac` fraction as foreground, computes RSP, and plots
    """
    from pathlib import Path

    simmod = load_sim_module()
    inputs_dir = Path(indir) / "inputs"

    df_power = pd.read_csv(os.path.join(indir, "family_2_power.csv"))
    df_null = pd.read_csv(os.path.join(indir, "family_1_null_calibration.csv"))

    geoms_set = set()
    if "geometry" in df_power.columns:
        geoms_set.update(df_power["geometry"].dropna().unique())
    if "geometry" in df_null.columns:
        geoms_set.update(df_null["geometry"].dropna().unique())
    geoms = sorted(geoms_set)
    os.makedirs(outdir, exist_ok=True)

    for geom in geoms:
        # prefer a representative row from power file (planted signals), else first null row
        if "geometry" in df_power.columns:
            rows = df_power[df_power["geometry"] == geom]
        else:
            rows = pd.DataFrame()

        if not rows.empty:
            # pick the highest-A_g replicate as representative
            try:
                row = rows.loc[rows["A_g"].idxmax()]
            except Exception:
                row = rows.iloc[0]
        else:
            if "geometry" in df_null.columns:
                rows = df_null[df_null["geometry"] == geom]
            else:
                rows = pd.DataFrame()
            if rows.empty:
                print(f"Skipping {geom}: no rows found in outputs")
                continue
            row = rows.iloc[0]

        z_path = inputs_dir / f"{geom}_z.npy"
        umis_path = inputs_dir / f"{geom}_umis.npy"
        if not z_path.exists() or not umis_path.exists():
            print(f"Skipping {geom}: missing inputs at {inputs_dir}")
            continue

        z = np.load(z_path)
        umis = np.load(umis_path)

        # regenerate expression based on available metadata
        seed = int(row["seed"]) if "seed" in row and pd.notna(row["seed"]) else None
        if pd.notna(row.get("null_type")):
            nt = row["null_type"]
            if nt == "A":
                x = simmod.generate_expression_null_A(umis, seed=seed)
            elif nt == "B":
                x = simmod.generate_expression_null_B(umis, seed=seed)
            else:
                x = simmod.generate_expression_null_C(
                    umis, donors=np.zeros(len(umis), dtype=int), seed=seed
                )
        else:
            variant = row.get("variant", "wedge")
            beta = float(row.get("beta", 1.0))
            sigma_deg = (
                float(row.get("sigma_deg", 20)) if pd.notna(row.get("sigma_deg", None)) else 20
            )
            sigma_rad = np.deg2rad(sigma_deg)
            x = simmod.generate_expression_alt(
                z,
                umis,
                variant=variant,
                theta_dagger=float(row.get("theta_0", 0.0)),
                sigma_theta=sigma_rad,
                beta_theta=beta,
                seed=seed,
            )

        theta = np.arctan2(z[:, 1], z[:, 0])
        thr = np.quantile(x, 1.0 - top_frac)
        y_fg = x >= thr
        if not y_fg.any():
            print(f"Skipping {geom}: no foreground cells at top_frac={top_frac}")
            continue

        radar = simmod.compute_rsp_radar(theta[y_fg], B=360, delta_deg=20)
        centers = radar.centers
        rsp = radar.rsp

        # Left: embedding; Right: polar RSP
        fig = plt.figure(figsize=(10, 4))
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2, projection="polar")

        sc = ax1.scatter(z[:, 0], z[:, 1], c=x, s=1, cmap="viridis")
        ax1.scatter(z[y_fg, 0], z[y_fg, 1], s=2, facecolors="none", edgecolors="k")
        # Title: prettified geometry (omit job ids)
        # include A_g if present in the row
        a_val = row.get("A_g", None)
        title = prettify_name(geom)
        if pd.notna(a_val):
            title = f"{title} — $A_g$={a_val:.3f}"
        ax1.set_title(title)
        # make embedding square with equal axes
        ax1.set_aspect("equal", adjustable="box")
        # set symmetric limits to ensure square appearance
        xlim = (np.min(z[:, 0]), np.max(z[:, 0]))
        ylim = (np.min(z[:, 1]), np.max(z[:, 1]))
        # center and make ranges equal
        xmid = 0.5 * (xlim[0] + xlim[1])
        ymid = 0.5 * (ylim[0] + ylim[1])
        half = max((xlim[1] - xlim[0]), (ylim[1] - ylim[0])) / 2.0
        ax1.set_xlim(xmid - half, xmid + half)
        ax1.set_ylim(ymid - half, ymid + half)
        fig.colorbar(sc, ax=ax1, label="Expression")

        # Polar RSP: centers are in radians already
        ax2.plot(centers, rsp, "-o", markersize=2)
        ax2.set_theta_zero_location("N")  # zero at top
        ax2.set_theta_direction(-1)  # clockwise
        ax2.set_title("RSP (foreground)", va="bottom")
        # set radial limits and r-labels
        ax2.set_ylim(-1.0, 1.0)
        # keep radial tick labels simple, ensure no underscores
        ax2.set_rlabel_position(135)
        # enforce square aspect by setting box aspect where supported
        try:
            ax2.set_box_aspect(1)
        except Exception:
            pass

        plt.tight_layout()
        outpath = os.path.join(outdir, f"embed_rsp_{geom}.png")
        fig.savefig(outpath)
        plt.close(fig)
        print("Saved", outpath)


def load_sim_module() -> object:
    """Dynamically import the simulation module so we can re-generate expression and RSP curves."""
    sim_path = os.path.join(os.getcwd(), "examples", "simulation.py")
    spec = importlib.util.spec_from_file_location("simmod", sim_path)
    simmod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(simmod)
    return simmod


def bootstrap_qq(
    pvals: np.ndarray, n_boot: int = 500, probs: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pvals = pvals[~np.isnan(pvals)]
    if probs is None:
        probs = np.linspace(0, 1, 100)
    q_mat = np.zeros((n_boot, len(probs)))
    n = len(pvals)
    if n == 0:
        return probs, np.zeros_like(probs), np.zeros_like(probs)
    for i in range(n_boot):
        samp = np.random.choice(pvals, size=n, replace=True)
        q_mat[i, :] = np.quantile(samp, probs)
    mean_q = q_mat.mean(axis=0)
    se_q = q_mat.std(axis=0, ddof=1)
    return probs, mean_q, se_q


def plot_qq_with_band(p_naive: np.ndarray, p_strat: np.ndarray, outpath: str):
    probs, mean_q_strat, se_q_strat = bootstrap_qq(p_strat, n_boot=500)
    plt.figure(figsize=(6, 6))
    # Plot naive raw
    if len(p_naive[~np.isnan(p_naive)]) > 0:
        exp = probs
        obs_naive = np.quantile(p_naive[~np.isnan(p_naive)], probs)
        plt.plot(exp, obs_naive, label="Naive", color="red", linestyle="--")
    # Plot stratified mean and band
    plt.plot(probs, mean_q_strat, label="Stratified (mean)", color="blue")
    plt.fill_between(
        probs,
        mean_q_strat - se_q_strat,
        mean_q_strat + se_q_strat,
        color="blue",
        alpha=0.3,
    )
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("Expected P-value")
    plt.ylabel("Observed P-value")
    plt.title("QQ with bootstrap band (Stratified)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def ks_histogram(p_naive: np.ndarray, p_strat: np.ndarray, outpath: str):
    p_naive = p_naive[~np.isnan(p_naive)]
    p_strat = p_strat[~np.isnan(p_strat)]

    # Direct KS against uniform
    def ks_vs_uniform(p):
        if len(p) == 0:
            return np.nan
        p_sorted = np.sort(p)
        expected = np.linspace(0, 1, len(p_sorted), endpoint=False) + 1.0 / (2 * len(p_sorted))
        return float(np.max(np.abs(p_sorted - expected)))

    ks_naive = ks_vs_uniform(p_naive)
    ks_strat = ks_vs_uniform(p_strat)

    plt.figure(figsize=(6, 4))
    plt.bar(["Naive", "Stratified"], [ks_naive, ks_strat], color=["red", "blue"])

    plt.ylabel("KS stat vs Uniform")
    plt.title("KS statistics (smaller is closer to uniform)")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def compute_discrepancy_rate(df: pd.DataFrame) -> pd.DataFrame:
    # For each geometry/scenario: fraction of genes where naive<0.05 and strat>0.05
    grouped = (
        df.groupby("geometry")
        .apply(lambda sub: ((sub["p_naive"] < 0.05) & (sub["p_strat"] >= 0.05)).mean())
        .reset_index(name="naive_only_fraction")
    )
    return grouped


def split_half_reproducibility(
    df: pd.DataFrame, indir: str, outpath: str, n_genes: int = 100, n_splits: int = 50
):
    simmod = load_sim_module()
    inputs_dir = os.path.join(indir, "inputs")
    # Select genes at random from df (that have seed and job info)
    if "seed" not in df.columns:
        print(
            "Skipping split-half reproducibility: 'seed' column not found in CSV outputs; re-run simulation with updated code to include seeds."
        )
        return

    df_sel = df.dropna(subset=["seed"]).sample(min(n_genes, len(df)), random_state=0)

    corrs = []
    for _, row in df_sel.iterrows():
        geom = row.get("geometry", None)
        # load inputs
        if geom is not None and os.path.exists(os.path.join(inputs_dir, f"{geom}_z.npy")):
            z = np.load(os.path.join(inputs_dir, f"{geom}_z.npy"))
            umis = np.load(os.path.join(inputs_dir, f"{geom}_umis.npy"))
        else:
            # fallback to family_2
            z = np.load(os.path.join(inputs_dir, "family_2_z.npy"))
            umis = np.load(os.path.join(inputs_dir, "family_2_umis.npy"))

        # regenerate expression depending on null/alt
        if pd.notna(row.get("null_type", None)):
            null_type = row["null_type"]
            seed = int(row["seed"])
            if null_type == "A":
                x = simmod.generate_expression_null_A(umis, seed=seed)
            elif null_type == "B":
                x = simmod.generate_expression_null_B(umis, seed=seed)
            else:
                x = simmod.generate_expression_null_C(
                    umis, donors=np.zeros(len(umis), dtype=int), seed=seed
                )
        else:
            seed = int(row["seed"])
            variant = row.get("variant", "wedge")
            beta = float(row.get("beta", 1.0))
            sigma_deg = (
                float(row.get("sigma_deg", 20)) if pd.notna(row.get("sigma_deg", None)) else 20
            )
            sigma_rad = np.deg2rad(sigma_deg)
            x = simmod.generate_expression_alt(
                z,
                umis,
                variant=variant,
                theta_dagger=float(row.get("theta_0", 0.0)),
                sigma_theta=sigma_rad,
                beta_theta=beta,
                seed=seed,
            )

        # now do splits
        As = []
        for s in range(n_splits):
            idx = np.random.choice(len(z), size=len(z) // 2, replace=False)
            theta = np.arctan2(z[:, 1], z[:, 0])
            y1 = np.zeros(len(z), dtype=bool)
            y2 = np.zeros(len(z), dtype=bool)
            # compute y in each split based on x subset
            y1[idx] = x[idx] >= np.quantile(x[idx], 0.9)
            # complementary set
            idx2 = np.setdiff1d(np.arange(len(z)), idx)
            y2[idx2] = x[idx2] >= np.quantile(x[idx2], 0.9)
            # compute A for each
            radar1 = simmod.compute_rsp_radar(theta[y1], B=360, delta_deg=20)
            s1 = simmod.compute_scalar_summaries(radar1).mean_abs_rsp
            radar2 = simmod.compute_rsp_radar(theta[y2], B=360, delta_deg=20)
            s2 = simmod.compute_scalar_summaries(radar2).mean_abs_rsp
            As.append((s1, s2))
        As = np.array(As)
        rho = np.corrcoef(As[:, 0], As[:, 1])[0, 1]
        if np.isfinite(rho):
            corrs.append(rho)

    plt.figure(figsize=(6, 4))
    plt.hist(corrs, bins=20)
    plt.xlabel("Split-half Pearson correlation of A_g")
    plt.title("Split-half reproducibility distribution")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def representative_rsp_curves(df: pd.DataFrame, indir: str, outpath: str, n_examples: int = 6):
    simmod = load_sim_module()
    inputs_dir = os.path.join(indir, "inputs")
    # pick examples: top A_g and median and low
    if "seed" not in df.columns:
        print(
            "Skipping representative RSP curves: 'seed' column not found in CSV outputs; re-run simulation to include seeds."
        )
        return

    df_valid = df.dropna(subset=["A_g", "seed"])
    if df_valid.empty:
        return
    df_sorted = df_valid.sort_values("A_g")
    picks = np.linspace(0, len(df_sorted) - 1, n_examples, dtype=int)
    # Make representative RSP figure square overall
    plt.figure(figsize=(12, 6))
    for i, idx in enumerate(picks):
        row = df_sorted.iloc[idx]
        geom = row.get("geometry", None)
        if geom is not None and os.path.exists(os.path.join(inputs_dir, f"{geom}_z.npy")):
            z = np.load(os.path.join(inputs_dir, f"{geom}_z.npy"))
            umis = np.load(os.path.join(inputs_dir, f"{geom}_umis.npy"))
        else:
            z = np.load(os.path.join(inputs_dir, "family_2_z.npy"))
            umis = np.load(os.path.join(inputs_dir, "family_2_umis.npy"))

        # regenerate x similarly to split-half
        if pd.notna(row.get("null_type", None)):
            if row["null_type"] == "A":
                x = simmod.generate_expression_null_A(umis, seed=int(row["seed"]))
            elif row["null_type"] == "B":
                x = simmod.generate_expression_null_B(umis, seed=int(row["seed"]))
            else:
                x = simmod.generate_expression_null_C(
                    umis, donors=np.zeros(len(umis), dtype=int), seed=int(row["seed"])
                )
        else:
            variant = row.get("variant", "wedge")
            beta = float(row.get("beta", 1.0))
            sigma_deg = (
                float(row.get("sigma_deg", 20)) if pd.notna(row.get("sigma_deg", None)) else 20
            )
            sigma_rad = np.deg2rad(sigma_deg)
            x = simmod.generate_expression_alt(
                z,
                umis,
                variant=variant,
                theta_dagger=float(row.get("theta_0", 0.0)),
                sigma_theta=sigma_rad,
                beta_theta=beta,
                seed=int(row["seed"]),
            )

        # compute radar for foreground
        theta = np.arctan2(z[:, 1], z[:, 0])
        threshold = np.quantile(x, 0.9)
        y = x >= threshold
        radar = simmod.compute_rsp_radar(theta[y], B=360, delta_deg=20)
        centers = radar.centers
        rsp = radar.rsp

        ax = plt.subplot(2, int(np.ceil(n_examples / 2)), i + 1, projection="polar")
        ax.plot(centers, rsp, "-o", markersize=2)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        # Remove job id and use LaTeX for A_g
        ax.set_title(
            f"{prettify_name(row.get('geometry'))} — $A_g$={row.get('A_g'):.3f}", va="bottom"
        )
        ax.set_ylim(-1.0, 1.0)
        try:
            ax.set_box_aspect(1)
        except Exception:
            pass
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


# ---------------------- Glue + CLI ----------------------


def _run_additional_panels(indir: str, outdir: str):
    df1 = pd.read_csv(os.path.join(indir, "family_1_null_calibration.csv"))
    df2 = pd.read_csv(os.path.join(indir, "family_2_power.csv"))

    # Null B: QQ band, KS, discrepancy bar
    dfb = df1[df1["geometry"].str.contains("NullB", na=False)]
    if not dfb.empty:
        p_naive = dfb["p_naive"].values
        p_strat = dfb["p_strat"].values
        plot_qq_with_band(p_naive, p_strat, os.path.join(outdir, "qq_nullB_band.png"))
        ks_histogram(p_naive, p_strat, os.path.join(outdir, "ks_nullB.png"))
        disc = compute_discrepancy_rate(df1)
        ax = disc.plot.bar(x="geometry", y="naive_only_fraction", legend=False)
        ax.set_xticklabels(
            [prettify_name(t.get_text()) for t in ax.get_xticklabels()], rotation=45, fontsize=8
        )
        ax.set_ylabel("Fraction naive-only (p_naive < 0.05 & p_strat ≥ 0.05)")
        plt.tight_layout()
        ax.get_figure().savefig(os.path.join(outdir, "discrepancy_bar.png"))
        plt.close()

    # Split-half reproducibility and representative RSP curves (may be slow/heavy)
    if not df2.empty:
        split_half_reproducibility(df2, indir, os.path.join(outdir, "split_half_repro.png"))
        representative_rsp_curves(
            pd.concat([df1, df2], ignore_index=True).dropna(subset=["A_g"]),
            indir,
            os.path.join(outdir, "repr_rsp_curves.png"),
        )
        # Embedding + RSP figures per geometry (demonstration)
        plot_embedding_with_rsp(indir, outdir)


def main(indir: str, outdir: str, extra_panels: bool = False):
    os.makedirs(outdir, exist_ok=True)
    plot_null_calibration(indir, outdir)
    plot_power(indir, outdir)
    plot_distortion_sensitivity(indir, outdir)
    plot_adequacy_heatmap(indir, outdir)
    if extra_panels:
        _run_additional_panels(indir, outdir)
    print("Plots saved to", outdir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot BioRSP simulation outputs")
    parser.add_argument(
        "--indir",
        type=str,
        default="sim_results",
        help="Input directory with CSV outputs",
    )
    parser.add_argument(
        "--outdir", type=str, default="sim_plots", help="Output directory for figures"
    )
    parser.add_argument(
        "--extra-panels",
        dest="extra_panels",
        action="store_true",
        help="Run additional (seed-dependent) panels like split-half and representative RSP curves",
    )
    args = parser.parse_args()
    main(args.indir, args.outdir, extra_panels=args.extra_panels)
