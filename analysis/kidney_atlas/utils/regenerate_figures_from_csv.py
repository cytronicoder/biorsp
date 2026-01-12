"""
Regenerate all figures from existing classification.csv using updated plotting functions.
This script reads the classification CSV and regenerates figures with new archetype names
and labels.
"""

import json
import logging
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ARCHETYPE_MAPPING = {
    "Ubiquitous Uniform": "I: Ubiquitous",
    "Focal Marker": "III: Patchy",
    "Regional Gradient": "II: Gradient",
    "Rare Scattered": "IV: Basal",
}

ARCHETYPE_COLORS = {
    "I: Ubiquitous": "#4DBEEE",
    "III: Patchy": "#D95319",
    "II: Gradient": "#77AC30",
    "IV: Basal": "#A2142F",
}


def update_archetype_names(df: pd.DataFrame) -> pd.DataFrame:
    """Update archetype names from old to new naming scheme."""
    df = df.copy()

    rename_map = {
        "coverage_expr": "Coverage",
        "spatial_score": "Spatial_Score",
        "r_mean": "Directionality",
        "archetype": "Archetype",
    }
    df.rename(columns=rename_map, inplace=True)

    if "Archetype" in df.columns:
        df["Archetype"] = df["Archetype"].map(lambda x: ARCHETYPE_MAPPING.get(x, x))
    return df


def plot_cs_scatter(
    df: pd.DataFrame,
    c_cut: float,
    s_cut: float,
    outdir: Path,
    max_points: int = 5000,
    seed: int = 42,
):
    """Create C-S scatter plot with quadrant boundaries and labeled cutoff lines."""
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 10))

    if len(df) > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(df), size=max_points, replace=False)
        plot_df = df.iloc[idx].copy()
    else:
        plot_df = df.copy()

    for archetype, color in ARCHETYPE_COLORS.items():
        mask = plot_df["Archetype"] == archetype
        ax.scatter(
            plot_df.loc[mask, "Coverage"],
            plot_df.loc[mask, "Spatial_Score"],
            c=color,
            s=15,
            alpha=0.6,
            label=f"{archetype} (n={mask.sum()})",
            rasterized=True,
        )

    ax.axvline(c_cut, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.axhline(s_cut, color="gray", linestyle="--", linewidth=1, alpha=0.7)

    ax.set_xlabel("Coverage Score $C$", fontsize=16)
    ax.set_ylabel("Spatial Bias Score $S$", fontsize=16)
    ax.set_title(
        f"Gene Archetypes (n={len(df):,} genes)\n$c_{{\\mathrm{{cut}}}}$={c_cut:.3f}, $s_{{\\mathrm{{cut}}}}$={s_cut:.3f}",
        fontsize=18,
    )

    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=13, frameon=True, borderpad=0.5)

    x_max = min(1.02, plot_df["Coverage"].max() * 1.1)
    y_max = plot_df["Spatial_Score"].quantile(0.99) * 1.2
    lim = max(x_max, y_max)
    ax.set_xlim(-0.02, lim)
    ax.set_ylim(-0.01, lim)

    ax.text(
        c_cut,
        lim * 0.95,
        f"$c_{{\\mathrm{{cut}}}}$={c_cut:.3f}",
        ha="center",
        va="top",
        fontsize=11,
        color="gray",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.7),
    )
    ax.text(
        lim * 0.95,
        s_cut,
        f"$s_{{\\mathrm{{cut}}}}$={s_cut:.3f}",
        ha="right",
        va="center",
        fontsize=11,
        color="gray",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.7),
    )

    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()

    for ext in ["png", "pdf"]:
        outfile = outdir / f"fig_cs_scatter.{ext}"
        plt.savefig(outfile, dpi=300 if ext == "png" else None, bbox_inches="tight")
        logger.info(f"Saved {outfile}")
    plt.close(fig)


def plot_cs_marginals(df: pd.DataFrame, c_cut: float, s_cut: float, outdir: Path):
    """Plot marginal distributions of Coverage and Spatial Bias Scores."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.hist(df["Coverage"], bins=50, color="steelblue", alpha=0.7, edgecolor="white")
    ax.axvline(
        c_cut,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"$c_{{\\mathrm{{cut}}}}$={c_cut:.3f}",
    )
    ax.set_xlabel("Coverage Score $C$", fontsize=16)
    ax.set_ylabel("Number of Genes", fontsize=16)
    ax.set_title("Distribution of Coverage Scores", fontsize=16)
    ax.legend(fontsize=12)

    ax = axes[1]
    ax.hist(df["Spatial_Score"], bins=50, color="darkorange", alpha=0.7, edgecolor="white")
    ax.axvline(
        s_cut,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"$s_{{\\mathrm{{cut}}}}$={s_cut:.3f}",
    )
    ax.set_xlabel("Spatial Bias Score $S$", fontsize=16)
    ax.set_ylabel("Number of Genes", fontsize=16)
    ax.set_title("Distribution of Spatial Bias Scores", fontsize=16)
    ax.legend(fontsize=12)

    plt.tight_layout()

    for ext in ["png", "pdf"]:
        outfile = outdir / f"fig_cs_marginals.{ext}"
        plt.savefig(outfile, dpi=300 if ext == "png" else None, bbox_inches="tight")
        logger.info(f"Saved {outfile}")
    plt.close(fig)


def plot_top_tables(df: pd.DataFrame, outdir: Path, n_top: int = 15):
    """Create figure showing top genes per archetype."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    gene_col = "gene_name" if "gene_name" in df.columns else "gene"

    for idx, archetype in enumerate(["I: Ubiquitous", "II: Gradient", "III: Patchy", "IV: Basal"]):
        ax = axes[idx]
        color = ARCHETYPE_COLORS[archetype]

        subset = df[df["Archetype"] == archetype].nlargest(n_top, "Spatial_Score")

        if len(subset) == 0:
            ax.text(0.5, 0.5, f"No genes in {archetype}", ha="center", va="center", fontsize=14)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")
            continue

        table_data = []
        for _, row in subset.iterrows():
            gene_name = row[gene_col]
            c = row["Coverage"]
            s = row["Spatial_Score"]
            table_data.append([gene_name, f"{c:.3f}", f"{s:.3f}"])

        table = ax.table(
            cellText=table_data,
            colLabels=["Gene", "C", "S"],
            cellLoc="center",
            loc="center",
            bbox=[0, 0, 1, 1],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2)

        for i in range(3):
            table[(0, i)].set_facecolor(color)
            table[(0, i)].set_text_props(weight="bold", color="white")

        for i in range(1, len(table_data) + 1):
            for j in range(3):
                table[(i, j)].set_facecolor("#f0f0f0" if i % 2 == 0 else "white")

        ax.set_title(
            f"{archetype}\n(n={len(df[df['Archetype'] == archetype]):,} genes)",
            fontsize=16,
            pad=20,
            loc="center",
        )
        ax.axis("off")

    plt.suptitle("Top Genes per Archetype", fontsize=18, y=1.05)
    plt.subplots_adjust(hspace=0.25, wspace=0.3, top=0.93, bottom=0.05)

    for ext in ["png", "pdf"]:
        outfile = outdir / f"fig_top_tables.{ext}"
        plt.savefig(outfile, dpi=300 if ext == "png" else None, bbox_inches="tight")
        logger.info(f"Saved {outfile}")
    plt.close(fig)


def plot_archetype_examples(
    adata: ad.AnnData,
    df: pd.DataFrame,
    context,
    config,
    outdir: Path,
    embedding_key: str,
):
    """Plot representative gene for each archetype."""
    from scipy import sparse

    from biorsp.core.engine import compute_rsp_radar
    from biorsp.preprocess.foreground import define_foreground

    examples_dir = outdir / "examples"
    examples_dir.mkdir(exist_ok=True)

    has_gene_name = "gene_name" in df.columns

    selected = {}
    for archetype in ARCHETYPE_COLORS:
        subset = df[df["Archetype"] == archetype]
        if len(subset) == 0:
            continue

        if archetype == "III: Patchy" or archetype == "II: Gradient":
            row = subset.nlargest(1, "Spatial_Score").iloc[0]
        else:
            c_mean = subset["Coverage"].mean()
            s_mean = subset["Spatial_Score"].mean()
            dist = np.sqrt(
                (subset["Coverage"] - c_mean) ** 2 + (subset["Spatial_Score"] - s_mean) ** 2
            )
            row = subset.iloc[dist.argmin()]

        gene_id = row["gene"]
        gene_name = row["gene_name"] if has_gene_name else gene_id
        ensg_id = row.get("ensg_id", gene_id)
        selected[archetype] = (ensg_id, gene_name)

    if len(selected) == 0:
        logger.warning("No archetypes have genes - skipping example plots")
        return

    n_archetypes = len(selected)
    n_cols = 4
    n_rows = (n_archetypes * 2 + n_cols - 1) // n_cols
    fig = plt.figure(figsize=(16, 4 * n_rows))

    for i, (archetype, (ensg_id, gene_name)) in enumerate(selected.items()):
        try:
            row = (i * 2) // n_cols
            base_pos = (row * n_cols) + ((i * 2) % n_cols) + 1

            gene_idx = adata.var_names.get_loc(ensg_id)
            if sparse.issparse(adata.X):
                x = adata.X[:, gene_idx].toarray().flatten()
            else:
                x = adata.X[:, gene_idx].copy()

            coords = context.coords
            gene_row = df[df["gene_name"] == gene_name].iloc[0]

            try:
                y, info = define_foreground(
                    x,
                    mode="quantile",
                    q=config.foreground_quantile,
                    rng=np.random.default_rng(config.seed),
                    min_nonzero=10,
                    min_fg=10,
                )
                if y is None:
                    logger.warning(f"Foreground is None for {gene_name}, using threshold approach")
                    threshold = np.quantile(x[x > 0], 0.9) if np.sum(x > 0) > 10 else 0
                    y = (x >= threshold).astype(float)
            except Exception as e:
                logger.warning(
                    f"Could not define foreground for {gene_name}: {e}, using threshold fallback"
                )
                threshold = np.quantile(x[x > 0], 0.9) if np.sum(x > 0) > 10 else 0
                y = (x >= threshold).astype(float)
        except Exception as e:
            logger.error(f"Error processing {gene_name} ({ensg_id}): {e}")
            continue

        ax_emb = plt.subplot(n_rows, n_cols, base_pos)
        ax_emb.scatter(coords[:, 0], coords[:, 1], c="lightgray", s=1, alpha=0.3, rasterized=True)
        fg_mask = y > 0.5
        ax_emb.scatter(
            coords[fg_mask, 0], coords[fg_mask, 1], c="red", s=3, alpha=0.7, rasterized=True
        )
        ax_emb.set_title(f"{gene_name}\n{archetype}", fontsize=16, pad=12)
        ax_emb.set_xlabel("UMAP 1", fontsize=16)
        ax_emb.set_ylabel("UMAP 2", fontsize=16)
        ax_emb.set_aspect("equal")

        ax_radar = plt.subplot(n_rows, n_cols, base_pos + 1, projection="polar")

        radar = compute_rsp_radar(
            context.r_norm, context.theta, y, config=config, sector_indices=context.sector_indices
        )

        from biorsp.plotting.radar import plot_radar

        plot_radar(
            radar,
            ax=ax_radar,
            title=f"C={gene_row['Coverage']:.3f}, S={gene_row['Spatial_Score']:.3f}",
            mode="signed",
            theta_convention="math",
            color="b",
            alpha=0.3,
            linewidth=1.5,
        )

        fig_single = plt.figure(figsize=(10, 4))

        ax_single_emb = plt.subplot(1, 2, 1)
        ax_single_emb.scatter(
            coords[:, 0], coords[:, 1], c="lightgray", s=1, alpha=0.3, rasterized=True
        )
        fg_mask = y > 0.5
        ax_single_emb.scatter(
            coords[fg_mask, 0], coords[fg_mask, 1], c="red", s=3, alpha=0.7, rasterized=True
        )
        ax_single_emb.set_title(f"{gene_name} - {archetype}")
        ax_single_emb.set_aspect("equal")

        ax_single_radar = plt.subplot(1, 2, 2, projection="polar")
        radar = compute_rsp_radar(
            context.r_norm, context.theta, y, config=config, sector_indices=context.sector_indices
        )

        plot_radar(
            radar,
            ax=ax_single_radar,
            title="R(θ)",
            mode="signed",
            theta_convention="math",
            color="b",
            alpha=0.3,
            linewidth=1.5,
        )

        plt.tight_layout()
        fig_single.savefig(
            examples_dir / f"{gene_name}_{archetype.replace(' ', '_').replace(':', '')}.png",
            dpi=150,
        )
        plt.close(fig_single)

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(
            outdir / "figures" / f"fig_archetype_examples.{ext}", dpi=300, bbox_inches="tight"
        )
        logger.info(f"Saved {outdir / 'figures' / f'fig_archetype_examples.{ext}'}")
    plt.close(fig)

    example_meta = pd.DataFrame(
        [
            {
                "Archetype": arch,
                "ensg_id": ensg_id,
                "gene_name": gene_name,
                **df[df["gene_name"] == gene_name].iloc[0].to_dict(),
            }
            for arch, (ensg_id, gene_name) in selected.items()
        ]
    )
    example_meta.to_csv(examples_dir / "example_metadata.csv", index=False)


def main():
    results_dir = Path("results/kpmp_1")
    classification_csv = results_dir / "classification.csv"
    figures_dir = results_dir / "figures"
    h5ad_path = Path("data/kpmp.h5ad")

    figures_dir.mkdir(exist_ok=True)

    logger.info(f"Loading classification data from {classification_csv}")
    df = pd.read_csv(classification_csv)

    logger.info(f"Loading AnnData from {h5ad_path}...")
    adata = ad.read_h5ad(h5ad_path)

    gene_name_to_ensg = {}
    if "feature_name" in adata.var.columns:
        for ensg_id, gene_name in zip(adata.var_names, adata.var["feature_name"]):
            if gene_name and not gene_name.startswith("ENSG"):
                gene_name_to_ensg[gene_name] = ensg_id
            gene_name_to_ensg[ensg_id] = ensg_id

    df["ensg_id"] = df["gene"].map(gene_name_to_ensg)
    df.loc[df["ensg_id"].isna(), "ensg_id"] = df.loc[df["ensg_id"].isna(), "gene"]
    if "gene_name" not in df.columns:
        logger.info("Adding gene_name column (same as gene column)")
        df["gene_name"] = df["gene"]

    c_cut = df["c_cut_used"].iloc[0]
    s_cut = df["s_cut_used"].iloc[0]

    logger.info("Classification statistics:")
    logger.info(f"  Total genes: {len(df):,}")
    logger.info(f"  c_cut: {c_cut:.4f}")
    logger.info(f"  s_cut: {s_cut:.4f}")
    logger.info("Archetype counts:")
    for archetype, count in df["Archetype"].value_counts().items():
        logger.info(f"  {archetype}: {count:,}")

    logger.info("Generating basic figures...")
    plot_cs_scatter(df, c_cut, s_cut, figures_dir)
    plot_cs_marginals(df, c_cut, s_cut, figures_dir)
    plot_top_tables(df, figures_dir)

    logger.info("Preparing BioRSP context...")
    from biorsp.preprocess.context import prepare_context
    from biorsp.utils.config import BioRSPConfig

    manifest_path = results_dir / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        embedding_key = manifest.get("embedding_key", "X_umap")
        subset_query = manifest.get("subset_query", None)

        if subset_query:
            logger.info(f"Applying subset: {subset_query}")
            adata = adata[adata.obs.eval(subset_query)].copy()
    else:
        embedding_key = "X_umap"

    config = BioRSPConfig(
        delta_deg=60.0,
        B=72,
        foreground_quantile=0.90,
        foreground_mode="quantile",
        empty_fg_policy="zero",
        seed=42,
    )

    _, context = prepare_context(adata, embedding_key, config=config)

    logger.info("Generating archetype examples...")
    plot_archetype_examples(adata, df, context, config, results_dir, embedding_key)

    logger.info("Figure regeneration complete!")


if __name__ == "__main__":
    main()
