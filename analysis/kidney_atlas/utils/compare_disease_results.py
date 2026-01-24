"""
Compare disease-stratified BioRSP results across conditions.

This utility script helps analyze and compare results from disease-stratified
BioRSP analysis, identifying genes with disease-specific spatial patterns.

Usage:
    python compare_disease_results.py results/disease_stratified

    python compare_disease_results.py results/tal_disease --plot

    python compare_disease_results.py results/disease_stratified --export comparison.csv
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare disease-stratified BioRSP results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "results_dir",
        type=str,
        help="Base directory containing disease-stratified results",
    )
    parser.add_argument(
        "--export",
        type=str,
        default=None,
        help="Export comparison table to CSV",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate comparison plots",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of top genes to show in plots (default: 20)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["Spatial_Bias_Score", "coverage_score", "p_value"],
        default="Spatial_Bias_Score",
        help="Metric to compare (default: spatial_score)",
    )
    return parser.parse_args()


def load_disease_results(results_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load gene_scores.csv from each disease subdirectory."""
    disease_data = {}

    for disease_folder in results_dir.iterdir():
        if not disease_folder.is_dir():
            continue

        scores_file = disease_folder / "gene_scores.csv"
        if scores_file.exists():
            df = pd.read_csv(scores_file)
            disease_name = disease_folder.name
            disease_data[disease_name] = df
            logger.info(f"Loaded {len(df)} genes for {disease_name}")
        else:
            logger.warning(f"No gene_scores.csv found in {disease_folder}")

    if len(disease_data) == 0:
        raise ValueError(f"No disease results found in {results_dir}")

    return disease_data


def merge_disease_data(
    disease_data: Dict[str, pd.DataFrame], metric: str = "Spatial_Bias_Score"
) -> pd.DataFrame:
    """Merge results across diseases for comparison."""
    diseases = sorted(disease_data.keys())
    merged = disease_data[diseases[0]][["gene", "gene_symbol", metric]].copy()
    merged = merged.rename(columns={metric: f"{metric}_{diseases[0]}"})

    for disease in diseases[1:]:
        df_disease = disease_data[disease][["gene", metric]]
        df_disease = df_disease.rename(columns={metric: f"{metric}_{disease}"})
        merged = merged.merge(df_disease, on="gene", how="outer")

    return merged


def compute_disease_changes(
    merged: pd.DataFrame, diseases: List[str], metric: str = "Spatial_Bias_Score"
) -> pd.DataFrame:
    """Compute changes relative to normal/reference condition."""
    reference = None
    for disease in diseases:
        if "normal" in disease.lower() or "healthy" in disease.lower():
            reference = disease
            break

    if reference is None:
        logger.warning("No reference condition found, using first disease")
        reference = diseases[0]

    logger.info(f"Using {reference} as reference condition")

    ref_col = f"{metric}_{reference}"
    for disease in diseases:
        if disease == reference:
            continue
        disease_col = f"{metric}_{disease}"
        change_col = f"change_{disease}_vs_{reference}"
        merged[change_col] = merged[disease_col] - merged[ref_col]

    return merged


def identify_disease_specific_genes(
    merged: pd.DataFrame,
    diseases: List[str],
    metric: str = "Spatial_Bias_Score",
    threshold: float = 0.1,
) -> Dict[str, List[str]]:
    """Identify genes with disease-specific patterns."""
    disease_specific = {}

    reference = None
    for disease in diseases:
        if "normal" in disease.lower() or "healthy" in disease.lower():
            reference = disease
            break
    if reference is None:
        reference = diseases[0]

    for disease in diseases:
        if disease == reference:
            continue

        change_col = f"change_{disease}_vs_{reference}"
        if change_col in merged.columns:
            specific_genes = merged[merged[change_col] > threshold].copy()
            specific_genes = specific_genes.sort_values(change_col, ascending=False)
            disease_specific[disease] = specific_genes.head(20)["gene_symbol"].tolist()

    return disease_specific


def plot_disease_comparison(
    merged: pd.DataFrame,
    diseases: List[str],
    metric: str = "Spatial_Bias_Score",
    top_n: int = 20,
    outdir: Path = None,
):
    """Generate comparison plots."""
    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 12))

    metric_cols = [f"{metric}_{d}" for d in diseases]
    merged_clean = merged.dropna(subset=metric_cols)
    merged_clean["variance"] = merged_clean[metric_cols].var(axis=1)
    top_genes = merged_clean.nlargest(top_n, "variance")

    heatmap_data = top_genes[metric_cols].values
    gene_labels = top_genes["gene_symbol"].values

    sns.heatmap(
        heatmap_data,
        yticklabels=gene_labels,
        xticklabels=[d.replace("_", " ").title() for d in diseases],
        cmap="viridis",
        annot=True,
        fmt=".2f",
        cbar_kws={"label": metric.replace("_", " ").title()},
        ax=ax,
    )

    ax.set_title(f"Top {top_n} Genes by {metric.replace('_', ' ').title()} Variance", fontsize=14)
    ax.set_ylabel("Gene", fontsize=12)
    ax.set_xlabel("Disease Condition", fontsize=12)
    plt.tight_layout()

    if outdir:
        plt.savefig(outdir / f"heatmap_{metric}.png", dpi=300, bbox_inches="tight")
        logger.info(f"Saved heatmap to {outdir / f'heatmap_{metric}.png'}")
    plt.close()

    reference = None
    for disease in diseases:
        if "normal" in disease.lower() or "healthy" in disease.lower():
            reference = disease
            break

    if reference:
        n_diseases = len(diseases) - 1
        fig, axes = plt.subplots(1, n_diseases, figsize=(6 * n_diseases, 6))
        if n_diseases == 1:
            axes = [axes]

        ax_idx = 0
        for disease in diseases:
            if disease == reference:
                continue

            change_col = f"change_{disease}_vs_{reference}"
            if change_col in merged.columns:
                ax = axes[ax_idx]

                ref_col = f"{metric}_{reference}"
                disease_col = f"{metric}_{disease}"

                plot_data = merged.dropna(subset=[ref_col, disease_col])

                ax.scatter(
                    plot_data[ref_col],
                    plot_data[disease_col],
                    alpha=0.5,
                    s=20,
                )

                max_val = max(plot_data[ref_col].max(), plot_data[disease_col].max())
                ax.plot([0, max_val], [0, max_val], "r--", alpha=0.5, linewidth=1)

                top_changed = plot_data.nlargest(5, change_col)
                for _, row in top_changed.iterrows():
                    ax.annotate(
                        row["gene_symbol"],
                        xy=(row[ref_col], row[disease_col]),
                        xytext=(5, 5),
                        textcoords="offset points",
                        fontsize=8,
                        alpha=0.7,
                    )

                ax.set_xlabel(f"{reference.replace('_', ' ').title()}", fontsize=10)
                ax.set_ylabel(f"{disease.replace('_', ' ').title()}", fontsize=10)
                ax.set_title(f"{disease.replace('_', ' ').title()} vs. Reference", fontsize=12)
                ax.grid(alpha=0.3)

                ax_idx += 1

        plt.tight_layout()
        if outdir:
            plt.savefig(outdir / f"scatter_{metric}.png", dpi=300, bbox_inches="tight")
            logger.info(f"Saved scatter plots to {outdir / f'scatter_{metric}.png'}")
        plt.close()


def generate_report(
    merged: pd.DataFrame,
    disease_data: Dict[str, pd.DataFrame],
    diseases: List[str],
    metric: str = "Spatial_Bias_Score",
    outdir: Path = None,
):
    """Generate a text report summarizing the comparison."""
    report_lines = [
        "# Disease-Stratified BioRSP Comparison Report",
        "",
        f"**Metric**: {metric.replace('_', ' ').title()}",
        f"**Diseases**: {', '.join([d.replace('_', ' ').title() for d in diseases])}",
        "",
        "## Summary Statistics",
        "",
    ]

    for disease in diseases:
        df = disease_data[disease]
        report_lines.append(f"### {disease.replace('_', ' ').title()}")
        report_lines.append(f"- Total genes: {len(df)}")
        report_lines.append(f"- Mean {metric}: {df[metric].mean():.3f}")
        report_lines.append(f"- Median {metric}: {df[metric].median():.3f}")
        if "Archetype" in df.columns:
            arch_counts = df["Archetype"].value_counts()
            report_lines.append("- Archetypes:")
            for arch, count in arch_counts.items():
                report_lines.append(f"  - {arch}: {count}")
        report_lines.append("")

    disease_specific = identify_disease_specific_genes(merged, diseases, metric)

    if disease_specific:
        report_lines.append("## Disease-Specific Genes (Top 10)")
        report_lines.append("")
        for disease, genes in disease_specific.items():
            report_lines.append(f"### {disease.replace('_', ' ').title()}")
            for i, gene in enumerate(genes[:10], 1):
                report_lines.append(f"{i}. {gene}")
            report_lines.append("")

    report_text = "\n".join(report_lines)

    if outdir:
        report_file = outdir / "comparison_report.md"
        with open(report_file, "w") as f:
            f.write(report_text)
        logger.info(f"Saved report to {report_file}")

    print("\n" + report_text)


def main():
    args = parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise ValueError(f"Results directory not found: {results_dir}")

    logger.info(f"Loading results from {results_dir}")

    disease_data = load_disease_results(results_dir)
    diseases = sorted(disease_data.keys())

    logger.info(f"Found {len(diseases)} disease conditions: {diseases}")

    logger.info(f"Merging data by {args.metric}...")
    merged = merge_disease_data(disease_data, metric=args.metric)

    merged = compute_disease_changes(merged, diseases, metric=args.metric)

    logger.info(f"Merged data: {len(merged)} genes")

    if args.export:
        export_file = Path(args.export)
        merged.to_csv(export_file, index=False)
        logger.info(f"Exported comparison to {export_file}")

    if args.plot:
        plot_outdir = results_dir / "comparison_plots"
        logger.info(f"Generating plots in {plot_outdir}")
        plot_disease_comparison(
            merged,
            diseases,
            metric=args.metric,
            top_n=args.top_n,
            outdir=plot_outdir,
        )

    generate_report(merged, disease_data, diseases, metric=args.metric, outdir=results_dir)


if __name__ == "__main__":
    main()
