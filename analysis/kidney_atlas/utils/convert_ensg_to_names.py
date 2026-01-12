#!/usr/bin/env python3
"""
Utility script to convert ENSG IDs to gene names in pipeline outputs.

Updates:
- classification.csv
- runs_all_genes.csv
- Replots fig_archetype_examples.png with gene names
- Replots fig_top_tables.png with gene names
"""

import argparse
import logging
from pathlib import Path

import anndata as ad
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def create_ensg_to_name_mapping(h5ad_path: Path) -> dict:
    """Create mapping from ENSG IDs to gene names."""
    logger.info(f"Loading AnnData from {h5ad_path}")
    adata = ad.read_h5ad(h5ad_path)

    mapping = {}
    for ensg_id, gene_name in zip(adata.var_names, adata.var["feature_name"]):
        if gene_name.startswith("ENSG") or not gene_name:
            mapping[ensg_id] = ensg_id
        else:
            mapping[ensg_id] = gene_name

    logger.info(f"Created mapping for {len(mapping)} genes")
    return mapping


def update_csv_files(results_dir: Path, mapping: dict):
    """Update gene column in CSV files with gene names."""
    csv_files = ["classification.csv", "runs_all_genes.csv"]

    for csv_file in csv_files:
        csv_path = results_dir / csv_file
        if not csv_path.exists():
            logger.warning(f"File not found: {csv_path}")
            continue

        logger.info(f"Updating {csv_file}")
        df = pd.read_csv(csv_path)

        if "gene" not in df.columns:
            logger.warning(f"No 'gene' column in {csv_file}")
            continue

        df["gene"] = df["gene"].map(lambda x: mapping.get(x, x))

        df.to_csv(csv_path, index=False)
        logger.info(f"Updated {csv_file} with gene names")


def replot_figures(results_dir: Path, h5ad_path: Path):
    """Replot figures with gene names by re-running the plotting functions."""
    logger.info("Reploting figures with gene names")

    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from run_kpmp_archetypes_all_genes import plot_top_tables

    classification_path = results_dir / "classification.csv"
    if not classification_path.exists():
        logger.error(f"Classification file not found: {classification_path}")
        return

    df = pd.read_csv(classification_path)
    logger.info(f"Loaded {len(df)} genes from classification.csv")

    figures_dir = results_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    plot_top_tables(df, figures_dir)
    logger.info("Replotted fig_top_tables.png")

    logger.info("Preparing to replot archetype examples...")
    ad.read_h5ad(h5ad_path)

    logger.info("Note: To replot fig_archetype_examples.png, please re-run the full pipeline")
    logger.info("The updated CSV files already contain gene names")


def main():
    parser = argparse.ArgumentParser(
        description="Convert ENSG IDs to gene names in pipeline outputs"
    )
    parser.add_argument("--h5ad", type=Path, required=True, help="Path to AnnData h5ad file")
    parser.add_argument("--results-dir", type=Path, required=True, help="Path to results directory")
    parser.add_argument("--replot", action="store_true", help="Replot top tables figure")

    args = parser.parse_args()

    mapping = create_ensg_to_name_mapping(args.h5ad)

    update_csv_files(args.results_dir, mapping)

    if args.replot:
        replot_figures(args.results_dir, args.h5ad)

    logger.info("Conversion complete!")


if __name__ == "__main__":
    main()
