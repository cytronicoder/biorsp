"""
Disease-stratified BioRSP Analysis for KPMP Data

This script stratifies KPMP kidney data by disease condition (normal, acute kidney
failure, chronic kidney disease) and runs BioRSP analysis separately for each condition.
Results are saved to separate folders for comparison.

It can be run on:
1. All cells (stratified by disease)
2. Specific cell types (e.g., TAL cells only, stratified by disease)

Key Features:
- Automatic detection of disease metadata columns
- Donor stratification within each disease group
- Parallel processing support
- Comprehensive provenance tracking
- Optional cell type filtering (e.g., TAL cells only)

Quick Examples:
    python run_disease_stratified_analysis.py \\
      --ref-data data/kpmp.h5ad \\
      --outdir results/disease_stratified \\
      --max-genes 100

    python run_disease_stratified_analysis.py \\
      --ref-data data/kpmp.h5ad \\
      --outdir results/tal_disease_stratified \\
      --celltype-key subclass.l1 \\
      --celltype-filter TAL \\
      --max-genes 100

    python run_disease_stratified_analysis.py \\
      --ref-data data/kpmp.h5ad \\
      --outdir results/disease_full \\
      --max-genes 500 \\
      --n-permutations 1000 \\
      --do-genegene \\
      --n-workers 4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import warnings
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse
from matplotlib import rcParams

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from plot_utils_validated import (
    plot_cs_marginals,
    plot_cs_scatter,
    plot_gene_exemplar,
)

from biorsp import (
    BioRSPConfig,
    classify_genes,
    score_gene_pairs,
    score_genes,
)

try:
    from analysis.kidney_atlas.utils.standardized_plotting import (
        generate_kidney_panels,
    )

    HAS_STANDARDIZED_PLOTTING = True
except ImportError:
    HAS_STANDARDIZED_PLOTTING = False

warnings.filterwarnings("ignore", message=".*dtype argument is deprecated.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="legacy_api_wrap")

rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42

try:
    import anndata
    import scanpy as sc
except ImportError:
    anndata = None
    sc = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


DISEASE_KEYS = [
    "disease_category",
    "disease",
    "condition",
    "dx",
    "phenotype",
    "injury_state",
    "disease_state",
]

DISEASE_MAPPINGS = {
    "healthy_living_donor": "healthy_reference",
    "healthy_stone_donor": "healthy_reference",
    "normal": "healthy_reference",
    "healthy": "healthy_reference",
    "control": "healthy_reference",
    "reference": "healthy_reference",
    "aki": "acute_kidney_injury",
    "acute": "acute_kidney_injury",
    "acute_kidney_injury": "acute_kidney_injury",
    "acute_kidney_failure": "acute_kidney_injury",
    "akf": "acute_kidney_injury",
    "ckd": "chronic_kidney_disease",
    "chronic": "chronic_kidney_disease",
    "chronic_kidney_disease": "chronic_kidney_disease",
}

ARCHETYPE_NAMES = {
    "Ubiquitous_uniform": "Ubiquitous",
    "localized_program": "Gradient",
    "niche_biomarker": "Patchy",
    "sparse_presence": "Basal",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Disease-Stratified BioRSP Analysis for KPMP Data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_disease_stratified_analysis.py --ref-data data/kpmp.h5ad \\
      --outdir results/disease_stratified --max-genes 100

  python run_disease_stratified_analysis.py --ref-data data/kpmp.h5ad \\
      --outdir results/tal_disease --celltype-key subclass.l1 \\
      --celltype-filter TAL --max-genes 100
""",
    )

    io_group = parser.add_argument_group("Input/Output")
    io_group.add_argument(
        "--ref-data",
        dest="ref_data",
        type=str,
        required=True,
        help="Path to reference data (.h5ad preferred)",
    )
    io_group.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="Base directory to save all results (subdirs per disease)",
    )

    cell_group = parser.add_argument_group("Cell Selection")
    cell_group.add_argument(
        "--disease-key",
        dest="disease_key",
        type=str,
        default=None,
        help="Metadata column for disease labels (auto-detect if not provided)",
    )
    cell_group.add_argument(
        "--celltype-key",
        dest="celltype_key",
        type=str,
        default="subclass.l1",
        help="Metadata column for cell type (default: subclass.l1)",
    )
    cell_group.add_argument(
        "--celltype-filter",
        dest="celltype_filter",
        type=str,
        nargs="+",
        default=None,
        help="Cell type label(s to analyze (e.g., TAL). If not provided, analyze all cells.",
    )
    cell_group.add_argument(
        "--donor-key",
        dest="donor_key",
        type=str,
        default="donor_id",
        help="Metadata column for donor identifiers (default: donor_id)",
    )
    cell_group.add_argument(
        "--subsample",
        type=int,
        default=None,
        help="Subsample to N cells per disease for faster testing (default: no subsampling)",
    )

    gene_group = parser.add_argument_group("Gene Selection")
    gene_group.add_argument(
        "--controls",
        type=str,
        help="Comma-separated control genes (e.g., 'SLC12A1,UMOD,EGF')",
    )
    gene_group.add_argument(
        "--min-pct",
        dest="min_pct",
        type=float,
        default=0.01,
        help="Minimum expression prevalence for discovery genes (default: 0.01)",
    )
    gene_group.add_argument(
        "--max-genes",
        dest="max_genes",
        type=int,
        default=None,
        help="Maximum discovery genes to analyze (default: all passing filters)",
    )
    gene_group.add_argument(
        "--exclude-patterns",
        dest="exclude_patterns",
        type=str,
        default="^MT-|^mt-|^RPS|^RPL",
        help="Regex pattern for genes to exclude (default: MT/ribosomal)",
    )

    scoring_group = parser.add_argument_group("Scoring Parameters")
    scoring_group.add_argument(
        "--embedding-key",
        dest="embedding_key",
        type=str,
        default=None,
        help="Key in adata.obsm for embedding (default: auto-detect X_umap)",
    )
    scoring_group.add_argument(
        "--B",
        type=int,
        default=72,
        help="Number of angular sectors (default: 72 = 5° resolution)",
    )
    scoring_group.add_argument(
        "--delta-deg",
        dest="delta_deg",
        type=float,
        default=60.0,
        help="Sector width in degrees (default: 60°)",
    )
    scoring_group.add_argument(
        "--foreground-quantile",
        dest="foreground_quantile",
        type=float,
        default=0.90,
        help="Quantile for foreground selection (default: 0.90)",
    )
    scoring_group.add_argument(
        "--expr-threshold-mode",
        dest="expr_threshold_mode",
        type=str,
        choices=["detect", "fixed", "nonzero_quantile"],
        default="detect",
        help="How to determine coverage threshold (default: detect)",
    )
    scoring_group.add_argument(
        "--expr-threshold-value",
        dest="expr_threshold_value",
        type=float,
        default=None,
        help="Fixed coverage threshold (only used if mode=fixed)",
    )
    scoring_group.add_argument(
        "--empty-fg-policy",
        dest="empty_fg_policy",
        type=str,
        choices=["nan", "zero"],
        default="zero",
        help="Policy for empty-foreground sectors (default: zero)",
    )
    scoring_group.add_argument(
        "--n-permutations",
        dest="n_permutations",
        type=int,
        default=200,
        help="Number of permutations for p-value calculation (default: 200)",
    )

    class_group = parser.add_argument_group("Archetype Classification")
    class_group.add_argument(
        "--c-cut",
        dest="c_cut",
        type=float,
        default=0.10,
        help="Coverage cutoff for archetype classification (default: 0.10)",
    )
    class_group.add_argument(
        "--s-cut",
        dest="s_cut",
        type=float,
        default=None,
        help="Spatial score cutoff (default: median of scored genes)",
    )

    genegene_group = parser.add_argument_group("Gene-Gene Analysis")
    genegene_group.add_argument(
        "--do-genegene",
        dest="do_genegene",
        action="store_true",
        help="Compute pairwise gene relationships (can be slow)",
    )

    runtime_group = parser.add_argument_group("Runtime")
    runtime_group.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    runtime_group.add_argument(
        "--n-workers",
        dest="n_workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)",
    )
    runtime_group.add_argument(
        "--strict-plots",
        dest="strict_plots",
        action="store_true",
        help="Raise exceptions on plotting errors instead of logging and continuing",
    )
    runtime_group.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke test mode: 50 genes, no permutations, all plot types (overrides max_genes/n_permutations)",
    )
    runtime_group.add_argument(
        "--debug-plots",
        dest="debug_plots",
        action="store_true",
        help="Generate intermediate debug plots (pointcloud, sector support)",
    )

    return parser.parse_args()


def discover_disease_key(adata: anndata.AnnData) -> str | None:
    """Auto-discover the disease/condition metadata column.

    Args:
        adata: Annotated data object.

    Returns:
        Discovered disease key or None if not found.
    """
    for key in DISEASE_KEYS:
        if key in adata.obs.columns:
            logger.info(f"Discovered disease column: {key}")
            return key

    logger.warning("Could not auto-discover disease column")
    logger.warning(f"Available columns: {list(adata.obs.columns)}")
    return None


def standardize_disease_labels(disease_series: pd.Series) -> pd.Series:
    """Standardize disease labels to canonical names.

    Args:
        disease_series: Disease labels from metadata.

    Returns:
        Standardized disease labels.
    """
    standardized = disease_series.str.lower().str.replace(" ", "_")
    standardized = standardized.map(lambda x: DISEASE_MAPPINGS.get(x, x))
    return standardized


def get_disease_groups(adata: anndata.AnnData, disease_key: str) -> dict[str, np.ndarray]:
    """Get cell indices for each disease group.

    Args:
        adata: Annotated data object.
        disease_key: Column name for disease labels.

    Returns:
        Mapping from disease name to cell indices.
    """
    disease_labels = standardize_disease_labels(adata.obs[disease_key])

    disease_groups = {}
    for disease in disease_labels.unique():
        if pd.isna(disease):
            continue
        mask = disease_labels == disease
        indices = np.where(mask)[0]
        if len(indices) > 0:
            disease_groups[disease] = indices
            logger.info(f"Disease group '{disease}': {len(indices)} cells")

    return disease_groups


def estimate_coverage(
    x: np.ndarray, mode: str, value: float | None, nonzero_q: float = 0.25
) -> float:
    """Estimate coverage using BioRSP threshold logic.

    Args:
        x: Expression vector.
        mode: Threshold mode (`detect`, `fixed`, `nonzero_quantile`).
        value: Fixed threshold value (for `fixed` mode).
        nonzero_q: Quantile for `nonzero_quantile` mode.

    Returns:
        Fraction of cells with expression above the threshold.
    """
    if mode == "fixed":
        if value is not None:
            threshold = float(value)
        else:
            is_integers = np.allclose(x, np.round(x))
            threshold = 1.0 if is_integers else 0.1
    elif mode == "nonzero_quantile":
        nonzero = x[x > 0]
        if len(nonzero) == 0:
            return 0.0
        threshold = float(np.percentile(nonzero, nonzero_q * 100))
    else:
        is_integers = (x.dtype.kind in "iu") or (np.allclose(x, np.round(x)) and np.max(x) > 1.0)
        threshold = 1.0 if is_integers else 1e-6

    return float(np.mean(x >= threshold))


def load_reference(ref_path: str) -> anndata.AnnData:
    """Load the reference dataset from a `.h5ad` file.

    Args:
        ref_path: Path to the reference dataset.

    Returns:
        Loaded AnnData object.

    Raises:
        FileNotFoundError: If the reference path does not exist.
        ValueError: If the file format is unsupported.
    """
    logger.info(f"Loading reference from {ref_path}")
    if not Path(ref_path).exists():
        raise FileNotFoundError(f"Reference file not found: {ref_path}")

    if ref_path.endswith(".h5ad"):
        adata = anndata.read_h5ad(ref_path)
    else:
        raise ValueError(f"Unsupported file format: {ref_path}")

    logger.info(f"Loaded {adata.n_obs} cells × {adata.n_vars} genes")
    return adata


def compute_file_checksum(filepath: str, algorithm: str = "sha256") -> str:
    """Compute a file checksum for provenance.

    Args:
        filepath: Path to the file.
        algorithm: Hash algorithm name.

    Returns:
        Hex-encoded checksum.
    """
    h = hashlib.new(algorithm)
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def build_symbol_mappings(
    adata: anndata.AnnData,
) -> tuple[dict[str, str], dict[str, str]]:
    """Build bidirectional mappings between `var_names` and gene symbols.

    Checks for `feature_name` (KPMP), `gene_symbols`, or `symbol` columns.

    Args:
        adata: AnnData object.

    Returns:
        Tuple of `(var_to_symbol, symbol_to_var)` mappings.
    """
    if "feature_name" in adata.var.columns:
        var_to_symbol = {}
        for var_name, feature_name in zip(adata.var_names, adata.var["feature_name"]):
            if pd.notna(feature_name) and feature_name != "":
                var_to_symbol[var_name] = str(feature_name)
            else:
                var_to_symbol[var_name] = var_name
    elif "gene_symbols" in adata.var.columns:
        var_to_symbol = adata.var["gene_symbols"].to_dict()
    elif "symbol" in adata.var.columns:
        var_to_symbol = adata.var["symbol"].to_dict()
    else:
        var_to_symbol = {v: v for v in adata.var_names}

    symbol_to_var = {v: k for k, v in var_to_symbol.items()}
    return var_to_symbol, symbol_to_var


def detect_embedding_key(adata: anndata.AnnData, user_key: str | None) -> str:
    """Detect or validate the embedding key.

    Args:
        adata: Annotated data object.
        user_key: User-specified key (takes priority).

    Returns:
        Validated embedding key.

    Raises:
        ValueError: If no valid embedding is found.
    """
    if user_key:
        if user_key not in adata.obsm:
            raise ValueError(f"Embedding key '{user_key}' not found in adata.obsm")
        logger.info(f"Using user-specified embedding: {user_key}")
        return user_key

    candidates = ["X_umap", "X_UMAP", "X_tsne", "X_pca"]
    for key in candidates:
        if key in adata.obsm:
            logger.info(f"Auto-detected embedding: {key}")
            return key

    available = list(adata.obsm.keys())
    raise ValueError(
        f"No embedding found. Available keys: {available}. Please specify --embedding-key"
    )


def select_genes(
    adata: anndata.AnnData,
    controls_str: str | None,
    symbol_to_var: dict[str, str],
    var_to_symbol: dict[str, str],
    min_pct: float,
    max_genes: int | None,
    exclude_patterns: str,
    seed: int,
    expr_threshold_mode: str = "detect",
    expr_threshold_value: float | None = None,
) -> tuple[list[str], list[str], dict]:
    """Select genes for analysis (controls plus discovery set).

    Args:
        adata: AnnData object.
        controls_str: Comma-separated control gene list.
        symbol_to_var: Mapping from symbol to var name.
        var_to_symbol: Mapping from var name to symbol.
        min_pct: Minimum coverage fraction for discovery genes.
        max_genes: Maximum number of genes to analyze.
        exclude_patterns: Regex pattern for excluding genes by symbol.
        seed: Random seed for subsampling.
        expr_threshold_mode: Threshold mode for coverage estimation.
        expr_threshold_value: Fixed threshold value when using `fixed` mode.

    Returns:
        Tuple of `(genes_to_analyze, control_vars, selection_info)`.
    """
    logger.info("Selecting genes for analysis...")

    control_vars = []
    if controls_str:
        control_symbols = [s.strip() for s in controls_str.split(",")]
        for sym in control_symbols:
            var = symbol_to_var.get(sym, sym)
            if var in adata.var_names:
                control_vars.append(var)
            else:
                logger.warning(f"Control gene '{sym}' not found in dataset")

    logger.info(f"Control genes: {len(control_vars)}")

    discovery_vars = []
    logger.info(
        f"Computing coverage for {adata.n_vars} genes using threshold mode: {expr_threshold_mode}"
    )

    for gene_var in adata.var_names:
        if gene_var in control_vars:
            continue

        if exclude_patterns:
            gene_sym = var_to_symbol.get(gene_var, gene_var)
            if re.search(exclude_patterns, gene_sym):
                continue

        idx = adata.var_names.get_loc(gene_var)
        if scipy.sparse.issparse(adata.X):
            x = adata.X[:, idx].toarray().flatten()
        else:
            x = adata.X[:, idx]

        coverage = estimate_coverage(x, expr_threshold_mode, expr_threshold_value)

        if coverage >= min_pct:
            discovery_vars.append(gene_var)

    logger.info(f"Discovery genes (passing filters): {len(discovery_vars)}")

    all_genes = list(dict.fromkeys(control_vars + discovery_vars))

    if max_genes and len(all_genes) > max_genes:
        rng = np.random.default_rng(seed)
        non_control = [g for g in all_genes if g not in control_vars]
        rng.shuffle(non_control)
        all_genes = control_vars + non_control[: max_genes - len(control_vars)]
        logger.info(f"Subsampled to {len(all_genes)} genes (seed={seed})")

    selection_info = {
        "n_control": len(control_vars),
        "n_discovery": len(discovery_vars),
        "n_total_selected": len(all_genes),
        "min_pct": min_pct,
        "exclude_patterns": exclude_patterns,
        "expr_threshold_mode": expr_threshold_mode,
    }

    return all_genes, control_vars, selection_info


def run_analysis_for_disease(
    adata_disease: anndata.AnnData,
    disease_name: str,
    genes_to_analyze: list[str],
    control_vars: list[str],
    var_to_symbol: dict[str, str],
    embedding_key: str,
    config: BioRSPConfig,
    args: argparse.Namespace,
    outdir: Path,
) -> dict:
    """Run complete BioRSP analysis for one disease group.

    Args:
        adata_disease: Subset of data for this disease.
        disease_name: Disease condition name.
        genes_to_analyze: List of genes to score.
        control_vars: Control gene identifiers.
        var_to_symbol: Mapping from var name to symbol.
        embedding_key: Embedding key to use.
        config: BioRSP configuration.
        args: Parsed CLI arguments.
        outdir: Output directory for this disease.

    Returns:
        Analysis metadata dictionary.
    """
    logger.info(f"\n{'=' * 80}")
    logger.info(f"Analyzing disease group: {disease_name}")
    logger.info(f"{'=' * 80}\n")

    outdir.mkdir(parents=True, exist_ok=True)

    analysis_meta = {
        "disease": disease_name,
        "n_cells": int(adata_disease.n_obs),
        "timestamp": datetime.now().isoformat(),
    }

    if args.donor_key in adata_disease.obs.columns:
        donor_counts = adata_disease.obs[args.donor_key].value_counts().to_dict()
        analysis_meta["donor_distribution"] = donor_counts
        logger.info(f"  Donors: {len(donor_counts)}")
    else:
        analysis_meta["donor_distribution"] = None

    if adata_disease.n_obs < 100:
        logger.warning("  WARNING: Very few cells (<100). Results may be unstable.")
        analysis_meta["warning"] = "insufficient_cells"

    logger.info(f"  Scoring {len(genes_to_analyze)} genes...")
    df_results = score_genes(
        adata_disease,
        genes=genes_to_analyze,
        embedding_key=embedding_key,
        config=config,
    )

    df_results["gene_symbol"] = df_results["gene"].map(var_to_symbol)
    df_results["is_control"] = df_results["gene"].isin(control_vars)

    logger.info(f"  Scored: {len(df_results)} genes")
    analysis_meta["n_genes_scored"] = len(df_results)

    logger.info("  Classifying genes...")
    df_classified = classify_genes(
        df_results,
        c_cut=args.c_cut,
        s_cut=args.s_cut,
    )

    df_classified["Archetype"] = df_classified["Archetype"].map(ARCHETYPE_NAMES)

    archetype_counts = df_classified["Archetype"].value_counts().to_dict()
    logger.info(f"  Archetypes: {archetype_counts}")
    analysis_meta["archetype_counts"] = archetype_counts

    results_file = outdir / "gene_scores.csv"
    df_classified.to_csv(results_file, index=False)
    logger.info(f"  Saved results to {results_file}")

    if args.do_genegene:
        logger.info("  Computing gene-gene relationships (filtered)...")

        fdr_cut = 0.05
        filter_mask = (df_classified["coverage_geom"] >= 0.9) & (
            (df_classified["q_value"] < fdr_cut)
            | (df_classified["Spatial_Bias_Score"] >= df_classified.attrs.get("s_cut", 0))
        )
        filtered_genes = df_classified.loc[filter_mask, "gene"].tolist()

        logger.info(f"  Gene-gene: {len(filtered_genes)}/{len(df_classified)} genes pass filters")

        if len(filtered_genes) > 0:
            df_pairs = score_gene_pairs(
                adata_disease,
                genes=filtered_genes,
                embedding_key=embedding_key,
                config=config,
            )

            pairs_file = outdir / "gene_pairs.csv"
            df_pairs.to_csv(pairs_file, index=False)
            logger.info(f"  Saved {len(df_pairs)} gene pairs to {pairs_file}")
            analysis_meta["n_gene_pairs"] = len(df_pairs)
            analysis_meta["n_genes_in_pairs"] = len(filtered_genes)
        else:
            logger.warning("  No genes passed filters for gene-gene analysis")
            analysis_meta["n_gene_pairs"] = 0
            analysis_meta["n_genes_in_pairs"] = 0
    else:
        analysis_meta["n_gene_pairs"] = 0

    logger.info("  Generating plots...")

    cutoffs = {
        "c_cut": df_classified.attrs.get("c_cut", args.c_cut if args.c_cut else 0.10),
        "s_cut": df_classified.attrs.get(
            "s_cut", args.s_cut if args.s_cut else df_classified["Spatial_Bias_Score"].median()
        ),
        "s_cut_method": df_classified.attrs.get("s_cut_method", "unknown"),
        "fdr_cut": 0.05,
    }

    analysis_meta["cutoffs"] = cutoffs

    plot_cs_scatter(df_classified, cutoffs, outdir, strict=args.strict_plots, seed=config.seed)

    plot_cs_marginals(df_classified, cutoffs, outdir, strict=args.strict_plots)

    plot_top_tables(df_classified, outdir, n_top=15)

    logger.info("  Generating gene exemplar plots...")
    exemplar_genes = select_exemplar_genes(df_classified, n_per_archetype=3)

    for gene_var in exemplar_genes:
        gene_row = df_classified[df_classified["gene"] == gene_var].iloc[0]
        coverage_threshold = gene_row.get("expr_threshold_value", 1.0)

        plot_gene_exemplar(
            adata_disease,
            gene_var,
            gene_row,
            embedding_key,
            config,
            outdir,
            var_to_symbol,
            coverage_threshold,
            strict=args.strict_plots,
        )

    logger.info(f"  Generated {len(exemplar_genes)} exemplar plots")
    analysis_meta["n_exemplar_plots"] = len(exemplar_genes)

    # Generate standardized plotting outputs
    if HAS_STANDARDIZED_PLOTTING:
        logger.info(f"  Generating standardized figures for {disease_name}...")
        try:
            c_cut = cutoffs["c_cut"]
            s_cut = cutoffs["s_cut"]

            generate_kidney_panels(
                df_classified,
                outdir,
                c_cut=c_cut,
                s_cut=s_cut,
                group_by=config.stratify_key,
            )
            logger.info(f"  Standardized figures generated for {disease_name}")
        except Exception as e:
            logger.warning(f"  Could not generate standardized figures: {e}")

    manifest_file = outdir / "manifest.json"
    with open(manifest_file, "w") as f:
        json.dump(analysis_meta, f, indent=2)
    logger.info(f"  Saved manifest to {manifest_file}")

    return analysis_meta


def select_exemplar_genes(df: pd.DataFrame, n_per_archetype: int = 3) -> list[str]:
    """Select representative genes for each archetype.

    Args:
        df: DataFrame with archetype assignments.
        n_per_archetype: Number of genes to select per archetype.

    Returns:
        List of exemplar gene identifiers.

    Notes:
        Selection prioritizes control genes and high `coverage_geom`.
    """
    exemplars = []

    for archetype in df["Archetype"].unique():
        archetype_df = df[df["Archetype"] == archetype].copy()

        archetype_df = archetype_df.sort_values(
            by=["is_control", "coverage_geom"], ascending=[False, False]
        )

        selected = archetype_df.head(n_per_archetype)
        exemplars.extend(selected["gene"].tolist())

    return exemplars


def plot_top_tables(df: pd.DataFrame, outdir: Path, n_top: int = 15):
    """Generate tables of top genes for each archetype.

    Args:
        df: DataFrame with gene scores and archetypes.
        outdir: Output directory.
        n_top: Number of top genes per archetype.
    """
    try:
        figures_dir = outdir / "figures"
        figures_dir.mkdir(exist_ok=True)

        for archetype in df["Archetype"].unique():
            archetype_df = df[df["Archetype"] == archetype].copy()

            top_genes = archetype_df.nlargest(n_top, "Spatial_Bias_Score")

            table_data = top_genes[
                [
                    "gene_symbol",
                    "Coverage",
                    "Spatial_Bias_Score",
                    "coverage_geom",
                    "p_value",
                    "q_value",
                ]
            ].copy()

            table_data.columns = ["Gene", "C", "S", "Cov Geom", "p-val", "q-val"]

            for col in ["C", "S", "Cov Geom"]:
                table_data[col] = table_data[col].apply(lambda x: f"{x:.3f}")
            for col in ["p-val", "q-val"]:
                table_data[col] = table_data[col].apply(
                    lambda x: f"{x:.2e}" if not np.isnan(x) else "N/A"
                )

            fig, ax = plt.subplots(figsize=(10, max(6, n_top * 0.4)))
            ax.axis("off")

            table = ax.table(
                cellText=table_data.values,
                colLabels=table_data.columns,
                cellLoc="left",
                loc="center",
                bbox=[0, 0, 1, 1],
            )

            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)

            for (i, _j), cell in table.get_celld().items():
                if i == 0:
                    cell.set_text_props(weight="bold", color="white")
                    cell.set_facecolor("#4472C4")
                else:
                    cell.set_facecolor("#F2F2F2" if i % 2 == 0 else "white")

            ax.set_title(f"Top {n_top} Genes: {archetype}", fontsize=14, fontweight="bold", pad=20)

            plt.tight_layout()

            safe_name = archetype.replace(":", "").replace(" ", "_").lower()
            outfile = figures_dir / f"table_{safe_name}_top{n_top}.png"
            plt.savefig(outfile, dpi=150, bbox_inches="tight")
            plt.close(fig)

        logger.info(f"  Saved top gene tables to {figures_dir}")

    except Exception as e:
        logger.error(f"Failed to create top gene tables: {e}")


def main():
    """Main analysis pipeline."""
    args = parse_args()

    if args.smoke:
        logger.info("=== SMOKE TEST MODE ===")
        args.max_genes = 50
        args.n_permutations = 0
        args.debug_plots = True
        logger.info("Overrides: max_genes=50, n_permutations=0, debug_plots=True")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    run_meta = {
        "timestamp": datetime.now().isoformat(),
        "args": vars(args),
        "biorsp_version": "3.0",
        "smoke_mode": args.smoke,
    }

    logger.info("[Stage 1] Loading reference dataset...")
    adata = load_reference(args.ref_data)

    try:
        run_meta["input_checksum"] = {
            "file": args.ref_data,
            "sha256": compute_file_checksum(args.ref_data),
        }
    except Exception as e:
        logger.warning(f"Could not compute checksum: {e}")
        run_meta["input_checksum"] = None

    run_meta["n_cells_total"] = int(adata.n_obs)
    run_meta["n_genes_total"] = int(adata.n_vars)

    var_to_symbol, symbol_to_var = build_symbol_mappings(adata)

    logger.info("[Stage 2] Detecting disease metadata...")

    if args.disease_key:
        disease_key = args.disease_key
        if disease_key not in adata.obs.columns:
            raise ValueError(
                f"Disease key '{disease_key}' not found.\n"
                f"Available columns: {list(adata.obs.columns)}"
            )
    else:
        disease_key = discover_disease_key(adata)
        if disease_key is None:
            raise ValueError("Could not auto-discover disease column. Please specify --disease-key")

    run_meta["disease_key"] = disease_key
    logger.info(f"Using disease column: {disease_key}")

    disease_groups = get_disease_groups(adata, disease_key)

    if len(disease_groups) == 0:
        raise ValueError(f"No disease groups found in column '{disease_key}'")

    run_meta["disease_groups"] = {
        name: int(len(indices)) for name, indices in disease_groups.items()
    }

    if args.celltype_filter:
        logger.info("[Stage 3] Filtering by cell type...")

        if not args.celltype_key:
            raise ValueError("Must specify --celltype-key when using --celltype-filter")

        if args.celltype_key not in adata.obs.columns:
            raise ValueError(
                f"Cell type key '{args.celltype_key}' not found.\n"
                f"Available columns: {list(adata.obs.columns)}"
            )

        celltype_mask = adata.obs[args.celltype_key].isin(args.celltype_filter)
        n_match = celltype_mask.sum()
        logger.info(f"Found {n_match} cells matching {args.celltype_filter}")

        if n_match == 0:
            unique_labels = adata.obs[args.celltype_key].unique().tolist()
            raise ValueError(
                f"No cells match labels {args.celltype_filter}.\nAvailable labels: {unique_labels}"
            )

        adata = adata[celltype_mask].copy()

        disease_groups = get_disease_groups(adata, disease_key)
        run_meta["celltype_filter"] = args.celltype_filter
        run_meta["disease_groups_after_filter"] = {
            name: int(len(indices)) for name, indices in disease_groups.items()
        }
    else:
        logger.info("[Stage 3] No cell type filtering (analyzing all cells)")
        run_meta["celltype_filter"] = None

    logger.info("[Stage 4] Detecting embedding...")
    embedding_key = detect_embedding_key(adata, args.embedding_key)
    run_meta["embedding_key"] = embedding_key

    logger.info("[Stage 5] Selecting genes...")
    genes_to_analyze, control_vars, selection_info = select_genes(
        adata,
        args.controls,
        symbol_to_var,
        var_to_symbol,
        args.min_pct,
        args.max_genes,
        args.exclude_patterns,
        args.seed,
        expr_threshold_mode=args.expr_threshold_mode,
        expr_threshold_value=args.expr_threshold_value,
    )
    run_meta["gene_selection"] = selection_info

    if len(genes_to_analyze) == 0:
        raise ValueError("No genes selected. Check --min-pct and --exclude-patterns")

    logger.info("[Stage 6] Configuring BioRSP...")

    config = BioRSPConfig(
        B=args.B,
        delta_deg=args.delta_deg,
        foreground_quantile=args.foreground_quantile,
        expr_threshold_mode=args.expr_threshold_mode,
        expr_threshold_value=args.expr_threshold_value,
        empty_fg_policy=args.empty_fg_policy,
        n_permutations=args.n_permutations,
        seed=args.seed,
        stratify_key=args.donor_key,
    )

    run_meta["config"] = asdict(config)
    logger.info(f"Config: B={config.B}, delta={config.delta_deg}°")
    logger.info(f"  n_permutations={config.n_permutations}, stratify_key={config.stratify_key}")

    logger.info("[Stage 7] Running BioRSP analysis per disease...")

    disease_results = {}

    for disease_name, indices in disease_groups.items():
        adata_disease = adata[indices].copy()

        for col in adata_disease.obs.columns:
            if hasattr(adata_disease.obs[col], "cat"):
                adata_disease.obs[col] = adata_disease.obs[col].astype(str)

        if args.subsample and args.subsample < adata_disease.n_obs:
            logger.info(f"  Subsampling {disease_name} to {args.subsample} cells")
            np.random.seed(args.seed)
            sub_idx = np.random.choice(adata_disease.n_obs, args.subsample, replace=False)
            adata_disease = adata_disease[sub_idx].copy()

        disease_outdir = outdir / disease_name

        try:
            analysis_meta = run_analysis_for_disease(
                adata_disease=adata_disease,
                disease_name=disease_name,
                genes_to_analyze=genes_to_analyze,
                control_vars=control_vars,
                var_to_symbol=var_to_symbol,
                embedding_key=embedding_key,
                config=config,
                args=args,
                outdir=disease_outdir,
            )
            disease_results[disease_name] = analysis_meta

        except Exception as e:
            logger.error(f"Failed to analyze {disease_name}: {e}")
            disease_results[disease_name] = {"error": str(e)}

    run_meta["disease_results"] = disease_results

    logger.info("[Stage 8] Saving summary...")

    summary_file = outdir / "analysis_summary.json"
    with open(summary_file, "w") as f:
        json.dump(run_meta, f, indent=2)
    logger.info(f"Saved summary to {summary_file}")

    readme_file = outdir / "README.md"
    with open(readme_file, "w") as f:
        f.write(generate_readme(run_meta, args))
    logger.info(f"Saved README to {readme_file}")

    logger.info("\n" + "=" * 80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Results saved to: {outdir}")
    logger.info(f"Disease groups analyzed: {list(disease_results.keys())}")
    logger.info("=" * 80 + "\n")


def generate_readme(run_meta: dict, args: argparse.Namespace) -> str:
    """Generate README documentation for the analysis.

    Args:
        run_meta: Run metadata dictionary.
        args: Parsed CLI arguments.

    Returns:
        Markdown text describing the analysis outputs.
    """
    readme = f"""# Disease-Stratified BioRSP Analysis


**Date**: {run_meta["timestamp"]}
**Input**: {args.ref_data}
**Total Cells**: {run_meta["n_cells_total"]:,}
**Total Genes**: {run_meta["n_genes_total"]:,}


**Disease Column**: {run_meta["disease_key"]}

"""

    if run_meta.get("celltype_filter"):
        readme += f"**Cell Type Filter**: {', '.join(args.celltype_filter)}\n\n"

    readme += "### Disease Groups\n\n"
    for disease, n_cells in run_meta.get(
        "disease_groups_after_filter", run_meta["disease_groups"]
    ).items():
        readme += f"- **{disease}**: {n_cells:,} cells\n"

    readme += "\n## Gene Selection\n\n"
    readme += f"- Control genes: {run_meta['gene_selection']['n_control']}\n"
    readme += f"- Discovery genes: {run_meta['gene_selection']['n_discovery']}\n"
    readme += f"- Total analyzed: {run_meta['gene_selection']['n_total_selected']}\n"
    readme += f"- Min expression: {run_meta['gene_selection']['min_pct']:.1%}\n"

    readme += "\n## BioRSP Configuration\n\n"
    readme += f"- Angular sectors (B): {run_meta['config']['B']}\n"
    readme += f"- Sector width: {run_meta['config']['delta_deg']}°\n"
    readme += f"- Permutations: {run_meta['config']['n_permutations']}\n"
    readme += f"- Donor stratification: {run_meta['config']['stratify_key']}\n"

    readme += "\n## Results by Disease\n\n"
    for disease, results in run_meta.get("disease_results", {}).items():
        readme += f"### {disease}\n\n"
        if "error" in results:
            readme += f"**Error**: {results['error']}\n\n"
        else:
            readme += f"- Cells analyzed: {results['n_cells']:,}\n"
            readme += f"- Genes scored: {results.get('n_genes_scored', 0)}\n"
            if results.get("archetype_counts"):
                readme += "- Archetypes:\n"
                for arch, count in results["archetype_counts"].items():
                    readme += f"  - {arch}: {count}\n"
            readme += f"- Results: `{disease}/gene_scores.csv`\n"
            if results.get("n_gene_pairs", 0) > 0:
                readme += f"- Gene pairs: `{disease}/gene_pairs.csv`\n"
            readme += "\n"

    readme += f"""## File Structure

```
{args.outdir}/
├── analysis_summary.json       # Complete metadata
├── README.md                    # This file
├── normal/                      # Normal condition results
│   ├── gene_scores.csv
│   ├── gene_pairs.csv (if --do-genegene)
│   └── radar_plots/
├── acute_kidney_failure/        # AKI results
│   └── ...
└── chronic_kidney_disease/      # CKD results
    └── ...
```


Results can be compared across disease conditions to identify:
- Disease-specific spatial patterns
- Genes with altered localization in disease
- Condition-specific biomarkers

For TAL-specific analysis, use:
```bash
python run_disease_stratified_analysis.py \\
  --ref-data data/kpmp.h5ad \\
  --outdir results/tal_disease \\
  --celltype-key subclass.l1 \\
  --celltype-filter TAL \\
  --max-genes 100
```
"""

    return readme


if __name__ == "__main__":
    main()
