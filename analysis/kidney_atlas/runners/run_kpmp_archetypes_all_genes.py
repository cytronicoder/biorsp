"""
KPMP All-Gene Archetypes Pipeline

End-to-end pipeline to score and classify (almost) all genes using BioRSP's
two primary scores:
- Coverage score C_g: fraction of cells with expression >= threshold
- Spatial bias score S_g: weighted RMS of R(θ) on bg-supported sectors

Outputs a biologist-friendly one-page "story" with tables + figures.
"""

import argparse
import json
import logging
import multiprocessing as mp
import time
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from scipy import sparse
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

from biorsp.plotting.spec import ARCHETYPE_COLORS, ARCHETYPE_ORDER  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import pyarrow as pa  # noqa: F401
    import pyarrow.parquet as pq  # noqa: F401

    HAS_PARQUET = True
except ImportError:
    HAS_PARQUET = False

try:
    from joblib import Parallel, delayed

    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

TIMINGS = {}


def profile_stage(stage_name: str):
    """Create a decorator that profiles execution time of a function.

    Args:
        stage_name: Name of the stage being profiled.

    Returns:
        Decorator that records elapsed time in the global `TIMINGS` dictionary.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not hasattr(wrapper, "_profiling_enabled"):
                wrapper._profiling_enabled = False

            if wrapper._profiling_enabled:
                start = time.time()
                result = func(*args, **kwargs)
                elapsed = time.time() - start
                TIMINGS[stage_name] = elapsed
                logger.info(f"[PROFILE] {stage_name}: {elapsed:.2f}s")
                return result
            else:
                return func(*args, **kwargs)

        def enable_profiling(enabled: bool = True):
            wrapper._profiling_enabled = enabled

        wrapper.enable_profiling = enable_profiling
        return wrapper

    return decorator


rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42

ARCHETYPE_NAMES = {
    (True, False): "Ubiquitous",  # High C, Low S
    (False, True): "Patchy",  # Low C, High S
    (True, True): "Gradient",  # High C, High S
    (False, False): "Basal",  # Low C, Low S
}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="KPMP All-Gene Archetypes Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--h5ad", type=str, default="data/kpmp.h5ad", help="Path to KPMP AnnData h5ad file"
    )

    parser.add_argument(
        "--embedding-key",
        type=str,
        default=None,
        help="Key in adata.obsm (auto-detect if not specified)",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        help="Pandas eval query on adata.obs (e.g., 'cell_type == \"TAL\"')",
    )

    parser.add_argument(
        "--outdir", type=str, default="results/kpmp_archetypes", help="Output directory"
    )

    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--max-cells",
        type=int,
        default=250000,
        help="Max cells (deterministic subsample if larger)",
    )
    parser.add_argument("--chunk-size", type=int, default=500, help="Genes per checkpoint chunk")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoints")

    parser.add_argument("--delta-deg", type=float, default=60.0)
    parser.add_argument("--B", type=int, default=72)
    parser.add_argument("--foreground-quantile", type=float, default=0.90)
    parser.add_argument("--empty-fg-policy", type=str, default="zero")

    parser.add_argument(
        "--expr-threshold-mode",
        type=str,
        default="detect",
        choices=["detect", "fixed"],
        help="How to determine expression threshold for C",
    )
    parser.add_argument(
        "--expr-threshold-value",
        type=float,
        default=None,
        help="Fixed expression threshold (if mode=fixed)",
    )

    parser.add_argument(
        "--min-coverage", type=float, default=0.005, help="Min coverage for gene filtering"
    )
    parser.add_argument(
        "--min-nonzero", type=int, default=50, help="Min nonzero cells for gene filtering"
    )

    parser.add_argument(
        "--derive-thresholds", action="store_true", default=True, help="Auto-derive c_cut and s_cut"
    )
    parser.add_argument("--c-cut", type=float, default=None)
    parser.add_argument("--s-cut", type=float, default=None)

    parser.add_argument(
        "--compute-pvalues",
        action="store_true",
        default=False,
        help="Compute permutation p-values for top genes",
    )
    parser.add_argument(
        "--pvalue-topk", type=int, default=200, help="Number of top genes for p-value computation"
    )
    parser.add_argument("--n-permutations", type=int, default=500)

    parser.add_argument(
        "--skip-reliability",
        action="store_true",
        default=False,
        help="Skip reliability checks (subsample stability, cross-embedding)",
    )

    parser.add_argument(
        "--n-workers",
        type=int,
        default=0,
        help="Number of parallel workers (0=single process, -1=all cores)",
    )
    parser.add_argument(
        "--use-parquet",
        action="store_true",
        default=HAS_PARQUET,
        help="Use parquet for checkpoints (faster I/O)",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        default=False,
        help="Enable profiling and log stage timings",
    )

    return parser.parse_args()


def create_gene_name_mapping(adata: ad.AnnData) -> Dict[str, str]:
    """Create mapping from ENSG IDs to gene names.

    Uses `adata.var["feature_name"]` if available, otherwise falls back to
    `adata.var_names`.

    Args:
        adata: AnnData object.

    Returns:
        Mapping from gene IDs to display names.
    """
    mapping = {}
    if "feature_name" in adata.var.columns:
        for ensg_id, gene_name in zip(adata.var_names, adata.var["feature_name"]):
            if gene_name and not gene_name.startswith("ENSG"):
                mapping[ensg_id] = gene_name
            else:
                mapping[ensg_id] = ensg_id
    else:
        mapping = {ensg_id: ensg_id for ensg_id in adata.var_names}

    return mapping


def add_gene_names_to_dataframe(
    df: pd.DataFrame, gene_name_mapping: Dict[str, str]
) -> pd.DataFrame:
    """Add a `gene_name` column using a gene ID mapping.

    Args:
        df: Input DataFrame with a `gene` column.
        gene_name_mapping: Mapping from gene ID to display name.

    Returns:
        DataFrame with an inserted `gene_name` column.
    """
    df = df.copy()
    df["gene_name"] = df["gene"].map(lambda x: gene_name_mapping.get(x, x))
    cols = df.columns.tolist()
    gene_idx = cols.index("gene")
    cols.insert(gene_idx + 1, cols.pop(cols.index("gene_name")))
    return df[cols]


def get_gene_expression_fast(adata: ad.AnnData, gene_idx: int) -> np.ndarray:
    """Extract a gene expression vector by index.

    Args:
        adata: AnnData object.
        gene_idx: Column index of the gene in `adata.X`.

    Returns:
        Expression vector of shape (n_cells,).
    """
    if sparse.issparse(adata.X):
        return adata.X[:, gene_idx].toarray().flatten()
    else:
        return adata.X[:, gene_idx].copy()


def compute_coverage_vectorized(
    X: np.ndarray, gene_indices: List[int], threshold: float
) -> np.ndarray:
    """Compute coverage for multiple genes in a vectorized manner.

    Args:
        X: Expression matrix (cells × genes).
        gene_indices: Indices of genes to compute coverage for.
        threshold: Expression threshold.

    Returns:
        Coverage values for each gene.
    """
    if sparse.issparse(X):
        X_sub = X[:, gene_indices]
        counts = np.array((X_sub >= threshold).sum(axis=0)).flatten()
    else:
        X_sub = X[:, gene_indices]
        counts = np.sum(X_sub >= threshold, axis=0)

    return counts / X.shape[0]


@profile_stage("gene_filtering")
def detect_expression_threshold(adata: ad.AnnData) -> Tuple[float, str]:
    """Detect an expression threshold based on data type.

    Args:
        adata: AnnData object.

    Returns:
        Tuple of `(threshold, method_label)`.
    """
    if sparse.issparse(adata.X):
        sample = adata.X[:1000].toarray() if adata.n_obs > 1000 else adata.X.toarray()
    else:
        sample = adata.X[:1000] if adata.n_obs > 1000 else adata.X

    is_integers = np.allclose(sample, np.round(sample)) and np.max(sample) > 1.0

    if is_integers:
        return 1.0, "detect_count"
    else:
        return 1e-6, "detect_continuous"


def fast_filter_genes(
    adata: ad.AnnData,
    min_coverage: float = 0.005,
    min_nonzero: int = 50,
    threshold: float = 1.0,
) -> Tuple[List[str], Dict[str, int]]:
    """Filter genes based on coverage and nonzero counts.

    Args:
        adata: AnnData object.
        min_coverage: Minimum coverage threshold.
        min_nonzero: Minimum number of nonzero cells.
        threshold: Expression threshold for coverage.

    Returns:
        Tuple of `(filtered_genes, stats_dict)`.
    """
    n_cells = adata.n_obs

    if sparse.issparse(adata.X):
        X = adata.X.tocsc()
        nonzero_counts = np.array((X > 0).sum(axis=0)).flatten()
        expr_counts = np.array((threshold <= X).sum(axis=0)).flatten()
    else:
        nonzero_counts = np.sum(adata.X > 0, axis=0)
        expr_counts = np.sum(threshold <= adata.X, axis=0)

    coverage = expr_counts / n_cells

    mask_coverage = coverage >= min_coverage
    mask_nonzero = nonzero_counts >= min_nonzero
    mask_final = mask_coverage & mask_nonzero

    filtered_genes = adata.var_names[mask_final].tolist()

    stats = {
        "total_genes": len(adata.var_names),
        "passed_coverage": int(np.sum(mask_coverage)),
        "passed_nonzero": int(np.sum(mask_nonzero)),
        "passed_both": len(filtered_genes),
        "filtered_out": len(adata.var_names) - len(filtered_genes),
    }

    logger.info(
        f"Gene filtering: {stats['total_genes']} → {stats['passed_both']} genes "
        f"(min_coverage={min_coverage}, min_nonzero={min_nonzero})"
    )

    return filtered_genes, stats


def derive_c_cut(df: pd.DataFrame, default: float = 0.10) -> float:
    """Derive the coverage cutoff `c_cut`.

    Args:
        df: DataFrame with a `Coverage` column.
        default: Default cutoff to use when the median is higher.

    Returns:
        Coverage cutoff value.

    Notes:
        If the median Coverage is below `default`, the cutoff is set to
        `max(0.05, median(Coverage))`.
    """
    median_c = df["Coverage"].median()

    if median_c < default:
        c_cut = max(0.05, median_c)
        logger.info(f"Median C ({median_c:.3f}) < {default}, using c_cut={c_cut:.3f}")
    else:
        c_cut = default
        logger.info(f"Using default c_cut={c_cut:.3f}")

    return c_cut


def derive_s_cut_from_null(
    adata: ad.AnnData,
    context,  # BioRSPContext
    genes: List[str],
    config,  # BioRSPConfig
    M: int = 200,
    K: int = 200,
    seed: int = 42,
) -> Tuple[float, float, Dict[str, Any]]:
    """Derive spatial score cutoffs from a null distribution.

    Args:
        adata: AnnData object.
        context: BioRSP context for scoring.
        genes: List of genes to sample.
        config: BioRSP configuration.
        M: Number of genes to sample for null estimation.
        K: Number of permutations per gene.
        seed: Random seed.

    Returns:
        Tuple of `(s_cut_95, s_cut_99, metadata)`.
    """
    from biorsp.preprocess.context import score_gene_with_context

    rng = np.random.default_rng(seed)
    sample_genes = rng.choice(genes, size=min(M, len(genes)), replace=False)

    all_null_scores = []

    for gene in tqdm(sample_genes, desc="Deriving s_cut from null"):
        result = score_gene_with_context(
            adata, gene, context, config, compute_pvalue=True, n_permutations=K
        )

        if "null_mean" in result and not np.isnan(result.get("null_mean", np.nan)):
            null_mean = result["null_mean"]
            null_sd = result.get("null_sd", 0.01)
            null_samples = rng.normal(null_mean, null_sd, size=10)
            all_null_scores.extend(null_samples)

    if len(all_null_scores) == 0:
        logger.warning("Could not derive null distribution, using default s_cut=0.05")
        return 0.05, 0.10, {"method": "default", "reason": "no_null_scores"}

    null_array = np.array(all_null_scores)
    s_cut_95 = float(np.percentile(null_array, 95))
    s_cut_99 = float(np.percentile(null_array, 99))

    logger.info(f"Derived s_cut: 95th={s_cut_95:.4f}, 99th={s_cut_99:.4f}")

    metadata = {
        "method": "null_permutation",
        "M_genes_sampled": len(sample_genes),
        "K_permutations": K,
        "n_null_scores": len(all_null_scores),
        "null_mean": float(np.mean(null_array)),
        "null_std": float(np.std(null_array)),
        "s_cut_95": s_cut_95,
        "s_cut_99": s_cut_99,
    }

    return s_cut_95, s_cut_99, metadata


def derive_s_cut_simple(df: pd.DataFrame, quantile: float = 0.75) -> float:
    """Derive a simple `s_cut` from observed Spatial_Bias_Score values.

    Args:
        df: DataFrame with `Spatial_Bias_Score` and `Coverage`.
        quantile: Quantile used as a fallback threshold.

    Returns:
        Spatial score cutoff.
    """
    s_values = df["Spatial_Bias_Score"].dropna()
    if len(s_values) == 0:
        return 0.05

    low_c_mask = df["Coverage"] < 0.02
    if low_c_mask.sum() > 50:
        null_proxy = df.loc[low_c_mask, "Spatial_Bias_Score"].dropna()
        s_cut = float(np.percentile(null_proxy, 95)) if len(null_proxy) > 0 else 0.05
    else:
        s_cut = float(np.percentile(s_values, 50))

    return max(0.02, s_cut)


def classify_genes(
    df: pd.DataFrame,
    c_cut: float,
    s_cut: float,
    fdr_cut: float = 0.05,
) -> pd.DataFrame:
    """Classify genes into 2×2 archetypes.

    Args:
        df: DataFrame with `Coverage` and `Spatial_Bias_Score`.
        c_cut: Coverage cutoff.
        s_cut: Spatial score cutoff.
        fdr_cut: FDR cutoff for marking significance (if `q_value` exists).

    Returns:
        DataFrame with archetype labels and cutoff columns.
    """
    df = df.copy()

    high_c = df["Coverage"] >= c_cut
    high_s = df["Spatial_Bias_Score"] >= s_cut

    def get_archetype(hc, hs):
        return ARCHETYPE_NAMES.get((hc, hs), "Unknown")

    df["Archetype"] = [get_archetype(c, s) for c, s in zip(high_c, high_s)]
    df["c_cut_used"] = c_cut
    df["s_cut_used"] = s_cut

    if "q_value" in df.columns:
        df["significant"] = df["q_value"] < fdr_cut

    return df


@dataclass
class SharedContext:
    """Lightweight context for sharing across processes."""

    h5ad_path: str
    gene_names: List[str]
    config_dict: Dict[str, Any]
    context_dict: Dict[str, Any]
    seed: int


def score_gene_chunk_worker(
    chunk_id: int,
    genes: List[str],
    shared_ctx: SharedContext,
) -> List[Dict[str, Any]]:
    """Score a chunk of genes in a worker process.

    Args:
        chunk_id: Chunk index.
        genes: List of genes to score.
        shared_ctx: Shared context containing paths and configuration.

    Returns:
        List of per-gene result dictionaries.
    """
    import anndata as ad
    import numpy as np

    from biorsp.preprocess.context import BioRSPContext, score_gene_with_context
    from biorsp.utils.config import BioRSPConfig

    adata = ad.read_h5ad(shared_ctx.h5ad_path)
    config = BioRSPConfig(**shared_ctx.config_dict)

    context = BioRSPContext(
        coords=np.array(shared_ctx.context_dict["coords"]),
        center=np.array(shared_ctx.context_dict["center"]),
        r_norm=np.array(shared_ctx.context_dict["r_norm"]),
        theta=np.array(shared_ctx.context_dict["theta"]),
        sector_indices=[
            np.array(s) if s is not None else None
            for s in shared_ctx.context_dict["sector_indices"]
        ],
        sector_sort_indices=[
            np.array(s) if s is not None else None
            for s in shared_ctx.context_dict["sector_sort_indices"]
        ],
        n_cells=shared_ctx.context_dict["n_cells"],
        norm_stats=shared_ctx.context_dict["norm_stats"],
        embedding_key=shared_ctx.context_dict.get("embedding_key", "X_umap"),
        subset_query=shared_ctx.context_dict.get("subset_query"),
        config=config,
        stratify_labels=shared_ctx.context_dict.get("stratify_labels"),
    )

    np.random.default_rng(np.random.SeedSequence(shared_ctx.seed + chunk_id))

    results = []
    for gene in genes:
        result = score_gene_with_context(adata, gene, context, config, compute_pvalue=False)
        results.append(result)

    return results


def save_chunk(df: pd.DataFrame, path: Path, use_parquet: bool = False):
    """Save chunk to disk."""
    if use_parquet and HAS_PARQUET:
        df.to_parquet(path.with_suffix(".parquet"), index=False, engine="pyarrow")
    else:
        df.to_csv(path.with_suffix(".csv"), index=False)


def load_chunk(path: Path, use_parquet: bool = False) -> pd.DataFrame:
    """Load chunk from disk."""
    parquet_path = path.with_suffix(".parquet")
    csv_path = path.with_suffix(".csv")

    if use_parquet and HAS_PARQUET and parquet_path.exists():
        return pd.read_parquet(parquet_path, engine="pyarrow")
    elif csv_path.exists():
        return pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(f"No checkpoint found at {path}")


@profile_stage("gene_scoring")
def score_all_genes_parallel(
    adata: ad.AnnData,
    genes: List[str],
    context,  # BioRSPContext
    config,  # BioRSPConfig
    outdir: Path,
    chunk_size: int = 500,
    resume: bool = False,
    n_workers: int = 0,
    use_parquet: bool = False,
    h5ad_path: Optional[str] = None,
) -> pd.DataFrame:
    """Score all genes with optional parallelization.

    Parameters
    ----------
    n_workers : int
        Number of parallel workers. 0 = single process, -1 = all cores
    use_parquet : bool
        Use parquet for faster I/O
    h5ad_path : str, optional
        Path to h5ad file for worker processes to reload
    """
    checkpoint_dir = outdir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    n_chunks = (len(genes) + chunk_size - 1) // chunk_size

    if n_workers == -1:
        n_workers = mp.cpu_count()
    elif n_workers == 0:
        n_workers = 1  # Single process

    logger.info(f"Scoring {len(genes)} genes in {n_chunks} chunks using {n_workers} worker(s)")

    if n_workers == 1:
        return score_all_genes_chunked_serial(
            adata, genes, context, config, outdir, chunk_size, resume, use_parquet
        )

    if h5ad_path is None:
        raise ValueError("h5ad_path required for parallel execution")

    context_dict = {
        "coords": context.coords.tolist(),
        "center": context.center.tolist(),
        "r_norm": context.r_norm.tolist(),
        "theta": context.theta.tolist(),
        "sector_indices": [s.tolist() if s is not None else None for s in context.sector_indices],
        "sector_sort_indices": [
            s.tolist() if s is not None else None for s in context.sector_sort_indices
        ],
        "n_cells": context.n_cells,
        "norm_stats": context.norm_stats,
        "embedding_key": context.embedding_key,
        "subset_query": context.subset_query,
        "stratify_labels": context.stratify_labels,
    }

    shared_ctx = SharedContext(
        h5ad_path=str(h5ad_path),
        gene_names=adata.var_names.tolist(),
        config_dict=asdict(config),
        context_dict=context_dict,
        seed=config.seed,
    )

    chunks_to_process = []
    for i in range(n_chunks):
        chunk_file = checkpoint_dir / f"chunk_{i:04d}"

        if resume and (
            chunk_file.with_suffix(".parquet").exists() or chunk_file.with_suffix(".csv").exists()
        ):
            continue

        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(genes))
        chunk_genes = genes[start_idx:end_idx]
        chunks_to_process.append((i, chunk_genes))

    if chunks_to_process:
        logger.info(f"Processing {len(chunks_to_process)} chunks in parallel")

        if HAS_JOBLIB:
            results = Parallel(n_jobs=n_workers, backend="loky", verbose=10)(
                delayed(score_gene_chunk_worker)(chunk_id, chunk_genes, shared_ctx)
                for chunk_id, chunk_genes in chunks_to_process
            )

            for (chunk_id, _), chunk_results in zip(chunks_to_process, results):
                chunk_df = pd.DataFrame(chunk_results)
                chunk_file = checkpoint_dir / f"chunk_{chunk_id:04d}"
                save_chunk(chunk_df, chunk_file, use_parquet)
        else:
            from concurrent.futures import ProcessPoolExecutor, as_completed

            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {
                    executor.submit(
                        score_gene_chunk_worker, chunk_id, chunk_genes, shared_ctx
                    ): chunk_id
                    for chunk_id, chunk_genes in chunks_to_process
                }

                with tqdm(total=len(futures), desc="Parallel scoring") as pbar:
                    for future in as_completed(futures):
                        chunk_id = futures[future]
                        chunk_results = future.result()
                        chunk_df = pd.DataFrame(chunk_results)
                        chunk_file = checkpoint_dir / f"chunk_{chunk_id:04d}"
                        save_chunk(chunk_df, chunk_file, use_parquet)
                        pbar.update(1)

    all_results = []
    for i in range(n_chunks):
        chunk_file = checkpoint_dir / f"chunk_{i:04d}"
        chunk_df = load_chunk(chunk_file, use_parquet)
        all_results.append(chunk_df)

    df = pd.concat(all_results, ignore_index=True)
    return df


def score_all_genes_chunked_serial(
    adata: ad.AnnData,
    genes: List[str],
    context,  # BioRSPContext
    config,  # BioRSPConfig
    outdir: Path,
    chunk_size: int = 500,
    resume: bool = False,
    use_parquet: bool = False,
) -> pd.DataFrame:
    """Score all genes in chunks (single process) with checkpointing."""
    from biorsp.preprocess.context import score_gene_with_context

    checkpoint_dir = outdir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    n_chunks = (len(genes) + chunk_size - 1) // chunk_size
    all_results = []

    for i in range(n_chunks):
        chunk_file = checkpoint_dir / f"chunk_{i:04d}"

        if resume and (
            chunk_file.with_suffix(".parquet").exists() or chunk_file.with_suffix(".csv").exists()
        ):
            logger.info(f"Loading checkpoint {chunk_file}")
            chunk_df = load_chunk(chunk_file, use_parquet)
            all_results.append(chunk_df)
            continue

        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(genes))
        chunk_genes = genes[start_idx:end_idx]

        chunk_results = []
        for gene in tqdm(chunk_genes, desc=f"Chunk {i + 1}/{n_chunks}"):
            result = score_gene_with_context(adata, gene, context, config, compute_pvalue=False)
            chunk_results.append(result)

        chunk_df = pd.DataFrame(chunk_results)
        save_chunk(chunk_df, chunk_file, use_parquet)
        all_results.append(chunk_df)

    df = pd.concat(all_results, ignore_index=True)
    return df


def score_all_genes_chunked(
    adata: ad.AnnData,
    genes: List[str],
    context,  # BioRSPContext
    config,  # BioRSPConfig
    outdir: Path,
    chunk_size: int = 500,
    resume: bool = False,
) -> pd.DataFrame:
    """Score all genes in chunks with checkpointing.

    DEPRECATED: Use score_all_genes_parallel instead.
    """
    from biorsp.preprocess.context import score_gene_with_context

    checkpoint_dir = outdir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    n_chunks = (len(genes) + chunk_size - 1) // chunk_size
    all_results = []

    for i in range(n_chunks):
        chunk_file = checkpoint_dir / f"chunk_{i:04d}.csv"

        if resume and chunk_file.exists():
            logger.info(f"Loading checkpoint {chunk_file}")
            chunk_df = pd.read_csv(chunk_file)
            all_results.append(chunk_df)
            continue

        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(genes))
        chunk_genes = genes[start_idx:end_idx]

        chunk_results = []
        for gene in tqdm(chunk_genes, desc=f"Chunk {i + 1}/{n_chunks}"):
            result = score_gene_with_context(adata, gene, context, config, compute_pvalue=False)
            chunk_results.append(result)

        chunk_df = pd.DataFrame(chunk_results)
        chunk_df.to_csv(chunk_file, index=False)
        all_results.append(chunk_df)

    df = pd.concat(all_results, ignore_index=True)
    return df


def add_pvalues_to_top_genes(
    adata: ad.AnnData,
    df: pd.DataFrame,
    context,
    config,
    topk: int = 200,
    n_permutations: int = 500,
) -> pd.DataFrame:
    """Add permutation p-values to top genes by spatial bias score."""
    from biorsp.preprocess.context import score_gene_with_context

    df = df.copy()
    df["p_value"] = np.nan
    df["q_value"] = np.nan

    top_genes = df.nlargest(topk, "Spatial_Bias_Score")["gene"].tolist()

    logger.info(f"Computing p-values for top {len(top_genes)} genes")

    pvals = {}
    for gene in tqdm(top_genes, desc="Computing p-values"):
        result = score_gene_with_context(
            adata, gene, context, config, compute_pvalue=True, n_permutations=n_permutations
        )
        pvals[gene] = result.get("p_value", np.nan)

    for gene, pval in pvals.items():
        df.loc[df["gene"] == gene, "p_value"] = pval

    valid_mask = df["p_value"].notna()
    if valid_mask.sum() > 0:
        _, qvals, _, _ = multipletests(df.loc[valid_mask, "p_value"], method="fdr_bh")
        df.loc[valid_mask, "q_value"] = qvals

    return df


def check_subsample_stability(
    adata: ad.AnnData,
    genes: List[str],
    embedding_key: str,
    config,
    seed: int = 42,
    subsample_frac: float = 0.7,
    n_genes: int = 500,
) -> Dict[str, Any]:
    """Check S score stability across two independent subsamples."""
    from biorsp.preprocess.context import prepare_context, score_gene_with_context

    rng = np.random.default_rng(seed)
    n_cells = adata.n_obs
    n_sub = int(n_cells * subsample_frac)

    idx1 = rng.choice(n_cells, size=n_sub, replace=False)
    idx2 = rng.choice(n_cells, size=n_sub, replace=False)

    adata1 = adata[idx1].copy()
    adata2 = adata[idx2].copy()

    _, context1 = prepare_context(adata1, embedding_key, config=config)
    _, context2 = prepare_context(adata2, embedding_key, config=config)

    test_genes = genes[:n_genes] if len(genes) > n_genes else genes

    s1, s2 = [], []
    for gene in tqdm(test_genes, desc="Subsample stability"):
        r1 = score_gene_with_context(adata1, gene, context1, config)
        r2 = score_gene_with_context(adata2, gene, context2, config)
        s1.append(r1.get("Spatial_Bias_Score", np.nan))
        s2.append(r2.get("Spatial_Bias_Score", np.nan))

    s1, s2 = np.array(s1), np.array(s2)
    valid = ~(np.isnan(s1) | np.isnan(s2))

    if valid.sum() < 10:
        return {"error": "insufficient_valid_genes", "spearman_r": np.nan}

    corr, pval = spearmanr(s1[valid], s2[valid])
    delta_s = np.abs(s1[valid] - s2[valid])

    return {
        "spearman_r": float(corr),
        "spearman_pval": float(pval),
        "n_genes_tested": int(valid.sum()),
        "median_delta_s": float(np.median(delta_s)),
        "mean_delta_s": float(np.mean(delta_s)),
    }


def check_cross_embedding(
    adata: ad.AnnData,
    genes: List[str],
    config,
    n_genes: int = 200,
) -> Optional[Dict[str, Any]]:
    """Check S score consistency between UMAP and t-SNE."""
    from biorsp.preprocess.context import prepare_context, score_gene_with_context

    has_umap = any(k in adata.obsm for k in ["X_umap", "X_UMAP"])
    has_tsne = any(k in adata.obsm for k in ["X_tsne", "X_tSNE"])

    if not (has_umap and has_tsne):
        logger.info("Cross-embedding check skipped (need both UMAP and t-SNE)")
        return None

    umap_key = "X_umap" if "X_umap" in adata.obsm else "X_UMAP"
    tsne_key = "X_tsne" if "X_tsne" in adata.obsm else "X_tSNE"

    _, ctx_umap = prepare_context(adata, umap_key, config=config)
    _, ctx_tsne = prepare_context(adata, tsne_key, config=config)

    test_genes = genes[:n_genes] if len(genes) > n_genes else genes

    s_umap, s_tsne = [], []
    for gene in tqdm(test_genes, desc="Cross-embedding check"):
        r_umap = score_gene_with_context(adata, gene, ctx_umap, config)
        r_tsne = score_gene_with_context(adata, gene, ctx_tsne, config)
        s_umap.append(r_umap.get("Spatial_Bias_Score", np.nan))
        s_tsne.append(r_tsne.get("Spatial_Bias_Score", np.nan))

    s_umap, s_tsne = np.array(s_umap), np.array(s_tsne)
    valid = ~(np.isnan(s_umap) | np.isnan(s_tsne))

    if valid.sum() < 10:
        return {"error": "insufficient_valid_genes"}

    corr, pval = spearmanr(s_umap[valid], s_tsne[valid])

    return {
        "umap_key": umap_key,
        "tsne_key": tsne_key,
        "spearman_r": float(corr),
        "spearman_pval": float(pval),
        "n_genes_tested": int(valid.sum()),
    }


def plot_cs_scatter(
    df: pd.DataFrame,
    c_cut: float,
    s_cut: float,
    outdir: Path,
    max_points: int = 5000,
    seed: int = 42,
):
    """Create C-S scatter plot with quadrant boundaries."""
    fig, ax = plt.subplots(figsize=(10, 10))

    if len(df) > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(df), size=max_points, replace=False)
        plot_df = df.iloc[idx].copy()
    else:
        plot_df = df.copy()

    for archetype in ARCHETYPE_ORDER:
        if archetype not in ARCHETYPE_COLORS:
            continue
        color = ARCHETYPE_COLORS[archetype]
        mask = plot_df["Archetype"] == archetype
        ax.scatter(
            plot_df.loc[mask, "Coverage"],
            plot_df.loc[mask, "Spatial_Bias_Score"],
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
    y_max = plot_df["Spatial_Bias_Score"].quantile(0.99) * 1.2
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
    ax.hist(df["Spatial_Bias_Score"], bins=50, color="darkorange", alpha=0.7, edgecolor="white")
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
        fig.savefig(outdir / f"fig_CS_marginals.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_top_tables(df: pd.DataFrame, outdir: Path, n_top: int = 15):
    """Create figure showing top genes per archetype."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    gene_col = "gene_name" if "gene_name" in df.columns else "gene"

    for idx, archetype in enumerate(ARCHETYPE_ORDER):
        color = ARCHETYPE_COLORS.get(archetype, "#888888")
        ax = axes[idx]
        subset = df[df["Archetype"] == archetype].copy()

        if len(subset) == 0:
            ax.axis("off")
            ax.set_title(f"{archetype}\n(n=0 genes)", fontsize=16, pad=32, loc="center")
            ax.text(
                0.5,
                0.5,
                "No genes in this category",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=16,
                color="gray",
            )
            continue

        if archetype == "Focal Marker" or archetype == "Regional Gradient":
            subset = subset.nlargest(min(n_top, len(subset)), "Spatial_Bias_Score")
        elif archetype == "Ubiquitous Uniform":
            subset = subset.nlargest(min(n_top, len(subset)), "Coverage")
        else:  # Rare Scattered
            subset = subset.nsmallest(min(n_top, len(subset)), "Coverage")

        ax.axis("off")

        display_cols = [gene_col, "Coverage", "Spatial_Bias_Score"]
        if "spatial_sign" in subset.columns:
            display_cols.append("spatial_sign")
            col_labels = ["Gene", "C", "S", "Sign"]
        else:
            col_labels = ["Gene", "C", "S"]

        table_data = subset[display_cols].round(4).values

        table = ax.table(
            cellText=table_data[: min(n_top, len(subset))],
            colLabels=col_labels,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)

        for j in range(len(col_labels)):
            table[(0, j)].set_facecolor(color)
            table[(0, j)].set_text_props(color="white", weight="bold")

        ax.set_title(
            f"{archetype}\n(n={len(df[df['Archetype'] == archetype]):,} genes)",
            fontsize=16,
            pad=32,
            loc="center",
        )

    plt.suptitle("Top Genes per Archetype", fontsize=18, y=1.05)
    plt.subplots_adjust(hspace=0.5, wspace=0.3, top=0.93, bottom=0.05)

    for ext in ["png", "pdf"]:
        fig.savefig(outdir / f"fig_top_tables.{ext}", dpi=300, bbox_inches="tight")
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

    selected = {}  # archetype -> (gene_id, gene_name)
    for archetype in ARCHETYPE_NAMES.values():
        subset = df[df["Archetype"] == archetype]
        if len(subset) == 0:
            continue

        if archetype == "Focal Marker" or archetype == "Regional Gradient":
            row = subset.nlargest(1, "Spatial_Bias_Score").iloc[0]
        else:
            c_mean = subset["Coverage"].mean()
            s_mean = subset["Spatial_Bias_Score"].mean()
            dist = np.sqrt(
                (subset["Coverage"] - c_mean) ** 2 + (subset["Spatial_Bias_Score"] - s_mean) ** 2
            )
            row = subset.iloc[dist.argmin()]

        gene_id = row["gene"]
        gene_name = row["gene_name"] if has_gene_name else gene_id
        selected[archetype] = (gene_id, gene_name)

    if len(selected) == 0:
        logger.warning("No archetypes have genes - skipping example plots")
        return

    n_archetypes = len(selected)
    n_cols = 4
    n_rows = (n_archetypes * 2 + n_cols - 1) // n_cols
    fig = plt.figure(figsize=(16, 4 * n_rows))

    for i, (archetype, (gene_id, gene_name)) in enumerate(selected.items()):
        row = (i * 2) // n_cols
        base_pos = (row * n_cols) + ((i * 2) % n_cols) + 1

        gene_idx = adata.var_names.get_loc(gene_id)
        if sparse.issparse(adata.X):
            x = adata.X[:, gene_idx].toarray().flatten()
        else:
            x = adata.X[:, gene_idx].copy()

        coords = context.coords
        gene_row = df[df["gene"] == gene_id].iloc[0]

        y, _ = define_foreground(
            x,
            mode=config.foreground_mode,
            q=config.foreground_quantile,
            rng=np.random.default_rng(config.seed),
        )
        if y is None:
            y = np.zeros(len(x))

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
            title=f"C={gene_row['Coverage']:.3f}, S={gene_row['Spatial_Bias_Score']:.3f}",
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
        fig_single.savefig(examples_dir / f"{gene_name}_{archetype.replace(' ', '_')}.png", dpi=150)
        plt.close(fig_single)

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(
            outdir / "figures" / f"fig_archetype_examples.{ext}", dpi=300, bbox_inches="tight"
        )
    plt.close(fig)

    example_meta = pd.DataFrame(
        [
            {
                "Archetype": arch,
                "gene_id": gene_id,
                "gene_name": gene_name,
                **df[df["gene"] == gene_id].iloc[0].to_dict(),
            }
            for arch, (gene_id, gene_name) in selected.items()
        ]
    )
    example_meta.to_csv(examples_dir / "example_metadata.csv", index=False)


def generate_report(
    df: pd.DataFrame,
    c_cut: float,
    s_cut: float,
    thresholds_meta: Dict[str, Any],
    filter_stats: Dict[str, int],
    reliability: Dict[str, Any],
    manifest: Dict[str, Any],
    outdir: Path,
):
    """Generate markdown report."""

    archetype_counts = df["Archetype"].value_counts().to_dict()

    report = f"""# KPMP All-Gene Archetypes Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}


- **Cells analyzed:** {manifest.get("n_cells", "N/A"):,}
- **Embedding used:** `{manifest.get("embedding_key", "N/A")}`
- **Subset query:** {manifest.get("subset_query", "None (all cells)")}
- **Total genes in dataset:** {filter_stats.get("total_genes", "N/A"):,}
- **Genes passing filters:** {filter_stats.get("passed_both", "N/A"):,}


| Criterion | Threshold | Passed |
|-----------|-----------|--------|
| Coverage | ≥ {manifest.get("min_coverage", 0.005)} | {filter_stats.get("passed_coverage", "N/A"):,} |
| Nonzero cells | ≥ {manifest.get("min_nonzero", 50)} | {filter_stats.get("passed_nonzero", "N/A"):,} |
| **Both** | - | **{filter_stats.get("passed_both", "N/A"):,}** |


- **Value used:** {c_cut:.4f}
- **Method:** {thresholds_meta.get("c_cut_method", "default or user-specified")}

- **Value used:** {s_cut:.4f}
- **Method:** {thresholds_meta.get("s_cut_method", "default or null-derived")}


| Archetype | Count | Percentage |
|-----------|-------|------------|
"""

    for archetype in ARCHETYPE_NAMES.values():
        count = archetype_counts.get(archetype, 0)
        pct = 100 * count / len(df) if len(df) > 0 else 0
        report += f"| {archetype} | {count:,} | {pct:.1f}% |\n"

    report += f"\n**Total classified:** {len(df):,}\n"

    report += "\n## Top Genes per Archetype\n"

    for archetype in ARCHETYPE_NAMES.values():
        subset = df[df["Archetype"] == archetype]
        if len(subset) == 0:
            continue

        report += f"\n### {archetype} (n={len(subset):,})\n\n"
        report += "| Gene | Coverage (C) | Spatial (S) | Sign |\n"
        report += "|------|--------------|-------------|------|\n"

        if archetype in ["Focal Marker", "Regional Gradient"]:
            top = subset.nlargest(10, "Spatial_Bias_Score")
        else:
            top = subset.nlargest(10, "Coverage")

        for _, row in top.iterrows():
            report += f"| {row['gene']} | {row['Coverage']:.4f} | {row['Spatial_Bias_Score']:.4f} | {int(row['spatial_sign'])} |\n"

    report += "\n## Reliability Checks\n"

    if "subsample_stability" in reliability:
        ss = reliability["subsample_stability"]
        report += f"""
- **Spearman correlation:** {ss.get("spearman_r", "N/A"):.3f}
- **Median |ΔS|:** {ss.get("median_delta_s", "N/A"):.4f}
- **Genes tested:** {ss.get("n_genes_tested", "N/A")}
"""

    if reliability.get("cross_embedding"):
        ce = reliability["cross_embedding"]
        report += f"""
- **UMAP vs. t-SNE Spearman:** {ce.get("spearman_r", "N/A"):.3f}
- **Genes tested:** {ce.get("n_genes_tested", "N/A")}
"""

    report += """

- `runs_all_genes.csv`: Complete per-gene results
- `classification.csv`: Gene-to-archetype mapping
- `derived_thresholds.json`: Threshold derivation details
- `manifest.json`: Run provenance and configuration
- `figures/`: Publication-ready figures
- `examples/`: Per-archetype example gene plots


1. **fig_CS_scatter**: All genes in C-S space with quadrant boundaries
2. **fig_CS_marginals**: Histograms of C and S distributions
3. **fig_top_tables**: Top genes per archetype
4. **fig_archetype_examples**: Representative gene visualizations
"""

    with open(outdir / "report.md", "w") as f:
        f.write(report)

    logger.info(f"Report saved to {outdir / 'report.md'}")


def main():
    args = parse_args()

    if args.profile:
        logger.info("Profiling enabled")
        for func_name in dir():
            obj = globals().get(func_name)
            if hasattr(obj, "enable_profiling"):
                obj.enable_profiling(True)

    start_time = time.time()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "figures").mkdir(exist_ok=True)

    logger.info(f"Output directory: {outdir}")
    logger.info(f"Workers: {args.n_workers if args.n_workers != 0 else 'single-process'}")
    logger.info(f"I/O format: {'parquet' if args.use_parquet and HAS_PARQUET else 'csv'}")

    stage_start = time.time()
    logger.info(f"Loading AnnData from {args.h5ad}")
    adata = ad.read_h5ad(args.h5ad)
    logger.info(f"Loaded {adata.n_obs:,} cells × {adata.n_vars:,} genes")
    if args.profile:
        TIMINGS["data_loading"] = time.time() - stage_start

    from biorsp.preprocess.context import discover_embedding_key, prepare_context
    from biorsp.utils.config import BioRSPConfig

    try:
        import sys

        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from analysis.kidney_atlas.utils.standardized_plotting import (
            generate_kidney_panels,
            save_kidney_manifest,
        )

        HAS_STANDARDIZED_PLOTTING = True
    except ImportError:
        HAS_STANDARDIZED_PLOTTING = False
        logger.warning("Standardized plotting not available")

    embedding_key = args.embedding_key
    if embedding_key is None:
        embedding_key = discover_embedding_key(adata)
    logger.info(f"Using embedding: {embedding_key}")

    config = BioRSPConfig(
        B=args.B,
        delta_deg=args.delta_deg,
        foreground_quantile=args.foreground_quantile,
        empty_fg_policy=args.empty_fg_policy,
        expr_threshold_mode=args.expr_threshold_mode,
        expr_threshold_value=args.expr_threshold_value,
        seed=args.seed,
        n_permutations=0,  # Stage A: no permutations
    )

    stage_start = time.time()
    logger.info("Preparing geometric context...")
    adata_sub, context = prepare_context(
        adata,
        embedding_key=embedding_key,
        subset=args.subset,
        config=config,
        max_cells=args.max_cells,
        seed=args.seed,
    )
    logger.info(f"Context prepared: {context.n_cells:,} cells")
    if args.profile:
        TIMINGS["context_preparation"] = time.time() - stage_start

    temp_h5ad = None
    if args.n_workers != 0 and args.n_workers != 1:
        temp_h5ad = outdir / "temp_subset.h5ad"
        if not temp_h5ad.exists():
            logger.info(f"Saving subsetted data for workers: {temp_h5ad}")
            adata_sub.write_h5ad(temp_h5ad)

    expr_threshold, expr_mode = detect_expression_threshold(adata_sub)
    if args.expr_threshold_mode == "fixed" and args.expr_threshold_value is not None:
        expr_threshold = args.expr_threshold_value
        expr_mode = "fixed"
    logger.info(f"Expression threshold: {expr_threshold} ({expr_mode})")

    stage_start = time.time()
    filtered_genes, filter_stats = fast_filter_genes(
        adata_sub,
        min_coverage=args.min_coverage,
        min_nonzero=args.min_nonzero,
        threshold=expr_threshold,
    )
    if args.profile:
        TIMINGS["gene_filtering"] = time.time() - stage_start

    if len(filtered_genes) == 0:
        logger.error("No genes passed filtering criteria!")
        sys.exit(1)

    logger.info("Creating gene name mapping...")
    gene_name_mapping = create_gene_name_mapping(adata)

    logger.info(f"Scoring {len(filtered_genes):,} genes...")
    df = score_all_genes_parallel(
        adata_sub,
        filtered_genes,
        context,
        config,
        outdir,
        chunk_size=args.chunk_size,
        resume=args.resume,
        n_workers=args.n_workers,
        use_parquet=args.use_parquet and HAS_PARQUET,
        h5ad_path=str(temp_h5ad) if temp_h5ad else str(args.h5ad),
    )

    df = add_gene_names_to_dataframe(df, gene_name_mapping)

    thresholds_meta = {}

    if args.c_cut is not None:
        c_cut = args.c_cut
        thresholds_meta["c_cut_method"] = "user_specified"
    else:
        c_cut = derive_c_cut(df, default=0.10)
        thresholds_meta["c_cut_method"] = "auto_derived"
    thresholds_meta["c_cut"] = c_cut

    if args.s_cut is not None:
        s_cut = args.s_cut
        thresholds_meta["s_cut_method"] = "user_specified"
    elif args.derive_thresholds:
        s_cut = derive_s_cut_simple(df)
        thresholds_meta["s_cut_method"] = "distribution_derived"
    else:
        s_cut = 0.05
        thresholds_meta["s_cut_method"] = "default"
    thresholds_meta["s_cut"] = s_cut

    logger.info(f"Thresholds: c_cut={c_cut:.4f}, s_cut={s_cut:.4f}")

    df = classify_genes(df, c_cut, s_cut)

    if args.compute_pvalues:
        config_pval_dict = asdict(config)
        config_pval_dict["n_permutations"] = args.n_permutations
        config_pval = BioRSPConfig(**config_pval_dict)
        df = add_pvalues_to_top_genes(
            adata_sub,
            df,
            context,
            config_pval,
            topk=args.pvalue_topk,
            n_permutations=args.n_permutations,
        )

    reliability = {}
    if not args.skip_reliability:
        stage_start = time.time()
        logger.info("Running reliability checks...")

        top_s_genes = df.nlargest(500, "Spatial_Bias_Score")["gene"].tolist()

        reliability["subsample_stability"] = check_subsample_stability(
            adata_sub,
            top_s_genes,
            embedding_key,
            config,
            seed=args.seed,
        )

        ce_result = check_cross_embedding(adata_sub, top_s_genes, config)
        if ce_result:
            reliability["cross_embedding"] = ce_result

        if args.profile:
            TIMINGS["reliability_checks"] = time.time() - stage_start

    stage_start = time.time()
    logger.info("Saving outputs...")

    df.to_csv(outdir / "runs_all_genes.csv", index=False)

    class_cols = ["gene"]
    if "gene_name" in df.columns:
        class_cols.append("gene_name")
    class_cols.extend(["Archetype", "Coverage", "Spatial_Bias_Score", "c_cut_used", "s_cut_used"])
    class_df = df[class_cols]
    class_df.to_csv(outdir / "classification.csv", index=False)

    with open(outdir / "derived_thresholds.json", "w") as f:
        json.dump(thresholds_meta, f, indent=2)

    manifest = {
        "h5ad_path": str(args.h5ad),
        "embedding_key": embedding_key,
        "subset_query": args.subset,
        "n_cells": context.n_cells,
        "n_genes_scored": len(df),
        "seed": args.seed,
        "max_cells": args.max_cells,
        "min_coverage": args.min_coverage,
        "min_nonzero": args.min_nonzero,
        "B": args.B,
        "delta_deg": args.delta_deg,
        "foreground_quantile": args.foreground_quantile,
        "empty_fg_policy": args.empty_fg_policy,
        "n_workers": args.n_workers,
        "use_parquet": args.use_parquet and HAS_PARQUET,
        "timestamp": datetime.now().isoformat(),
        "filter_stats": filter_stats,
    }
    with open(outdir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    if args.profile:
        TIMINGS["output_saving"] = time.time() - stage_start

    stage_start = time.time()
    logger.info("Generating figures...")
    plot_cs_scatter(df, c_cut, s_cut, outdir / "figures")
    plot_cs_marginals(df, c_cut, s_cut, outdir / "figures")
    plot_top_tables(df, outdir / "figures")
    plot_archetype_examples(adata_sub, df, context, config, outdir, embedding_key)

    # Generate standardized plotting outputs
    if HAS_STANDARDIZED_PLOTTING:
        logger.info("Generating standardized figures...")
        try:
            generate_kidney_panels(
                df,
                outdir,
                c_cut=c_cut,
                s_cut=s_cut,
            )

            save_kidney_manifest(
                outdir,
                params=vars(args),
                n_genes=len(df),
                n_cells=context.n_cells,
                c_cut=c_cut,
                s_cut=s_cut,
                runtime_seconds=time.time() - start_time,
            )
            logger.info("Standardized figures generated")
        except Exception as e:
            logger.warning(f"Could not generate standardized figures: {e}")

    if args.profile:
        TIMINGS["plotting"] = time.time() - stage_start

    generate_report(df, c_cut, s_cut, thresholds_meta, filter_stats, reliability, manifest, outdir)

    if temp_h5ad and temp_h5ad.exists():
        temp_h5ad.unlink()

    total_time = time.time() - start_time

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"Output directory: {outdir.resolve()}")
    logger.info(f"Genes scored: {len(df):,}")
    logger.info(f"Total time: {total_time:.1f}s ({total_time / 60:.1f} min)")
    logger.info("Archetype counts:")
    for arch, count in df["Archetype"].value_counts().items():
        logger.info(f"  {arch}: {count:,}")

    if args.profile:
        logger.info("\n" + "=" * 60)
        logger.info("PROFILING SUMMARY")
        logger.info("=" * 60)
        for stage, elapsed in sorted(TIMINGS.items(), key=lambda x: -x[1]):
            pct = 100 * elapsed / total_time
            logger.info(f"  {stage:.<40} {elapsed:>8.2f}s ({pct:>5.1f}%)")
        logger.info("=" * 60)

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
