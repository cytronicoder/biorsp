"""Compare disease-stratified BioRSP results and generate publication-ready outputs.

Ranking definitions (used by --rank_by):
- a2: angular bias proxy (max |delta metric| vs reference across diseases)
- a1: coverage/dispersion proxy (max |delta coverage_expr| vs reference)
- rmsd: deviation of the chosen metric across diseases (std)
- effect: median delta magnitude vs reference
- fdr: smallest BH-adjusted q-value across diseases
- composite: robust blend favoring higher a2, higher rmsd, larger effect, lower FDR
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import norm

from analysis.kidney_atlas.utils.plot_utils_validated import plot_gene_exemplar
from biorsp.api import BioRSPConfig
from biorsp.plotting.standard import make_standard_plot_set

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RANK_MODES = {"a2", "a1", "rmsd", "fdr", "effect", "composite"}


@dataclass
class RunContext:
    """Container for run-wide context and configuration."""

    diseases: list[str]
    reference: str
    metric: str
    seed: int
    top_k: int
    min_expr_cells: int
    include_abstain: bool
    rank_by: str
    embedding_key: str
    h5ad_path: Path | None
    outdir: Path
    group_key: str | None
    groups: list[str] | None
    global_only: bool
    per_group_only: bool


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for disease comparison and plotting."""

    parser = argparse.ArgumentParser(description=__doc__ or "")
    parser.add_argument("results_dir", type=str, help="Base directory with disease-stratified runs")
    parser.add_argument("--h5ad", type=str, default=None, help="Path to AnnData for gene plots")
    parser.add_argument(
        "--embedding-key",
        type=str,
        default="X_umap",
        help="Embedding key in AnnData to use for overlays (default: X_umap)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="spatial_score",
        help="Metric to compare; aliases accepted: spatial_score, Spatial_Bias_Score, coverage_expr, coverage_score, p_value, q_value",
    )
    parser.add_argument(
        "--rank_by",
        type=str,
        default="composite",
        choices=sorted(RANK_MODES),
        help="Ranking mode (see module docstring for definitions)",
    )
    parser.add_argument("--top_k", type=int, default=25, help="Top K genes to plot and report")
    parser.add_argument(
        "--min_expr_cells",
        type=int,
        default=25,
        help="Minimum cells expressing a gene in any condition to keep it",
    )
    parser.add_argument(
        "--include_abstain",
        action="store_true",
        help="Include abstained genes in ranking; otherwise they are reported but excluded from ranks",
    )
    parser.add_argument(
        "--group_key",
        type=str,
        default=None,
        help="Optional obs column to stratify plots by subgroup (e.g., subtype)",
    )
    parser.add_argument(
        "--groups",
        type=str,
        default=None,
        help="Comma-separated allowlist of groups to include (requires --group_key)",
    )
    parser.add_argument(
        "--global_only",
        action="store_true",
        help="Only run global (non-stratified) analysis even if group_key is provided",
    )
    parser.add_argument(
        "--per_group_only",
        action="store_true",
        help="Only run per-group analysis; skip global aggregation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling and plotting",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Optional output directory (defaults to <results_dir>/comparison_outputs)",
    )
    return parser.parse_args()


def normalize_metric_name(metric: str) -> str:
    """Normalize user metric aliases to expected column names."""

    metric_alias_map = {
        "spatial_bias_score": "spatial_score",
        "spatial_score": "spatial_score",
        "coverage_expr": "coverage_expr",
        "coverage_score": "coverage_expr",
        "p_value": "p_value",
        "q_value": "q_value",
    }
    key = metric.lower()
    return metric_alias_map.get(key, key)


def pick_reference(diseases: Iterable[str]) -> str:
    """Choose reference disease: prefer healthy/normal/control else first."""

    for disease in diseases:
        name = disease.lower()
        if "healthy" in name or "normal" in name or "control" in name:
            return disease
    return list(diseases)[0]


def load_disease_results(results_dir: Path) -> dict[str, pd.DataFrame]:
    """Load gene_scores per disease directory."""

    disease_data: dict[str, pd.DataFrame] = {}
    for disease_folder in sorted(results_dir.iterdir()):
        if not disease_folder.is_dir():
            continue
        scores_file = disease_folder / "gene_scores.csv"
        if not scores_file.exists():
            logger.warning(f"No gene_scores.csv found in {disease_folder}")
            continue
        df = pd.read_csv(scores_file)
        disease_data[disease_folder.name] = df
        logger.info(f"Loaded {len(df)} genes for {disease_folder.name}")

    if not disease_data:
        raise ValueError(f"No disease results found in {results_dir}")
    return disease_data


def merge_disease_tables(
    disease_data: Mapping[str, pd.DataFrame],
    metric: str,
) -> pd.DataFrame:
    """Merge per-disease tables, keeping key columns for ranking and QC."""

    diseases = sorted(disease_data.keys())
    metric_col = normalize_metric_name(metric)

    base_cols = [
        "gene_symbol",
        "coverage_expr",
        "spatial_score",
        "p_value",
        "q_value",
        "n_cells_total",
        "warnings",
        "Archetype",
        "expr_threshold_value",
    ]
    merged = None

    for disease in diseases:
        df = disease_data[disease].copy()
        for col in base_cols:
            if col not in df.columns:
                df[col] = np.nan
        df["n_cells_expr"] = (df["coverage_expr"] * df["n_cells_total"]).fillna(0)
        suffix_map = {col: f"{col}_{disease}" for col in base_cols + ["n_cells_expr"]}
        df = df[["gene"] + base_cols + ["n_cells_expr"]]
        df = df.rename(columns=suffix_map)
        merged = df if merged is None else merged.merge(df, on=["gene"], how="outer")

    if merged is None:
        raise ValueError("Failed to merge disease tables")

    metric_cols = [c for c in merged.columns if c.startswith(metric_col + "_")]
    if not metric_cols:
        raise KeyError(f"Metric '{metric_col}' not found in merged columns")

    symbol_cols = [c for c in merged.columns if c.startswith("gene_symbol_")]
    merged["gene_symbol"] = merged[symbol_cols].bfill(axis=1).iloc[:, 0]
    return merged


def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR for 1D array; NaNs preserved."""

    p = np.asarray(pvals, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    mask = ~np.isnan(p)
    m = mask.sum()
    if m == 0:
        return q
    order = np.argsort(p[mask])
    ranked = p[mask][order]
    ranks = np.arange(1, m + 1)
    q_vals = ranked * m / ranks
    q_vals = np.minimum.accumulate(q_vals[::-1])[::-1]
    q_vals[q_vals > 1] = 1
    q[mask] = q_vals[np.argsort(order)]
    return q


def _robust_z(x: pd.Series) -> pd.Series:
    """Rank-based z-score with clipping for robustness."""

    ranks = x.rank(method="average", pct=True)
    z = pd.Series(norm.ppf(ranks)).replace([np.inf, -np.inf], 0)
    return z.clip(-4, 4)


def compute_gene_stats(
    merged: pd.DataFrame,
    diseases: list[str],
    metric: str,
    reference: str,
    min_expr_cells: int,
    include_abstain: bool,
) -> pd.DataFrame:
    """Compute effect sizes, ranking features, and QC columns."""

    metric_col = normalize_metric_name(metric)
    metric_cols = [f"{metric_col}_{d}" for d in diseases]
    coverage_cols = [f"coverage_expr_{d}" for d in diseases]
    p_cols = [f"p_value_{d}" for d in diseases]
    q_cols = [f"q_value_{d}" for d in diseases]
    n_expr_cols = [f"n_cells_expr_{d}" for d in diseases]
    warning_cols = [f"warnings_{d}" for d in diseases]

    ref_col = f"{metric_col}_{reference}"
    merged["reference_metric"] = merged[ref_col]

    deltas = []
    for disease in diseases:
        if disease == reference:
            continue
        col = f"{metric_col}_{disease}"
        merged[f"delta_{disease}"] = merged[col] - merged[ref_col]
        deltas.append(f"delta_{disease}")

    merged["effect_size"] = merged[deltas].apply(lambda row: np.nanmedian(row.values.astype(float)), axis=1)
    merged["effect_size_abs"] = merged["effect_size"].abs()
    merged["a2_score"] = merged[deltas].abs().max(axis=1)
    merged["a1_score"] = merged[[c for c in coverage_cols if c in merged]].sub(
        merged[f"coverage_expr_{reference}"], axis=0
    ).abs().max(axis=1)
    merged["rmsd_score"] = merged[metric_cols].std(axis=1, ddof=0)

    merged["p_min"] = merged[p_cols].min(axis=1, skipna=True)
    q_vals = merged[q_cols].min(axis=1, skipna=True)
    merged["q_min"] = q_vals
    need_q = q_vals.isna()
    if need_q.any():
        merged.loc[need_q, "q_min"] = _bh_fdr(merged.loc[need_q, "p_min"].values)

    merged["abstain_reason"] = merged[warning_cols].bfill(axis=1).iloc[:, 0]
    merged["abstain"] = merged["abstain_reason"].notna() & (merged["abstain_reason"].astype(str) != "nan")

    merged["n_cells_total"] = merged[f"n_cells_total_{reference}"]
    merged["n_cells_expr"] = merged[n_expr_cols].max(axis=1, skipna=True)
    merged["frac_expr"] = merged["n_cells_expr"] / merged["n_cells_total"].replace(0, np.nan)

    filtered = merged[merged["n_cells_expr"].fillna(0) >= min_expr_cells].copy()
    if not include_abstain:
        filtered = filtered[~filtered["abstain"]]

    filtered["gene_symbol"] = filtered["gene_symbol"].fillna(filtered["gene"])
    return filtered


def rank_genes(df: pd.DataFrame, rank_by: str) -> pd.DataFrame:
    """Rank genes according to the chosen metric."""

    rank_by = rank_by.lower()
    if rank_by not in RANK_MODES:
        raise ValueError(f"rank_by must be one of {sorted(RANK_MODES)}")

    df = df.copy()
    if rank_by == "a2":
        df["score"] = df["a2_score"]
        df = df.sort_values("score", ascending=False)
    elif rank_by == "a1":
        df["score"] = df["a1_score"]
        df = df.sort_values("score", ascending=False)
    elif rank_by == "rmsd":
        df["score"] = df["rmsd_score"]
        df = df.sort_values("score", ascending=False)
    elif rank_by == "effect":
        df["score"] = df["effect_size_abs"]
        df = df.sort_values("score", ascending=False)
    elif rank_by == "fdr":
        df["score"] = df["q_min"]
        df = df.sort_values("score", ascending=True)
    else:
        z_a2 = _robust_z(df["a2_score"].fillna(0))
        z_rmsd = _robust_z(df["rmsd_score"].fillna(0))
        z_effect = _robust_z(df["effect_size_abs"].fillna(0))
        z_fdr = -_robust_z((-np.log10(df["q_min"].replace(0, np.nan))).fillna(0))
        df["score"] = z_a2 + z_rmsd + z_effect + z_fdr
        df = df.sort_values("score", ascending=False)

    df["rank"] = np.arange(1, len(df) + 1)
    return df


def _save_metadata(outdir: Path, args: argparse.Namespace, ctx: RunContext) -> None:
    """Persist run metadata to JSON."""

    git_commit = None
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        git_commit = None

    meta = {
        "args": vars(args),
        "reference": ctx.reference,
        "diseases": ctx.diseases,
        "seed": ctx.seed,
        "git_commit": git_commit,
    }
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "run_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)


def plot_rank_scatter(df: pd.DataFrame, outdir: Path) -> None:
    """Scatter of A2 vs RMSD colored by -log10(FDR)."""

    outdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(
        df["a2_score"],
        df["rmsd_score"],
        c=-np.log10(df["q_min"].replace(0, np.nan)),
        cmap="viridis",
        alpha=0.8,
        s=25,
    )
    ax.set_xlabel("A2 (|delta metric| max)")
    ax.set_ylabel("RMSD (metric across diseases)")
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("-log10(FDR)")
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(outdir / f"rank_scatter.{ext}", dpi=300 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


def plot_qc(df: pd.DataFrame, outdir: Path) -> None:
    """QC overview plots."""

    outdir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    sns.histplot(df["frac_expr"].dropna(), ax=axes[0], bins=20, color="#4C72B0")
    axes[0].set_title("Expression fraction")
    sns.histplot(df["q_min"].replace([np.inf, -np.inf], np.nan).dropna(), ax=axes[1], bins=20, color="#55A868")
    axes[1].set_title("q-value distribution")
    abstain_counts = df["abstain"].value_counts(dropna=False)
    axes[2].bar(
        ["Included", "Abstain"],
        [abstain_counts.get(False, 0), abstain_counts.get(True, 0)],
        color="#C44E52",
    )
    axes[2].set_title("Abstention")
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(outdir / f"qc_overview.{ext}", dpi=300 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


def make_standard_plots(scores_df: pd.DataFrame, outdir: Path) -> None:
    """Delegate to standardized plot set for context."""

    try:
        thresholds = {}
        if "C_cut" in scores_df.columns:
            thresholds["C_cut"] = scores_df["C_cut"].iloc[0]
        if "S_cut" in scores_df.columns:
            thresholds["S_cut"] = scores_df["S_cut"].iloc[0]
        make_standard_plot_set(
            scores_df=scores_df,
            outdir=outdir,
            thresholds=thresholds if thresholds else None,
            truth_col=None,
            debug=False,
        )
    except Exception as exc:  # pragma: no cover - defensive plotting guard
        logger.warning(f"Standard plot generation skipped: {exc}")


def generate_gene_panels(
    adata: anndata.AnnData,
    top_genes: pd.DataFrame,
    embedding_key: str,
    outdir: Path,
    seed: int,
) -> None:
    """Generate per-gene RSP exemplar plots and a combined PDF panel."""

    if adata is None:
        logger.warning("No AnnData provided; skipping gene exemplar plots")
        return

    config = BioRSPConfig(seed=seed)
    var_to_symbol = (
        adata.var.get("feature_name")
        if "feature_name" in adata.var
        else pd.Series(adata.var_names, index=adata.var_names)
    ).to_dict()

    rsp_dir = outdir / "figures" / "rsp"
    rsp_dir.mkdir(parents=True, exist_ok=True)

    exemplar_paths = []
    for _, row in top_genes.iterrows():
        gene = row["gene"]
        if gene not in adata.var_names:
            continue
        coverage_thr = row.get("expr_threshold_value", np.nan)
        if np.isnan(coverage_thr):
            gene_values = adata[:, gene].X
            gene_values = gene_values.toarray() if hasattr(gene_values, "toarray") else gene_values
            coverage_thr = np.quantile(gene_values, 0.9)
        plot_gene_exemplar(
            adata=adata,
            gene=gene,
            gene_row=row,
            embedding_key=embedding_key,
            config=config,
            outdir=rsp_dir,
            var_to_symbol=var_to_symbol,
            coverage_threshold=coverage_thr,
            strict=False,
        )
        safe_name = str(row.get("gene_symbol", gene)).replace("/", "_").replace(":", "_")
        exemplar_paths.append(rsp_dir / "exemplars" / f"exemplar_{safe_name}.png")

    if exemplar_paths:
        panel_pdf = rsp_dir / f"panel_top_{len(exemplar_paths)}.pdf"
        with PdfPages(panel_pdf) as pdf:
            for path in exemplar_paths:
                if not path.exists():
                    continue
                img = plt.imread(path)
                fig, ax = plt.subplots(figsize=(11, 3))
                ax.imshow(img)
                ax.axis("off")
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)


def write_ranked_table(df: pd.DataFrame, ctx: RunContext, outdir: Path, comparison_label: str) -> Path:
    """Write ranked_genes.csv with required columns."""

    cols = [
        "gene",
        "gene_symbol",
        "comparison",
        "n_cells_total",
        "n_cells_expr",
        "frac_expr",
        "a1_score",
        "a2_score",
        "rmsd_score",
        "effect_size",
        "effect_size_abs",
        "p_min",
        "q_min",
        "abstain",
        "abstain_reason",
        "rank",
        "score",
    ]
    present_cols = [c for c in cols if c in df.columns]
    df_out = df.copy()
    df_out["comparison"] = comparison_label
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "ranked_genes.csv"
    df_out[present_cols].to_csv(path, index=False)
    return path


def write_summary(
    df: pd.DataFrame,
    top_df: pd.DataFrame,
    ctx: RunContext,
    outdir: Path,
    comparison_label: str,
) -> None:
    """Write scientist-friendly markdown summary."""

    filtered_out = len(df) - len(top_df)
    lines = [
        "# Disease-Stratified BioRSP Comparison Report",
        "",
        f"**Comparison**: {comparison_label}",
        f"**Reference condition**: {ctx.reference}",
        f"**Metric**: {ctx.metric}",
        f"**Ranking**: {ctx.rank_by} (see definitions in file header)",
        f"**Top K**: {ctx.top_k}",
        f"**Min expr cells**: {ctx.min_expr_cells} (filtered {filtered_out} genes below threshold)",
        "",
        "## Top 10 genes",
        "",
    ]

    top10 = top_df.head(10)
    top_table = top10[["gene_symbol", "effect_size", "q_min", "a2_score", "rmsd_score"]].copy()
    lines.append(top_table.to_markdown(index=False))
    lines.extend(
        [
            "",
            "## Notes",
            "- Scores are embedding-based and do not imply physical tissue distances.",
            "- Multiple testing: q-values are BH-adjusted; non-finite p-values are masked.",
            "- Batch effects may alter embeddings; interpret gradients cautiously.",
        ]
    )

    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "summary.md", "w") as f:
        f.write("\n".join(lines))


def run_for_context(
    disease_data: Mapping[str, pd.DataFrame],
    adata: anndata.AnnData | None,
    args: argparse.Namespace,
    ctx: RunContext,
    label: str,
    outdir: Path,
) -> None:
    """Execute the full pipeline for one context (global or subgroup)."""

    merged = merge_disease_tables(disease_data, metric=ctx.metric)
    stats = compute_gene_stats(
        merged=merged,
        diseases=ctx.diseases,
        metric=ctx.metric,
        reference=ctx.reference,
        min_expr_cells=ctx.min_expr_cells,
        include_abstain=ctx.include_abstain,
    )
    ranked = rank_genes(stats, rank_by=ctx.rank_by)
    top_df = ranked.head(ctx.top_k)

    write_ranked_table(ranked, ctx, outdir, comparison_label=label)
    write_summary(ranked, top_df, ctx, outdir, comparison_label=label)
    plot_rank_scatter(ranked, outdir / "figures")
    plot_qc(ranked, outdir / "figures")

    if adata is not None:
        generate_gene_panels(adata, top_df, ctx.embedding_key, outdir, ctx.seed)

    ref_df = disease_data.get(ctx.reference)
    if ref_df is not None:
        make_standard_plots(ref_df, outdir / "figures" / "standard")

    _save_metadata(outdir, args, ctx)


def load_optional_adata(h5ad_path: str | None) -> anndata.AnnData | None:
    """Load AnnData if provided."""

    if h5ad_path is None:
        return None
    path = Path(h5ad_path)
    if not path.exists():
        raise FileNotFoundError(f"AnnData file not found: {path}")
    return anndata.read_h5ad(path)


def main() -> None:
    args = parse_args()

    results_dir = Path(args.results_dir)
    outdir = Path(args.outdir) if args.outdir else results_dir / "comparison_outputs"
    disease_data = load_disease_results(results_dir)
    diseases = sorted(disease_data.keys())
    reference = pick_reference(diseases)

    adata = load_optional_adata(args.h5ad)
    groups = None
    if args.groups:
        groups = [g.strip() for g in args.groups.split(",") if g.strip()]

    ctx = RunContext(
        diseases=diseases,
        reference=reference,
        metric=normalize_metric_name(args.metric),
        seed=args.seed,
        top_k=args.top_k,
        min_expr_cells=args.min_expr_cells,
        include_abstain=args.include_abstain,
        rank_by=args.rank_by,
        embedding_key=args.embedding_key,
        h5ad_path=Path(args.h5ad) if args.h5ad else None,
        outdir=outdir,
        group_key=args.group_key,
        groups=groups,
        global_only=args.global_only,
        per_group_only=args.per_group_only,
    )

    if not ctx.per_group_only:
        run_for_context(
            disease_data=disease_data,
            adata=adata,
            args=args,
            ctx=ctx,
            label="global",
            outdir=outdir / "global",
        )

    if args.group_key and not ctx.global_only and adata is not None:
        group_values = adata.obs[args.group_key].unique().tolist()
        if ctx.groups:
            group_values = [g for g in group_values if g in ctx.groups]
        for g in group_values:
            subset = adata[adata.obs[args.group_key] == g].copy()
            if subset.n_obs == 0:
                continue
            group_outdir = outdir / "by_group" / str(g)
            run_for_context(
                disease_data=disease_data,
                adata=subset,
                args=args,
                ctx=ctx,
                label=str(g),
                outdir=group_outdir,
            )


if __name__ == "__main__":
    main()
