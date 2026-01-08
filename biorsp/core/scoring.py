"""Implementation of BioRSP scoring logic."""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

from biorsp.core.engine import compute_rsp_radar
from biorsp.preprocess.foreground import define_foreground
from biorsp.preprocess.geometry import compute_vantage, get_sector_indices, polar_coordinates
from biorsp.preprocess.normalization import normalize_radii
from biorsp.utils.config import BioRSPConfig

logger = logging.getLogger(__name__)


def _get_expression(adata: AnnData, gene: str, use_raw: bool = False) -> np.ndarray:
    """Extract expression vector for a gene."""
    source = adata
    if use_raw and adata.raw is not None:
        source = adata.raw

    try:
        idx = source.var_names.get_loc(gene)
        if isinstance(source.X, sparse.spmatrix):
            return source.X[:, idx].toarray().flatten()
        return source.X[:, idx]
    except KeyError as err:
        raise ValueError(f"Gene {gene} not found in adata{' (raw)' if use_raw else ''}.") from err


def _detect_threshold(x: np.ndarray, config: BioRSPConfig) -> Tuple[float, str]:
    """Determine expression threshold $t_g$."""
    mode = config.expr_threshold_mode
    val = config.expr_threshold_value
    nonzero_q = config.nonzero_quantile

    if mode == "fixed":
        if val is not None:
            return float(val), "fixed"

        is_integers = np.allclose(x, np.round(x))
        return (1.0 if is_integers else 0.1), "fixed_inferred"

    if mode == "nonzero_quantile":
        nonzero = x[x > 0]
        if len(nonzero) == 0:
            return 0.0, "nonzero_quantile_empty"
        return float(np.percentile(nonzero, nonzero_q * 100)), "nonzero_quantile"

    is_integers = (x.dtype.kind in "iu") or (np.allclose(x, np.round(x)) and np.max(x) > 1.0)
    if is_integers:
        return 1.0, "detect_count"
    else:
        return 1e-6, "detect_continuous"


def _prepare_embedding(
    adata: AnnData,
    embedding_key: str,
    subset: Optional[Union[dict, str, List[str]]],
    config: BioRSPConfig,
) -> Tuple[AnnData, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[np.ndarray], Dict]:
    """Shared preprocessing for score_genes and score_gene_pairs."""
    if subset is not None:
        if isinstance(subset, dict):
            for k, v in subset.items():
                adata = adata[adata.obs[k] == v].copy()
        elif isinstance(subset, str):
            adata = adata[adata.obs.eval(subset)].copy()
        else:
            adata = adata[subset].copy()

    if adata.n_obs == 0:
        raise ValueError("Subset resulted in 0 cells.")

    coords = adata.obsm[embedding_key]
    if coords.shape[1] != 2:
        raise ValueError("Embedding must be 2D.")

    center = compute_vantage(
        coords,
        method=config.vantage,
        knn_k=config.center_knn_k,
        density_percentile=config.center_density_percentile,
        tol=config.geom_median_tol,
        max_iter=config.geom_median_max_iter,
        seed=config.seed,
    )

    r, theta = polar_coordinates(coords, center)
    r_norm, norm_stats = normalize_radii(r)
    sector_indices = get_sector_indices(theta, config.B, config.delta_deg)

    meta = {"n_cells": adata.n_obs, "center": center, "norm_stats": norm_stats}
    return adata, coords, center, r_norm, theta, sector_indices, meta


def _compute_spatial_score_from_radar(radar) -> Tuple[float, float]:
    """Compute S_g and r_mean from a RadarResult using bg_supported mask."""
    mask = radar.bg_supported_mask
    if mask is None or not np.any(mask):
        return 0.0, 0.0

    valid_rsp = np.nan_to_num(radar.rsp[mask], nan=0.0)
    w = radar.sector_weights[mask]

    sum_w = np.sum(w)
    if sum_w <= 0:
        return 0.0, 0.0

    s_g = float(np.sqrt(np.sum(w * valid_rsp**2) / sum_w))
    r_mean = float(np.sum(w * valid_rsp) / sum_w)
    return s_g, r_mean


def _permute_p_value(
    r_norm: np.ndarray,
    theta: np.ndarray,
    y_observed: np.ndarray,
    sector_indices: List[np.ndarray],
    config: BioRSPConfig,
    observed_s: float,
    stratify_labels: Optional[np.ndarray] = None,
    rng_seed: int = 42,
) -> Tuple[float, float, float, float, int]:
    """Stratified permutation inference for S_g."""
    n_perm = config.n_permutations
    if n_perm <= 0:
        return np.nan, np.nan, np.nan, np.nan, 0

    rng = np.random.default_rng(rng_seed)
    n_cells = len(y_observed)

    if stratify_labels is not None:
        unique_labels = np.unique(stratify_labels)
        strata_indices = [np.where(stratify_labels == label)[0] for label in unique_labels]
    else:
        strata_indices = [np.arange(n_cells)]

    sector_sort_indices = []
    for idx_s in sector_indices:
        if idx_s.size > 0:
            sector_sort_indices.append(np.argsort(r_norm[idx_s]))
        else:
            sector_sort_indices.append(None)

    null_scores = []
    for _ in range(n_perm):
        y_perm = np.copy(y_observed)
        for idxs in strata_indices:
            perm_idxs = rng.permutation(idxs)
            y_perm[idxs] = y_perm[perm_idxs]

        radar_null = compute_rsp_radar(
            r_norm,
            theta,
            y_perm,
            config=config,
            sector_indices=sector_indices,
            sector_sort_indices=sector_sort_indices,
        )
        s_null, _ = _compute_spatial_score_from_radar(radar_null)
        null_scores.append(s_null)

    null_scores = np.array(null_scores)
    p_val = (1 + np.sum(null_scores >= observed_s)) / (1 + n_perm)
    null_mean = np.mean(null_scores)
    null_sd = np.std(null_scores) + 1e-9
    z_score = (observed_s - null_mean) / null_sd

    return float(p_val), float(null_mean), float(null_sd), float(z_score), n_perm


def score_genes_impl(
    adata: AnnData,
    genes: List[str],
    embedding_key: str,
    subset: Optional[Union[dict, str, List[str]]],
    config: BioRSPConfig,
) -> pd.DataFrame:

    adata_sub, _, _, r_norm, theta, sector_indices, _ = _prepare_embedding(
        adata, embedding_key, subset, config
    )

    stratify_labels = None
    if config.stratify_key is not None and config.stratify_key in adata_sub.obs:
        vals = adata_sub.obs[config.stratify_key].values
        if np.issubdtype(vals.dtype, np.number):
            stratify_labels = pd.qcut(vals, config.n_strata, labels=False, duplicates="drop")
        else:
            stratify_labels = vals

    results = []

    show_progress = len(genes) > 1 and config.n_permutations > 0
    for gene in tqdm(genes, desc="Scoring genes", disable=not show_progress):
        try:
            x = _get_expression(adata_sub, gene, use_raw=config.coverage_use_raw)
        except ValueError as e:
            logger.warning(str(e))
            continue

        t_g, t_mode = _detect_threshold(x, config)
        coverage_expr = float(np.mean(x >= t_g))

        y, fg_info = define_foreground(
            x,
            mode=config.foreground_mode,
            q=config.foreground_quantile,
            abs_threshold=config.foreground_threshold,
            min_fg=config.min_fg_total,
            rng=np.random.default_rng(config.seed),
        )

        warnings = []
        if coverage_expr < 0.01:
            warnings.append("low_coverage_expr")

        if y is None:
            results.append(
                {
                    "gene": gene,
                    "n_cells_total": len(x),
                    "expr_threshold_mode": t_mode,
                    "expr_threshold_value": t_g,
                    "coverage_expr": coverage_expr,
                    "spatial_score": 0.0,
                    "spatial_sign": 0,
                    "r_mean_bg": 0.0,
                    "coverage_bg": 0.0,
                    "coverage_fg": 0.0,
                    "p_value": np.nan,
                    "z_score": np.nan,
                    "warnings": ";".join(warnings + ["insufficient_internal_fg"]),
                }
            )
            continue

        radar = compute_rsp_radar(r_norm, theta, y, config=config, sector_indices=sector_indices)
        s_g, r_mean = _compute_spatial_score_from_radar(radar)

        bg_mask = radar.bg_supported_mask
        cov_bg = float(np.mean(bg_mask)) if bg_mask is not None else 0.0
        if cov_bg < 0.8:
            warnings.append("low_coverage_bg")

        if bg_mask is not None and np.any(bg_mask):
            n_fg_sector = radar.n_fg_per_sector[bg_mask]
            cov_fg = float(np.mean(n_fg_sector >= config.min_fg_sector))
        else:
            cov_fg = 0.0

        gene_seed = (config.seed + hash(gene)) % (2**32)
        p_val, _, _, z_score, _ = _permute_p_value(
            r_norm,
            theta,
            y,
            sector_indices,
            config,
            s_g,
            stratify_labels=stratify_labels,
            rng_seed=gene_seed,
        )

        results.append(
            {
                "gene": gene,
                "n_cells_total": len(x),
                "expr_threshold_mode": t_mode,
                "expr_threshold_value": t_g,
                "coverage_expr": coverage_expr,
                "spatial_score": s_g,
                "spatial_sign": int(np.sign(r_mean)),
                "r_mean_bg": r_mean,
                "coverage_bg": cov_bg,
                "coverage_fg": cov_fg,
                "p_value": p_val,
                "z_score": z_score,
                "warnings": ";".join(warnings),
            }
        )

    df = pd.DataFrame(results)
    if "p_value" in df.columns and not df["p_value"].isna().all():
        valid_idx = df["p_value"].notna()
        _, qvals, _, _ = multipletests(df.loc[valid_idx, "p_value"], method="fdr_bh")
        df.loc[valid_idx, "q_value"] = qvals
    else:
        df["q_value"] = np.nan

    return df


def score_gene_pairs_impl(
    adata: AnnData,
    genes: List[str],
    embedding_key: str,
    subset: Optional[Union[dict, str, List[str]]],
    config: BioRSPConfig,
) -> pd.DataFrame:

    adata_sub, _, _, r_norm, theta, sector_indices, _ = _prepare_embedding(
        adata, embedding_key, subset, config
    )

    gene_data = {}
    for gene in tqdm(genes, desc="Computing profiles"):
        try:
            x = _get_expression(adata_sub, gene, use_raw=config.coverage_use_raw)
            y, _ = define_foreground(
                x,
                mode=config.foreground_mode,
                q=config.foreground_quantile,
                min_fg=config.min_fg_total,
            )
            if y is None:
                continue

            radar = compute_rsp_radar(
                r_norm, theta, y, config=config, sector_indices=sector_indices
            )
            s_g, r_mean = _compute_spatial_score_from_radar(radar)

            gene_data[gene] = {
                "rsp": radar.rsp,
                "mask": radar.bg_supported_mask,
                "r_mean": r_mean,
                "weights": radar.sector_weights,
            }
        except Exception as e:
            logger.warning(f"Error profiling {gene}: {e}")

    pairs = []
    valid_genes = list(gene_data.keys())
    n = len(valid_genes)
    for i in range(n):
        for j in range(i + 1, n):
            gA, gB = valid_genes[i], valid_genes[j]
            dA, dB = gene_data[gA], gene_data[gB]

            shared_mask = dA["mask"] & dB["mask"]
            shared_frac = float(np.mean(shared_mask))

            if shared_frac < config.min_shared_mask_fraction:
                pairs.append(
                    {
                        "gene_a": gA,
                        "gene_b": gB,
                        "similarity_profile": np.nan,
                        "similarity_sign": 0,
                        "copattern_score": 0.0,
                        "shared_mask_fraction": shared_frac,
                        "warnings": "low_shared_mask",
                    }
                )
                continue

            profA = np.nan_to_num(dA["rsp"][shared_mask], nan=0.0)
            profB = np.nan_to_num(dB["rsp"][shared_mask], nan=0.0)

            if np.std(profA) < 1e-9 or np.std(profB) < 1e-9:
                corr = 0.0
            else:
                corr = float(np.corrcoef(profA, profB)[0, 1])

            sim_sign = 1 if np.sign(dA["r_mean"]) == np.sign(dB["r_mean"]) else -1
            copattern = corr * sim_sign * np.sqrt(shared_frac)

            pairs.append(
                {
                    "gene_a": gA,
                    "gene_b": gB,
                    "similarity_profile": corr,
                    "similarity_sign": sim_sign,
                    "copattern_score": copattern,
                    "shared_mask_fraction": shared_frac,
                    "warnings": "",
                }
            )

    return pd.DataFrame(pairs)


def classify_genes_impl(
    df: pd.DataFrame, c_cut: Optional[float], s_cut: Optional[float], fdr_cut: float
) -> pd.DataFrame:
    df = df.copy()

    if c_cut is None:
        c_cut = 0.10

    method = "manual"
    if s_cut is None:
        if "q_value" in df.columns and not df["q_value"].dropna().empty:
            method = "fdr"

            s_cut = BioRSPConfig().effect_floor
        else:

            method = "empirical_mad"
            scores = df["spatial_score"].values
            med = np.median(scores)
            mad = np.median(np.abs(scores - med))
            s_cut = float(med + 2 * mad)

    def classify(row):
        high_c = row["coverage_expr"] >= c_cut

        if method == "fdr":
            high_s = (row["q_value"] < fdr_cut) and (row["spatial_score"] > 0)
        else:
            high_s = row["spatial_score"] >= s_cut

        if high_c and high_s:
            return "localized_program"
        elif high_c and not high_s:
            return "housekeeping_uniform"
        elif not high_c and high_s:
            return "niche_biomarker"
        else:
            return "sparse_presence"

    df["archetype"] = df.apply(classify, axis=1)

    df.attrs["c_cut"] = c_cut
    df.attrs["s_cut"] = s_cut
    df.attrs["s_cut_method"] = method

    return df
