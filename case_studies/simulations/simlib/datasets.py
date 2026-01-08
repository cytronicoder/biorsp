"""
Multi-gene dataset construction and AnnData packaging.

Builds gene panels with shared coordinates and library sizes.
"""

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from numpy.random import Generator

from .expression import (
    generate_confounded_null,
    generate_expression_from_field,
    generate_signal_field,
)


def make_gene_panel(
    coords: np.ndarray,
    libsize: np.ndarray,
    archetype_counts: Dict[str, int],
    rng: Generator,
    params: Dict[str, Any] = None,
) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    """
    Create multi-gene expression matrix with archetypes.

    Parameters
    ----------
    coords : np.ndarray
        Cell coordinates (n, 2)
    libsize : np.ndarray
        Shared library sizes (n,)
    archetype_counts : Dict[str, int]
        Number of genes per archetype, e.g., {'uniform': 5, 'wedge': 3}
    rng : Generator
        Random number generator
    params : Dict, optional
        Global expression parameters

    Returns
    -------
    X : np.ndarray
        Expression matrix (n_cells, n_genes)
    var_names : List[str]
        Gene names
    truth_df : pd.DataFrame
        Ground truth: gene, archetype, pattern, target_coverage, etc.
    """
    params = params or {}

    X_list = []
    var_names = []
    truth_rows = []

    gene_idx = 0
    for archetype, count in archetype_counts.items():
        for i in range(count):
            gene_name = f"{archetype}_{i}"
            var_names.append(gene_name)

            if archetype == "housekeeping_uniform":
                pattern = "uniform"
                field_params = {"base": 0.7}
                target_coverage = 0.7
            elif archetype == "niche_biomarker":
                pattern = "wedge_rim"
                angle = rng.uniform(-np.pi, np.pi)
                field_params = {"angle_center": angle, "width_rad": np.pi / 6, "steepness": 3.0}
                target_coverage = 0.05
            elif archetype == "localized_program":
                pattern = rng.choice(["core", "rim", "halfplane_gradient"])
                field_params = {"steepness": 3.0}
                target_coverage = 0.5
            elif archetype == "sparse_presence":
                pattern = "sparse"
                field_params = {"base": 0.02}
                target_coverage = 0.02
            else:

                pattern = archetype
                field_params = {}
                target_coverage = 0.3

            field = generate_signal_field(coords, pattern, field_params)

            expr_params = {**params, "abundance": params.get("abundance", 1e-3)}
            counts = generate_expression_from_field(
                field, libsize, rng, expr_model="nb", params=expr_params
            )

            X_list.append(counts)

            truth_rows.append(
                {
                    "gene": gene_name,
                    "archetype": archetype,
                    "pattern": pattern,
                    "target_coverage": target_coverage,
                    **field_params,
                }
            )
            gene_idx += 1

    X = np.column_stack(X_list)
    truth_df = pd.DataFrame(truth_rows)

    return X, var_names, truth_df


def make_module_panel(
    coords: np.ndarray,
    libsize: np.ndarray,
    modules: List[Dict[str, Any]],
    n_null: int,
    rng: Generator,
    params: Dict[str, Any] = None,
) -> Tuple[np.ndarray, List[str], pd.DataFrame, pd.DataFrame]:
    """
    Create gene panel with modules for gene-gene testing.

    Parameters
    ----------
    coords : np.ndarray
        Cell coordinates (n, 2)
    libsize : np.ndarray
        Shared library sizes (n,)
    modules : List[Dict]
        Module definitions: [{'name': 'mod1', 'n_genes': 3, 'pattern': 'wedge', 'params': {...}}]
    n_null : int
        Number of null genes (no pattern)
    rng : Generator
        Random number generator
    params : Dict, optional
        Expression parameters

    Returns
    -------
    X : np.ndarray
        Expression matrix (n_cells, n_genes)
    var_names : List[str]
        Gene names
    truth_edges_df : pd.DataFrame
        Ground truth edges: gene_a, gene_b, is_true_edge, module_a, module_b
    truth_df : pd.DataFrame
        Per-gene truth
    """
    params = params or {}
    X_list = []
    var_names = []
    truth_rows = []

    for mod_idx, mod in enumerate(modules):
        mod_name = mod.get("name", f"mod_{mod_idx}")
        n_genes = mod["n_genes"]
        pattern = mod["pattern"]
        pattern_params = mod.get("params", {})

        field = generate_signal_field(coords, pattern, pattern_params)

        for g_idx in range(n_genes):
            gene_name = f"{mod_name}_g{g_idx}"
            var_names.append(gene_name)

            noise_scale = params.get("module_noise", 0.1)
            field_noisy = np.clip(field + rng.normal(0, noise_scale, len(field)), 0, 1)

            counts = generate_expression_from_field(
                field_noisy, libsize, rng, expr_model="nb", params=params
            )
            X_list.append(counts)

            truth_rows.append(
                {
                    "gene": gene_name,
                    "module": mod_name,
                    "pattern": pattern,
                    **pattern_params,
                }
            )

    for i in range(n_null):
        gene_name = f"null_{i}"
        var_names.append(gene_name)

        counts, _ = generate_confounded_null(coords, libsize, rng, null_type="iid", params=params)
        X_list.append(counts)

        truth_rows.append(
            {
                "gene": gene_name,
                "module": "null",
                "pattern": "uniform",
            }
        )

    X = np.column_stack(X_list)
    truth_df = pd.DataFrame(truth_rows)

    edges = []
    for i, gene_a in enumerate(var_names):
        for j, gene_b in enumerate(var_names):
            if i >= j:
                continue
            mod_a = truth_df.iloc[i]["module"]
            mod_b = truth_df.iloc[j]["module"]
            is_true_edge = (mod_a == mod_b) and (mod_a != "null")
            edges.append(
                {
                    "gene_a": gene_a,
                    "gene_b": gene_b,
                    "module_a": mod_a,
                    "module_b": mod_b,
                    "is_true_edge": is_true_edge,
                }
            )

    truth_edges_df = pd.DataFrame(edges)

    return X, var_names, truth_edges_df, truth_df


def package_as_anndata(
    coords: np.ndarray,
    X: np.ndarray,
    var_names: List[str],
    obs_meta: Dict[str, np.ndarray] = None,
    embedding_key: str = "X_sim",
):
    """
    Package simulation into AnnData object.

    Parameters
    ----------
    coords : np.ndarray
        Cell coordinates (n, 2)
    X : np.ndarray
        Expression matrix (n, n_genes)
    var_names : List[str]
        Gene names
    obs_meta : Dict[str, np.ndarray], optional
        Additional cell-level metadata
    embedding_key : str, optional
        Key for storing coordinates in obsm

    Returns
    -------
    adata : AnnData
        Packaged dataset
    """
    try:
        from anndata import AnnData
    except ImportError as err:
        raise ImportError("anndata is required for packaging datasets") from err

    n_cells, n_genes = X.shape
    assert len(var_names) == n_genes

    obs_data = {"cell_id": [f"cell_{i}" for i in range(n_cells)]}
    if obs_meta:
        obs_data.update(obs_meta)
    obs_df = pd.DataFrame(obs_data)
    obs_df.index = obs_df["cell_id"]

    var_df = pd.DataFrame({"gene": var_names})
    var_df.index = var_names

    adata = AnnData(X=X, obs=obs_df, var=var_df)
    adata.obsm[embedding_key] = coords

    return adata
