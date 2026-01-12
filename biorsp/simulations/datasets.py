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


def make_factorial_panel(
    coords: np.ndarray,
    libsize: np.ndarray,
    rng: Generator,
    n_per_archetype: int = 20,
    pattern_variants: List[str] = None,
    include_abstention_stress: bool = False,
) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    """
    Create multi-gene panel with 2x2 factorial design (coverage × organization).

    This generates genes spanning 4 archetypes:
    - housekeeping: high coverage, no spatial structure (iid)
    - regional_program: high coverage, spatially structured (radial patterns)
    - sparse_noise: low coverage, no spatial structure (iid)
    - niche_marker: low coverage, spatially structured (localized radial patterns)

    IMPORTANT: Pattern selection is aligned with what BioRSP's S score can detect.
    S measures *radial* organization (mean radius of expressing vs non-expressing cells).

    Patterns used:
    - regional_program: core, rim, radial_gradient (broad radial structure → high |S|)
    - niche_marker: wedge_core, wedge_rim (localized + radial bias → high |S|)
    - housekeeping/sparse_noise: uniform (no radial structure → low |S|)

    Parameters
    ----------
    coords : np.ndarray
        Cell coordinates (n, 2)
    libsize : np.ndarray
        Library sizes (n,)
    rng : Generator
        Random number generator
    n_per_archetype : int, optional
        Number of genes per archetype (total = 4 × n_per_archetype)
    pattern_variants : List[str], optional
        Override default patterns. NOT recommended unless you understand S-detectability.
        Default uses S-detectable patterns per archetype.
    include_abstention_stress : bool, optional
        Include extreme low-coverage genes that should trigger abstention

    Returns
    -------
    X : np.ndarray
        Expression matrix (n_cells, n_genes)
    var_names : List[str]
        Gene names
    truth_df : pd.DataFrame
        Ground truth with columns: gene, archetype, coverage_regime, organization_regime,
        pattern_variant, target_coverage, achieved_coverage
    """
    from .expression import generate_factorial_gene

    regional_program_patterns = ["core", "rim", "radial_gradient"]
    niche_marker_patterns = ["wedge_core", "wedge_rim"]

    if pattern_variants is not None:
        regional_program_patterns = pattern_variants
        niche_marker_patterns = pattern_variants

    X_list = []
    var_names = []
    truth_rows = []

    gene_idx = 0

    archetypes = [
        ("high", "iid"),  # housekeeping
        ("high", "structured"),  # regional_program
        ("low", "iid"),  # sparse_noise
        ("low", "structured"),  # niche_marker
    ]

    for cov_regime, org_regime in archetypes:
        for i in range(n_per_archetype):
            if org_regime == "structured":
                if cov_regime == "high":
                    patterns_for_archetype = regional_program_patterns
                else:
                    patterns_for_archetype = niche_marker_patterns

                pattern_var = patterns_for_archetype[i % len(patterns_for_archetype)]

                if pattern_var == "wedge":
                    pattern_params = {
                        "angle_center": rng.uniform(-np.pi, np.pi),
                        "width_rad": rng.uniform(np.pi / 6, np.pi / 3),
                    }
                elif pattern_var in ["core", "rim"]:
                    pattern_params = {"steepness": rng.uniform(3.0, 8.0)}
                elif pattern_var == "radial_gradient":
                    pattern_params = {
                        "direction": rng.choice(["outward", "inward"]),
                        "strength": rng.uniform(0.6, 1.0),
                    }
                elif pattern_var in ["wedge_core", "wedge_rim"]:
                    pattern_params = {
                        "angle_center": rng.uniform(-np.pi, np.pi),
                        "width_rad": rng.uniform(np.pi / 4, np.pi / 2),
                        "steepness": rng.uniform(3.0, 6.0),
                    }
                elif pattern_var == "halfplane_gradient":
                    pattern_params = {"phi": rng.uniform(-np.pi, np.pi)}
                else:
                    pattern_params = {}
            else:
                pattern_var = "none"
                pattern_params = {}

            counts, meta = generate_factorial_gene(
                coords=coords,
                libsize=libsize,
                rng=rng,
                coverage_regime=cov_regime,
                organization_regime=org_regime,
                pattern_variant=pattern_var if org_regime == "structured" else "uniform",
                pattern_params=pattern_params,
            )

            gene_name = f"{meta['archetype']}_{i}"
            X_list.append(counts)
            var_names.append(gene_name)

            truth_rows.append(
                {
                    "gene": gene_name,
                    "gene_idx": gene_idx,
                    "archetype": meta["archetype"],
                    "coverage_regime": cov_regime,
                    "organization_regime": org_regime,
                    "pattern_variant": pattern_var,
                    "target_coverage": meta["target_coverage"],
                    "achieved_coverage": meta["achieved_coverage"],
                }
            )
            gene_idx += 1

    if include_abstention_stress:
        for i in range(5):
            counts, meta = generate_factorial_gene(
                coords=coords,
                libsize=libsize,
                rng=rng,
                coverage_regime="low",
                organization_regime="structured" if i % 2 == 0 else "iid",
                pattern_variant="wedge_core",  # Changed from pure wedge
                coverage_params={"target": 0.02},
            )
            gene_name = f"abstention_stress_{i}"
            X_list.append(counts)
            var_names.append(gene_name)
            truth_rows.append(
                {
                    "gene": gene_name,
                    "gene_idx": gene_idx,
                    "archetype": "abstention_stress",
                    "coverage_regime": "extreme_low",
                    "organization_regime": meta["organization_regime"],
                    "pattern_variant": meta.get("pattern_variant", "none"),
                    "target_coverage": 0.02,
                    "achieved_coverage": meta["achieved_coverage"],
                }
            )
            gene_idx += 1

    X = np.column_stack(X_list)
    truth_df = pd.DataFrame(truth_rows)

    return X, var_names, truth_df


def make_module_panel_structured(
    coords: np.ndarray,
    libsize: np.ndarray,
    rng: Generator,
    n_modules: int = 4,
    genes_per_module: int = 12,
    n_null_genes: int = 20,
    module_patterns: List[str] = None,
) -> Tuple[np.ndarray, List[str], pd.DataFrame, pd.DataFrame]:
    """
    Create gene panel with distinct spatial modules for gene-gene co-patterning.

    Each module shares a common spatial pattern, so genes within a module
    should show high co-patterning scores. Genes across modules and null genes
    should show low co-patterning.

    Parameters
    ----------
    coords : np.ndarray
        Cell coordinates (n, 2)
    libsize : np.ndarray
        Library sizes (n,)
    rng : Generator
        Random number generator
    n_modules : int, optional
        Number of co-expression modules
    genes_per_module : int, optional
        Number of genes per module
    n_null_genes : int, optional
        Number of null (spatially random) genes
    module_patterns : List[str], optional
        Spatial patterns for each module. If None, auto-generated with distinct angles.

    Returns
    -------
    X : np.ndarray
        Expression matrix (n_cells, n_genes)
    var_names : List[str]
        Gene names
    truth_edges_df : pd.DataFrame
        Ground truth edges: gene_a, gene_b, is_true_edge, module_a, module_b
    truth_genes_df : pd.DataFrame
        Per-gene truth: gene, module, pattern, is_module_gene
    """
    from .expression import generate_expression_targeted

    if module_patterns is None:
        angles = np.linspace(0, 2 * np.pi, n_modules, endpoint=False)
        module_patterns = [("wedge", {"angle_center": a, "width_rad": np.pi / 4}) for a in angles]

    X_list = []
    var_names = []
    truth_rows = []

    for mod_idx, (pattern, pattern_params) in enumerate(module_patterns):
        module_name = f"module_{mod_idx}"

        for g_idx in range(genes_per_module):
            gene_name = f"{module_name}_g{g_idx}"
            var_names.append(gene_name)

            noisy_params = {**pattern_params}
            if "angle_center" in noisy_params:
                noisy_params["angle_center"] += rng.uniform(-0.1, 0.1)
            if "width_rad" in noisy_params:
                noisy_params["width_rad"] += rng.uniform(-0.05, 0.05)

            counts, meta = generate_expression_targeted(
                coords=coords,
                libsize=libsize,
                rng=rng,
                pattern=pattern,
                target_coverage=rng.uniform(0.15, 0.35),
                pattern_params=noisy_params,
            )

            X_list.append(counts)
            truth_rows.append(
                {
                    "gene": gene_name,
                    "module": module_name,
                    "pattern": pattern,
                    "is_module_gene": True,
                    **meta,
                }
            )

    from .expression import generate_confounded_null

    for i in range(n_null_genes):
        gene_name = f"null_{i}"
        var_names.append(gene_name)

        counts, _ = generate_confounded_null(
            coords, libsize, rng, null_type="iid", params={"base_prob": rng.uniform(0.1, 0.4)}
        )
        X_list.append(counts)
        truth_rows.append(
            {
                "gene": gene_name,
                "module": "null",
                "pattern": "iid",
                "is_module_gene": False,
            }
        )

    X = np.column_stack(X_list)
    truth_genes_df = pd.DataFrame(truth_rows)

    edges = []
    n_genes = len(var_names)
    for i in range(n_genes):
        for j in range(i + 1, n_genes):
            gene_a = var_names[i]
            gene_b = var_names[j]
            mod_a = truth_genes_df.iloc[i]["module"]
            mod_b = truth_genes_df.iloc[j]["module"]
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

    return X, var_names, truth_edges_df, truth_genes_df
