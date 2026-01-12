"""I/O module for BioRSP.

Handles loading of expression and spatial data, and saving of results.
Supports CSV, TSV, Parquet, and optionally AnnData (h5ad).

Features:
- Smart index detection (no blind index_col=0)
- Robust type conversion with clear error messages
- Alignment utilities to prevent silent mismatches
- Preserve cell IDs whenever possible
"""

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import anndata

logger = logging.getLogger(__name__)


def _infer_index_col(df: pd.DataFrame) -> Optional[str]:
    """Infer whether the first column should be used as index.

    Returns the column name if it should be index, None otherwise.

    Criteria:
    - First column dtype is object/string-like
    - High fraction (>0.9) of values are non-numeric
    - Values are mostly unique (>0.95)
    """
    if df.empty or df.shape[1] == 0:
        return None

    first_col = df.columns[0]
    values = df[first_col]

    if not pd.api.types.is_string_dtype(values) and not pd.api.types.is_object_dtype(values):
        return None

    try:
        numeric = pd.to_numeric(values, errors="coerce")
        non_numeric_frac = numeric.isna().sum() / len(values)
        if non_numeric_frac < 0.9:
            return None
    except Exception:
        pass

    unique_frac = values.nunique() / len(values)
    if unique_frac < 0.95:
        return None

    return first_col


def load_expression_matrix(
    path: str,
    transpose: bool = False,
    return_type: Literal["dataframe", "anndata"] = "dataframe",
    genes: Optional[List[str]] = None,
    layer: Optional[str] = None,
    use_raw: bool = False,
) -> Union[pd.DataFrame, "anndata.AnnData"]:
    """Load expression matrix from file with smart index detection.

    Args:
        path: Path to file (csv, tsv, txt, h5ad, parquet).
        transpose: Whether to transpose the matrix (cells x genes expected).
        return_type: "dataframe" or "anndata" (for h5ad only).
        genes: Optional list of genes to extract (for h5ad, prevents full densification).
        layer: Optional layer name for h5ad files.
        use_raw: Use .raw.X from h5ad if available.

    Returns:
        DataFrame with cells as rows and genes as columns (and cell IDs as index if detected),
        or AnnData object if return_type="anndata".

    Raises:
        ValueError: If file format unsupported, or h5ad too large without genes specified.
        ImportError: If anndata not installed for .h5ad files.

    Notes:
        - For CSV/TSV: Uses smart index detection instead of blindly index_col=0
        - For h5ad: If genes not specified and matrix is large/sparse, will raise error
          to prevent accidental densification. Pass genes parameter or use return_type="anndata".
        - After transpose, validates that index looks like cell IDs (many unique values)
    """
    path_obj = Path(path)
    ext = path_obj.suffix.lower()

    if ext in [".csv", ".txt"]:
        df = pd.read_csv(path)
        index_col = _infer_index_col(df)
        if index_col:
            df = df.set_index(index_col)
            logger.debug(f"Detected index column: {index_col}")
    elif ext == ".tsv":
        df = pd.read_csv(path, sep="\t")
        index_col = _infer_index_col(df)
        if index_col:
            df = df.set_index(index_col)
            logger.debug(f"Detected index column: {index_col}")
    elif ext == ".parquet":
        df = pd.read_parquet(path)
    elif ext == ".h5ad":
        try:
            import anndata
        except ImportError as exc:
            raise ImportError(
                "anndata is required for .h5ad files. Install with: pip install anndata"
            ) from exc

        adata = anndata.read_h5ad(path)

        if return_type == "anndata":
            return adata

        if genes is not None:
            gene_mask = adata.var_names.isin(genes)
            if gene_mask.sum() == 0:
                raise ValueError(f"None of the requested genes found in {path}")
            logger.info(f"Extracting {gene_mask.sum()} genes from h5ad")

            if use_raw and adata.raw is not None:
                X = adata.raw.X[:, adata.raw.var_names.isin(genes)]
                var_names = adata.raw.var_names[adata.raw.var_names.isin(genes)]
            else:
                X = adata.X[:, gene_mask]
                var_names = adata.var_names[gene_mask]

            if hasattr(X, "toarray"):
                X = X.toarray()
            df = pd.DataFrame(X, index=adata.obs_names, columns=var_names)
        else:
            n_cells, n_genes = adata.shape
            if hasattr(adata.X, "toarray"):
                density = adata.X.nnz / (n_cells * n_genes) if n_cells * n_genes > 0 else 0
                if n_cells * n_genes > 10_000_000:
                    raise ValueError(
                        f"Expression matrix is large ({n_cells} cells × {n_genes} genes, "
                        f"density={density:.2%}). To prevent accidental densification:\n"
                        f"  - Pass genes=[...] to extract specific genes, or\n"
                        f"  - Use return_type='anndata' to work with AnnData directly"
                    )

            df = adata.to_df(layer=layer)
    else:
        raise ValueError(
            f"Unsupported file extension: {ext}. Supported: .csv, .tsv, .txt, .parquet, .h5ad"
        )

    if df.index.duplicated().any():
        logger.warning(
            f"Duplicate cell IDs found in expression matrix: {df.index.duplicated().sum()} duplicates"
        )
    if df.columns.duplicated().any():
        logger.warning(f"Duplicate gene names found: {df.columns.duplicated().sum()} duplicates")

    if transpose:
        df = df.T
        unique_frac = df.index.nunique() / len(df.index)
        if unique_frac < 0.95:
            logger.warning(
                f"After transpose, index has low uniqueness ({unique_frac:.1%}). "
                "This may not be cell IDs. Check if transpose is correct."
            )

    return df


def load_spatial_coords(path: str) -> pd.DataFrame:
    """Load spatial coordinates from file with smart detection.

    Args:
        path: Path to file (csv, tsv, txt, parquet).

    Returns:
        DataFrame with exactly two columns ["x", "y"] and cell IDs as index if detected.
        Coordinates are validated: no NaN/inf, and index uniqueness if IDs present.

    Raises:
        ValueError: If coordinates invalid or cannot find exactly 2 numeric columns.

    Notes:
        - Prefers columns named x,y (case-insensitive), or X,Y, or umap_1,umap_2
        - Otherwise selects first two numeric columns
        - Uses smart index detection instead of blindly index_col=0
    """
    path_obj = Path(path)
    ext = path_obj.suffix.lower()

    if ext in [".csv", ".txt"]:
        df = pd.read_csv(path)
        index_col = _infer_index_col(df)
        if index_col:
            df = df.set_index(index_col)
            logger.debug(f"Detected index column: {index_col}")
    elif ext == ".tsv":
        df = pd.read_csv(path, sep="\t")
        index_col = _infer_index_col(df)
        if index_col:
            df = df.set_index(index_col)
            logger.debug(f"Detected index column: {index_col}")
    elif ext == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(
            f"Unsupported file extension: {ext}. Supported: .csv, .tsv, .txt, .parquet"
        )

    coord_cols = None

    lower_cols = {col.lower(): col for col in df.columns}
    if "x" in lower_cols and "y" in lower_cols:
        coord_cols = [lower_cols["x"], lower_cols["y"]]
    elif "umap_1" in lower_cols and "umap_2" in lower_cols:
        coord_cols = [lower_cols["umap_1"], lower_cols["umap_2"]]
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 2:
            coord_cols = numeric_cols[:2]

    if coord_cols is None or len(coord_cols) != 2:
        raise ValueError(
            f"Could not find exactly 2 coordinate columns in {path}. "
            f"Available columns: {list(df.columns)}"
        )

    coords_df = df[coord_cols].copy()
    coords_df.columns = ["x", "y"]

    if coords_df.isna().any().any():
        raise ValueError(f"Coordinates contain NaN values in {path}")
    if np.isinf(coords_df.values).any():
        raise ValueError(f"Coordinates contain infinite values in {path}")

    if coords_df.index.duplicated().any():
        logger.warning(
            f"Duplicate cell IDs in coordinates: {coords_df.index.duplicated().sum()} duplicates"
        )

    return coords_df


def load_umi_counts(
    path: str,
    n_cells: Optional[int] = None,
    column: Optional[str] = None,
) -> Union[pd.Series, np.ndarray]:
    """Load UMI counts from file with robust type handling.

    Args:
        path: Path to file (csv, tsv, txt). Expected a single column or named column.
        n_cells: Expected number of cells for validation.
        column: Optional column name to select for UMI counts.

    Returns:
        pd.Series indexed by cell ID if IDs detected, otherwise np.ndarray.
        Always returns numeric values (int or float).

    Raises:
        ValueError: If column not found, multiple numeric columns without specification,
                   or length mismatch with n_cells.
        TypeError: If values cannot be converted to numeric.

    Notes:
        - Preserves cell IDs when CSV has explicit ID column + numeric column
        - If no IDs detected, returns ndarray and emits warning
        - Uses robust conversion: pd.to_numeric with errors="raise" then np.asarray
    """
    path_obj = Path(path)
    ext = path_obj.suffix.lower()

    if ext in [".csv", ".txt"]:
        df = pd.read_csv(path)
    elif ext == ".tsv":
        df = pd.read_csv(path, sep="\t")
    else:
        raise ValueError(f"Unsupported file extension: {ext}. Supported: .csv, .tsv, .txt")

    index_col = _infer_index_col(df)
    cell_ids = None
    if index_col:
        cell_ids = df[index_col]
        df = df.drop(columns=[index_col])
        logger.debug(f"Detected cell ID column: {index_col}")

    if column:
        if column not in df.columns:
            raise ValueError(
                f"UMI column '{column}' not found in {path}. Available: {list(df.columns)}"
            )
        counts_raw = df[column]
    elif "umi" in df.columns:
        counts_raw = df["umi"]
    elif "umis" in df.columns:
        counts_raw = df["umis"]
    elif "umi_count" in df.columns:
        counts_raw = df["umi_count"]
    elif "umi_counts" in df.columns:
        counts_raw = df["umi_counts"]
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 1:
            counts_raw = df[numeric_cols[0]]
            logger.debug(f"Using single numeric column: {numeric_cols[0]}")
        elif len(numeric_cols) == 0:
            raise ValueError(f"No numeric columns found in {path}. Cannot determine UMI counts.")
        else:
            raise ValueError(
                f"Multiple numeric columns found in {path}: {list(numeric_cols)}. "
                "Specify column name with --umi-column or rename one to 'umi'/'umis'."
            )

    try:
        counts_numeric = pd.to_numeric(counts_raw, errors="raise")
    except (ValueError, TypeError) as e:
        raise TypeError(f"UMI counts contain non-numeric values in {path}: {e}") from e

    counts_array = np.asarray(counts_numeric, dtype=np.float64)

    if n_cells is not None and len(counts_array) != n_cells:
        raise ValueError(
            f"UMI counts length ({len(counts_array)}) does not match expected number of cells ({n_cells})."
        )

    if cell_ids is not None:
        counts_series = pd.Series(counts_array, index=cell_ids, name="umi_counts")
        if counts_series.index.duplicated().any():
            logger.warning(
                f"Duplicate cell IDs in UMI counts: {counts_series.index.duplicated().sum()}"
            )
        return counts_series
    else:
        logger.warning(
            f"No cell IDs detected in {path}. Returning ndarray. "
            "This requires positional alignment with expression/coordinates."
        )
        return counts_array


def save_results(results: Dict[str, Any], path: str) -> None:
    """Save results to file with enhanced serialization.

    Args:
        results: Dictionary of results.
        path: Output path (.json).

    Notes:
        - Automatically creates parent directories if they don't exist
        - Handles Path objects, pandas Series/DataFrame, dataclasses, numpy scalars
        - Stable formatting: sorted keys, indent=2
    """
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    def default_converter(o):
        if isinstance(o, Path):
            return str(o)
        elif isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.floating):
            return float(o)
        elif isinstance(o, np.bool_):
            return bool(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, pd.Series):
            return {"index": o.index.tolist(), "values": o.tolist()}
        elif isinstance(o, pd.DataFrame):
            return o.to_dict(orient="split")
        elif hasattr(o, "__dataclass_fields__"):
            from dataclasses import asdict

            return asdict(o)
        raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, default=default_converter, indent=2, sort_keys=True)


def align_inputs(
    expr: Union[pd.DataFrame, "anndata.AnnData"],
    coords: pd.DataFrame,
    umi: Optional[Union[pd.Series, np.ndarray]] = None,
    how: Literal["inner", "left"] = "inner",
    min_overlap: float = 0.5,
    verbose: bool = True,
) -> Tuple[
    Union[pd.DataFrame, "anndata.AnnData"],
    pd.DataFrame,
    Optional[Union[pd.Series, np.ndarray]],
    Dict[str, Any],
]:
    """Align expression, coordinates, and optional UMI counts by cell ID.

    This function prevents silent mismatches between inputs by:
    - Checking cell ID overlap across all inputs
    - Aligning by intersection or left-join
    - Reporting dropped cells and overlap statistics
    - Enforcing minimum overlap threshold

    Args:
        expr: Expression DataFrame (cells x genes) or AnnData object
        coords: Coordinates DataFrame with ["x", "y"] columns
        umi: Optional UMI counts (Series indexed by cell ID, or ndarray for positional alignment)
        how: Alignment strategy - "inner" (intersection) or "left" (expr-driven)
        min_overlap: Minimum required overlap fraction (0.0-1.0)
        verbose: Print alignment report

    Returns:
        Tuple of (aligned_expr, aligned_coords, aligned_umi, report_dict)

    Raises:
        ValueError: If overlap below min_overlap threshold
        TypeError: If umi is ndarray but lengths don't match

    Notes:
        - If umi is ndarray, requires exact positional match with expr
        - Report dict contains: n_expr, n_coords, n_umi, n_overlap, dropped_*, overlap_fraction_*
    """
    if hasattr(expr, "obs_names"):
        expr_ids = expr.obs_names
        is_anndata = True
    else:
        expr_ids = expr.index
        is_anndata = False

    coords_ids = coords.index

    report = {
        "n_expr": len(expr_ids),
        "n_coords": len(coords_ids),
        "n_umi": None,
        "n_overlap": 0,
        "dropped_expr": [],
        "dropped_coords": [],
        "dropped_umi": [],
        "overlap_fraction_expr": 0.0,
        "overlap_fraction_coords": 0.0,
        "overlap_fraction_umi": None,
    }

    umi_ids = None
    if umi is not None:
        if isinstance(umi, pd.Series):
            umi_ids = umi.index
            report["n_umi"] = len(umi_ids)
        elif isinstance(umi, np.ndarray):
            report["n_umi"] = len(umi)
            if len(umi) != len(expr_ids):
                raise TypeError(
                    f"UMI counts provided as ndarray with length {len(umi)}, "
                    f"but expression has {len(expr_ids)} cells. "
                    "For positional alignment, lengths must match. "
                    "Consider converting UMI to pd.Series with cell IDs."
                )
            logger.warning(
                "UMI counts provided as ndarray. Using positional alignment. "
                "This assumes UMI order matches expression order exactly."
            )
        else:
            raise TypeError(f"umi must be pd.Series or np.ndarray, got {type(umi)}")

    if how == "inner" or how == "left":
        common_ids = expr_ids.intersection(coords_ids)
        if umi_ids is not None:
            common_ids = common_ids.intersection(umi_ids)
    else:
        raise ValueError(f"how must be 'inner' or 'left', got {how}")

    report["n_overlap"] = len(common_ids)

    if len(expr_ids) > 0:
        report["overlap_fraction_expr"] = len(common_ids) / len(expr_ids)
    if len(coords_ids) > 0:
        report["overlap_fraction_coords"] = len(common_ids) / len(coords_ids)
    if umi_ids is not None and len(umi_ids) > 0:
        report["overlap_fraction_umi"] = len(common_ids) / len(umi_ids)

    if report["overlap_fraction_expr"] < min_overlap:
        raise ValueError(
            f"Insufficient overlap between inputs: {report['overlap_fraction_expr']:.1%} < {min_overlap:.1%}\n"
            f"  Expression: {len(expr_ids)} cells\n"
            f"  Coordinates: {len(coords_ids)} cells\n"
            f"  Overlap: {len(common_ids)} cells\n"
            "Check that cell IDs match across input files."
        )

    report["dropped_expr"] = list(expr_ids.difference(common_ids))
    report["dropped_coords"] = list(coords_ids.difference(common_ids))
    if umi_ids is not None:
        report["dropped_umi"] = list(umi_ids.difference(common_ids))

    aligned_expr = expr[common_ids].copy() if is_anndata else expr.loc[common_ids].copy()
    aligned_coords = coords.loc[common_ids].copy()

    aligned_umi = None
    if umi is not None:
        if isinstance(umi, pd.Series):
            aligned_umi = umi.loc[common_ids].copy()
        else:
            if is_anndata:
                keep_mask = expr.obs_names.isin(common_ids)
            else:
                keep_mask = expr.index.isin(common_ids)
            aligned_umi = umi[keep_mask]

    if verbose:
        logger.info("=" * 60)
        logger.info("Input Alignment Report")
        logger.info("=" * 60)
        logger.info(
            f"Expression:   {report['n_expr']:>6} cells → {len(common_ids):>6} retained ({report['overlap_fraction_expr']:>6.1%})"
        )
        logger.info(
            f"Coordinates:  {report['n_coords']:>6} cells → {len(common_ids):>6} retained ({report['overlap_fraction_coords']:>6.1%})"
        )
        if report["n_umi"] is not None:
            logger.info(
                f"UMI counts:   {report['n_umi']:>6} cells → {len(common_ids):>6} retained ({report['overlap_fraction_umi']:>6.1%})"
            )
        logger.info(f"Alignment:    {how}")
        logger.info(f"Final:        {len(common_ids)} cells")
        logger.info("=" * 60)

    return aligned_expr, aligned_coords, aligned_umi, report


__all__ = [
    "load_expression_matrix",
    "load_spatial_coords",
    "load_umi_counts",
    "save_results",
    "align_inputs",
]
