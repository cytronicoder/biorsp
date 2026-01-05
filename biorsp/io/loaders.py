"""
I/O module for BioRSP.

Handles loading of expression and spatial data, and saving of results.
Supports CSV, TSV, and optionally AnnData (h5ad).
"""

import json
import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def load_expression_matrix(path: str, transpose: bool = False) -> pd.DataFrame:
    """
    Load expression matrix from file.

    Args:
        path: Path to file (csv, tsv, txt, h5ad).
        transpose: Whether to transpose the matrix (cells x genes expected).

    Returns:
        DataFrame with cells as rows and genes as columns.
    """
    ext = os.path.splitext(path)[1].lower()

    if ext in [".csv", ".txt"]:
        df = pd.read_csv(path, index_col=0)
    elif ext == ".tsv":
        df = pd.read_csv(path, sep="\t", index_col=0)
    elif ext == ".h5ad":
        try:
            import anndata

            adata = anndata.read_h5ad(path)
            df = adata.to_df()
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("anndata is required for .h5ad files") from exc
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    if transpose:
        df = df.T

    return df


def load_spatial_coords(path: str) -> np.ndarray:
    """
    Load spatial coordinates from file.

    Args:
        path: Path to file (csv, tsv). Expected columns: x, y (or first two columns).

    Returns:
        (N, 2) numpy array of coordinates.
    """
    ext = os.path.splitext(path)[1].lower()

    if ext in [".csv", ".txt"]:
        df = pd.read_csv(path, index_col=0)
    elif ext == ".tsv":
        df = pd.read_csv(path, sep="\t", index_col=0)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    if "x" in df.columns and "y" in df.columns:
        coords = df[["x", "y"]].values
    else:
        coords = df.iloc[:, :2].values

    return coords


def save_results(results: Dict[str, Any], path: str) -> None:
    """
    Save results to file.

    Args:
        results: Dictionary of results.
        path: Output path (.json).
    """

    # Convert numpy types to python types for JSON serialization
    def default_converter(o):
        if isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.floating):
            return float(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, default=default_converter, indent=2)


def load_umi_counts(
    path: str, n_cells: Optional[int] = None, column: Optional[str] = None
) -> np.ndarray:
    """
    Load UMI counts from file.

    Args:
        path: Path to file (csv, tsv, txt). Expected a single column or named column.
        n_cells: Expected number of cells for validation.
        column: Optional column name to select for UMI counts.

    Returns:
        (N,) numpy array of UMI counts.
    """
    ext = os.path.splitext(path)[1].lower()

    if ext in [".csv", ".txt"]:
        df = pd.read_csv(path)
    elif ext == ".tsv":
        df = pd.read_csv(path, sep="\t")
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    if column:
        if column not in df.columns:
            raise ValueError(f"UMI column '{column}' not found in {path}.")
        counts = df[column].values
    elif "umi" in df.columns:
        counts = df["umi"].values
    elif "umis" in df.columns:
        counts = df["umis"].values
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 1:
            counts = df[numeric_cols[0]].values
        else:
            raise ValueError(
                "UMI column not found. Provide a column named 'umi'/'umis' or "
                "specify --umi-column."
            )

    # Ensure numeric dtype and raise if coercion fails
    counts = pd.to_numeric(counts, errors="raise").to_numpy()

    if n_cells is not None and len(counts) != n_cells:
        raise ValueError(
            f"UMI counts length ({len(counts)}) does not match number of cells ({n_cells})."
        )

    return counts


__all__ = ["load_expression_matrix", "load_spatial_coords", "load_umi_counts", "save_results"]
