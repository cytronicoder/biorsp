"""
I/O module for BioRSP.

Handles loading of expression and spatial data, and saving of results.
Supports CSV, TSV, and optionally AnnData (h5ad).
"""

import json
import os
from typing import Any, Dict

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


__all__ = ["load_expression_matrix", "load_spatial_coords", "save_results"]
