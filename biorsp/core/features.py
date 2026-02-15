"""Feature namespace resolution utilities."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

SYMBOL_COLUMNS: tuple[str, ...] = ("hugo_symbol", "gene_name", "gene_symbol")


def _pick_symbol_column(adata_like: Any) -> str | None:
    if not hasattr(adata_like, "var") or adata_like.var is None:
        return None
    for col in SYMBOL_COLUMNS:
        if col in adata_like.var.columns:
            return str(col)
    return None


def resolve_feature_index(
    adata_like: Any,
    feature: str,
) -> tuple[int, str, str | None, str]:
    """Resolve feature index without mutating `var_names`.

    Returns `(idx, display_label, symbol_column_used, resolution_source)`.
    """
    key = str(feature).strip()
    if key == "":
        raise KeyError("Feature name is empty.")

    var_names = pd.Index(adata_like.var_names)
    symbol_col = _pick_symbol_column(adata_like)

    if symbol_col is not None:
        raw = (
            adata_like.var[symbol_col]
            .astype("string")
            .fillna("")
            .astype(str)
            .str.strip()
        )
        non_empty = raw[raw != ""]
        counts = non_empty.value_counts()
        dup = counts[counts > 1]
        if not dup.empty:
            warnings.warn(
                (
                    f"Duplicate symbols detected in {symbol_col}; "
                    "resolution keeps first occurrence."
                ),
                RuntimeWarning,
                stacklevel=2,
            )

        if key in non_empty.values:
            idx = int(np.flatnonzero((raw == key).to_numpy())[0])
            return idx, key, symbol_col, "symbol"

    if key in var_names:
        loc = var_names.get_loc(key)
        if isinstance(loc, (int, np.integer)):
            idx = int(loc)
        elif isinstance(loc, np.ndarray) and loc.size > 0:
            idx = int(np.flatnonzero(loc)[0])
        else:
            raise KeyError(f"Feature '{feature}' did not resolve uniquely in var_names.")
        label = key
        if symbol_col is not None:
            sym = str(adata_like.var.iloc[idx][symbol_col]).strip()
            if sym != "":
                label = sym
        return idx, label, symbol_col, "var_names"

    raise KeyError(f"Feature '{feature}' not found in symbol columns or var_names.")


def get_feature_vector(expr_matrix: Any, idx: int) -> np.ndarray:
    vec = expr_matrix[:, int(idx)]
    if sp.issparse(vec):
        return vec.toarray().ravel().astype(float)
    return np.asarray(vec).ravel().astype(float)
