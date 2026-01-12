"""Input validation for BioRSP."""

from typing import Optional, Union

import numpy as np
import pandas as pd


def validate_inputs(
    coords: np.ndarray,
    expression: Union[np.ndarray, pd.Series, pd.DataFrame],
    umi_counts: Optional[np.ndarray] = None,
):
    """Validate coordinate and expression arrays.

    Parameters
    ----------
    coords : np.ndarray
        (N, 2) array of spatial coordinates.
    expression : Union[np.ndarray, pd.Series, pd.DataFrame]
        Expression values for one or more features.
    umi_counts : Optional[np.ndarray]
        (N,) array of total UMI counts per cell.

    Raises
    ------
    ValueError
        If shapes or types are invalid.

    """
    if not isinstance(coords, np.ndarray):
        raise ValueError(f"coords must be a numpy array, got {type(coords)}")

    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"coords must have shape (N, 2), got {coords.shape}")

    n_cells = coords.shape[0]

    if isinstance(expression, (pd.Series, pd.DataFrame)):
        expr_len = len(expression)
    else:
        expr_len = expression.shape[0]

    if expr_len != n_cells:
        raise ValueError(
            f"Expression length ({expr_len}) does not match number of cells ({n_cells})"
        )

    if umi_counts is not None:
        if umi_counts.shape[0] != n_cells:
            raise ValueError(
                f"UMI counts length ({umi_counts.shape[0]}) does not match "
                f"number of cells ({n_cells})"
            )
        if np.any(umi_counts < 0):
            raise ValueError("UMI counts must be non-negative")

    if np.any(np.isnan(coords)):
        raise ValueError("Coordinates contain NaNs")
