"""
Validation utilities for plotting and data integrity.

Prevents silent failures and empty figure generation.
"""

from typing import List

import pandas as pd


class ValidationError(Exception):
    """Raised when data validation fails."""

    pass


def validate_dataframe_for_plot(
    df: pd.DataFrame,
    required_columns: List[str],
    min_rows: int = 1,
    name: str = "DataFrame",
) -> None:
    """
    Validate DataFrame before plotting.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate
    required_columns : List[str]
        Required column names
    min_rows : int
        Minimum required rows
    name : str
        Name for error messages

    Raises
    ------
    ValidationError
        If validation fails
    """
    # Check exists
    if df is None:
        raise ValidationError(f"{name} is None")

    # Check not empty
    if len(df) < min_rows:
        raise ValidationError(
            f"{name} has {len(df)} rows, but {min_rows} required. "
            f"Cannot generate plot with insufficient data."
        )

    # Check required columns
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValidationError(
            f"{name} missing required columns: {missing}. " f"Available columns: {list(df.columns)}"
        )

    # Check for all-NaN required columns
    for col in required_columns:
        if df[col].isna().all():
            raise ValidationError(
                f"{name} column '{col}' is all NaN. " f"Cannot plot with no valid data."
            )


def log_dataframe_stats(df: pd.DataFrame, name: str = "DataFrame"):
    """
    Log DataFrame statistics for debugging.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to log
    name : str
        Name for log messages
    """
    print(f"\n{name} Statistics:")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")

    # Numeric columns stats
    numeric_cols = df.select_dtypes(include=["number"]).columns
    if len(numeric_cols) > 0:
        print("  Numeric columns summary:")
        for col in numeric_cols:
            n_valid = df[col].notna().sum()
            n_total = len(df)
            if n_valid > 0:
                mean_val = df[col].mean()
                std_val = df[col].std()
                print(
                    f"    {col}: {n_valid}/{n_total} valid, mean={mean_val:.3f}, std={std_val:.3f}"
                )
            else:
                print(f"    {col}: 0/{n_total} valid (all NaN)")


def check_estimate_convergence(
    results_df: pd.DataFrame, metric: str, replicate_col: str = "replicate", threshold: float = 0.05
) -> dict:
    """
    Check if an estimate (e.g., FPR, mean score) stabilizes with increasing reps.

    Helps validate that n_reps is sufficient for stable estimates.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results with replicate column
    metric : str
        Column name to check convergence (e.g., 'spatial_score', 'p_value')
    replicate_col : str
        Column name for replicate numbers
    threshold : float
        Acceptable relative change threshold (e.g., 0.05 = 5%)

    Returns
    -------
    dict
        Convergence report with keys:
        - 'converged': bool, whether estimate stabilized
        - 'final_estimate': float, estimate at max reps
        - 'max_deviation': float, max % deviation from final estimate
        - 'convergence_rep': int, rep number where convergence achieved
        - 'values_by_rep': dict, estimate at each replicate count
    """
    if metric not in results_df.columns:
        return {"error": f"Metric {metric} not found in results"}

    # Group by cumulative replicates
    n_reps = results_df[replicate_col].max() + 1  # 0-indexed
    values_by_rep = {}
    max_devs = []

    for rep in range(1, n_reps + 1):
        subset = results_df[results_df[replicate_col] < rep]
        if len(subset) > 0:
            # For p-values, use median; for scores, use mean
            if "p_value" in metric or "pval" in metric.lower():
                est = subset[metric].median()
            else:
                est = subset[metric].mean()
            values_by_rep[rep] = float(est)

    if len(values_by_rep) < 2:
        return {"warning": "Insufficient replicates to assess convergence"}

    final = values_by_rep[max(values_by_rep.keys())]
    if final == 0:
        final = 1e-6  # Avoid division by zero

    max_deviation = 0
    convergence_rep = None

    for rep, val in sorted(values_by_rep.items()):
        deviation = abs((val - final) / final)
        max_devs.append(deviation)

        if deviation <= threshold and convergence_rep is None:
            convergence_rep = rep

        max_deviation = max(max_deviation, deviation)

    converged = convergence_rep is not None

    return {
        "converged": converged,
        "final_estimate": final,
        "max_deviation": max_deviation,
        "convergence_rep": convergence_rep,
        "values_by_rep": values_by_rep,
        "recommendation": (
            f"Use {convergence_rep or n_reps} reps" if converged else f"Consider >={n_reps} reps"
        ),
    }
