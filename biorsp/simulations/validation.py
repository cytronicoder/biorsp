"""
Validation utilities for plotting and data integrity.

Prevents silent failures and empty figure generation.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd


class ValidationError(Exception):
    """Raised when data validation fails."""

    pass


class RunsValidationReport:
    """Result of runs.csv validation with diagnostics."""

    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.stats: Dict[str, Any] = {}
        self.invariants_checked: List[str] = []
        self.invariants_failed: List[str] = []

    @property
    def valid(self) -> bool:
        return len(self.errors) == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "n_errors": len(self.errors),
            "n_warnings": len(self.warnings),
            "errors": self.errors,
            "warnings": self.warnings,
            "stats": self.stats,
            "invariants_checked": self.invariants_checked,
            "invariants_failed": self.invariants_failed,
        }

    def to_json(self, path: Union[str, Path]) -> None:
        """Write validation report to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    def __repr__(self) -> str:
        status = "✓ VALID" if self.valid else "✗ INVALID"
        return f"RunsValidationReport({status}, {len(self.errors)} errors, {len(self.warnings)} warnings)"


def load_and_validate_runs(
    path: Union[str, Path],
    benchmark: str,
    expected_shapes: Optional[List[str]] = None,
    expected_N: Optional[List[int]] = None,
    expected_null_types: Optional[List[str]] = None,
    expected_coverage_regimes: Optional[List[str]] = None,
    expected_organization_regimes: Optional[List[str]] = None,
    write_debug_json: bool = True,
) -> tuple[pd.DataFrame, RunsValidationReport]:
    """
    Load runs.csv and validate against scientific invariants.

    This function ensures that benchmark outputs are scientifically correct
    by checking:
    1. All expected sweep dimensions are present
    2. Pattern variants match organization regimes (IID → uniform, structured → spatial)
    3. Coverage values are in expected ranges for each regime
    4. No unexpected constant columns that should vary

    Parameters
    ----------
    path : str or Path
        Path to runs.csv
    benchmark : str
        Benchmark type: 'archetypes' or 'calibration'
    expected_shapes : list of str, optional
        Expected geometry shapes (e.g., ['disk', 'ellipse', 'annulus'])
    expected_N : list of int, optional
        Expected sample sizes (e.g., [500, 1000, 2000])
    expected_null_types : list of str, optional
        For calibration: expected null types
    expected_coverage_regimes : list of str, optional
        For archetypes: expected coverage regimes
    expected_organization_regimes : list of str, optional
        For archetypes: expected organization regimes
    write_debug_json : bool
        Whether to write debug JSON alongside the input file

    Returns
    -------
    df : pd.DataFrame
        Loaded DataFrame
    report : RunsValidationReport
        Validation report with errors, warnings, and statistics
    """
    path = Path(path)
    report = RunsValidationReport()

    # Load data
    if not path.exists():
        report.errors.append(f"File not found: {path}")
        return pd.DataFrame(), report

    try:
        df = pd.read_csv(path)
    except Exception as e:
        report.errors.append(f"Failed to read CSV: {e}")
        return pd.DataFrame(), report

    report.stats["n_rows"] = len(df)
    report.stats["columns"] = list(df.columns)

    if len(df) == 0:
        report.errors.append("Empty DataFrame")
        return df, report

    # Common validations
    if "shape" in df.columns:
        actual_shapes = set(df["shape"].unique())
        report.stats["shapes"] = list(actual_shapes)
        if expected_shapes:
            missing = set(expected_shapes) - actual_shapes
            if missing:
                report.errors.append(f"Missing expected shapes: {missing}")
            report.invariants_checked.append("all_expected_shapes_present")
            if missing:
                report.invariants_failed.append("all_expected_shapes_present")

    if "N" in df.columns:
        actual_N = set(df["N"].unique())
        report.stats["N_values"] = sorted(actual_N)
        if expected_N:
            missing = set(expected_N) - actual_N
            if missing:
                report.errors.append(f"Missing expected N values: {missing}")
            report.invariants_checked.append("all_expected_N_present")
            if missing:
                report.invariants_failed.append("all_expected_N_present")

    # Benchmark-specific validations
    if benchmark == "archetypes":
        report = _validate_archetypes_invariants(
            df, report, expected_coverage_regimes, expected_organization_regimes
        )
    elif benchmark == "calibration":
        report = _validate_calibration_invariants(df, report, expected_null_types)

    # Check for suspicious all-constant columns
    for col in ["pattern_variant", "shape", "N"]:
        if col in df.columns and df[col].nunique() == 1 and len(df) > 10:
            report.warnings.append(
                f"Column '{col}' is constant ({df[col].iloc[0]}) across {len(df)} rows - "
                "expected variation in sweep"
            )

    # Write debug JSON if requested
    if write_debug_json:
        debug_path = path.with_suffix(".validation.json")
        report.to_json(debug_path)

    return df, report


def _validate_archetypes_invariants(
    df: pd.DataFrame,
    report: RunsValidationReport,
    expected_coverage_regimes: Optional[List[str]],
    expected_organization_regimes: Optional[List[str]],
) -> RunsValidationReport:
    """Validate archetype-specific invariants."""

    # Check coverage/organization regimes
    if "coverage_regime" in df.columns:
        actual_cov = set(df["coverage_regime"].unique())
        report.stats["coverage_regimes"] = list(actual_cov)
        if expected_coverage_regimes:
            missing = set(expected_coverage_regimes) - actual_cov
            if missing:
                report.errors.append(f"Missing expected coverage regimes: {missing}")

    if "organization_regime" in df.columns:
        actual_org = set(df["organization_regime"].unique())
        report.stats["organization_regimes"] = list(actual_org)
        if expected_organization_regimes:
            missing = set(expected_organization_regimes) - actual_org
            if missing:
                report.errors.append(f"Missing expected organization regimes: {missing}")

    # CRITICAL: Check pattern_variant matches organization_regime
    if "pattern_variant" in df.columns and "organization_regime" in df.columns:
        report.invariants_checked.append("pattern_matches_organization")

        # For IID rows, pattern_variant should be 'iid' or 'none' or 'uniform'
        iid_rows = df[df["organization_regime"] == "iid"]
        if len(iid_rows) > 0:
            iid_patterns = set(iid_rows["pattern_variant"].unique())
            valid_iid_patterns = {"iid", "none", "uniform"}
            invalid_iid = iid_patterns - valid_iid_patterns
            if invalid_iid:
                report.errors.append(
                    f"IID organization has invalid patterns: {invalid_iid}. "
                    f"Expected one of: {valid_iid_patterns}. "
                    "This indicates pattern_variant is not being set per-condition."
                )
                report.invariants_failed.append("pattern_matches_organization")

        # For structured rows, pattern_variant should NOT be 'iid', 'none', 'uniform'
        struct_rows = df[df["organization_regime"] == "structured"]
        if len(struct_rows) > 0:
            struct_patterns = set(struct_rows["pattern_variant"].unique())
            report.stats["structured_patterns"] = list(struct_patterns)
            invalid_struct = struct_patterns & {"iid", "none", "uniform"}
            if invalid_struct:
                report.warnings.append(
                    f"Structured organization has non-spatial patterns: {invalid_struct}"
                )

    # Check coverage values match regimes
    if "Coverage" in df.columns and "coverage_regime" in df.columns:
        report.invariants_checked.append("coverage_matches_regime")

        high_cov = df[df["coverage_regime"] == "high"]["Coverage"].dropna()
        low_cov = df[df["coverage_regime"] == "low"]["Coverage"].dropna()

        if len(high_cov) > 0:
            report.stats["high_coverage_mean"] = float(high_cov.mean())
            report.stats["high_coverage_range"] = [float(high_cov.min()), float(high_cov.max())]
            if high_cov.mean() < 0.30:
                report.warnings.append(
                    f"High coverage regime has low mean: {high_cov.mean():.2f} (expected >0.30)"
                )

        if len(low_cov) > 0:
            report.stats["low_coverage_mean"] = float(low_cov.mean())
            report.stats["low_coverage_range"] = [float(low_cov.min()), float(low_cov.max())]
            if low_cov.mean() > 0.40:
                report.warnings.append(
                    f"Low coverage regime has high mean: {low_cov.mean():.2f} (expected <0.40)"
                )

    # Check true_archetype distribution
    if "true_archetype" in df.columns:
        arch_counts = df["true_archetype"].value_counts().to_dict()
        report.stats["archetype_distribution"] = arch_counts
        expected_archetypes = {"housekeeping", "regional_program", "sparse_noise", "niche_marker"}
        # Also accept alternate names
        alt_names = {"Ubiquitous", "Gradient", "Basal", "Patchy"}
        actual_archetypes = set(arch_counts.keys())
        if not (actual_archetypes & (expected_archetypes | alt_names)):
            report.warnings.append(f"Unexpected archetype names: {actual_archetypes}")

    return report


def _validate_calibration_invariants(
    df: pd.DataFrame,
    report: RunsValidationReport,
    expected_null_types: Optional[List[str]],
) -> RunsValidationReport:
    """Validate calibration-specific invariants."""

    if "null_type" in df.columns:
        actual_nulls = set(df["null_type"].unique())
        report.stats["null_types"] = list(actual_nulls)
        if expected_null_types:
            missing = set(expected_null_types) - actual_nulls
            if missing:
                report.errors.append(f"Missing expected null types: {missing}")

    # Check p-value distribution under IID null
    if "p_value" in df.columns and "null_type" in df.columns:
        report.invariants_checked.append("iid_pvalues_uniform")

        iid_pvals = df[df["null_type"] == "iid"]["p_value"].dropna()
        if len(iid_pvals) >= 20:
            # Simple uniformity check: expect ~5% below 0.05
            fpr = (iid_pvals < 0.05).mean()
            report.stats["iid_fpr_05"] = float(fpr)

            # Allow 2x tolerance for small samples
            if fpr > 0.15:
                report.warnings.append(
                    f"IID null FPR={fpr:.1%} > 15% (expected ~5%) - possible miscalibration"
                )
            if fpr < 0.01 and len(iid_pvals) > 50:
                report.warnings.append(
                    f"IID null FPR={fpr:.1%} is very low - check if permutations are working"
                )

    # Check abstention rate
    if "abstain_flag" in df.columns:
        abstain_rate = df["abstain_flag"].mean()
        report.stats["abstain_rate"] = float(abstain_rate)
        if abstain_rate > 0.20:
            report.warnings.append(
                f"High abstention rate: {abstain_rate:.1%} - check data generation"
            )

    return report


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

    if df is None:
        raise ValidationError(f"{name} is None")

    if len(df) < min_rows:
        raise ValidationError(
            f"{name} has {len(df)} rows, but {min_rows} required. "
            f"Cannot generate plot with insufficient data."
        )

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValidationError(
            f"{name} missing required columns: {missing}. Available columns: {list(df.columns)}"
        )

    for col in required_columns:
        if df[col].isna().all():
            raise ValidationError(
                f"{name} column '{col}' is all NaN. Cannot plot with no valid data."
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

    n_reps = results_df[replicate_col].max() + 1
    values_by_rep = {}
    max_devs = []

    for rep in range(1, n_reps + 1):
        subset = results_df[results_df[replicate_col] < rep]
        if len(subset) > 0:
            if "p_value" in metric or "pval" in metric.lower():
                est = subset[metric].median()
            else:
                est = subset[metric].mean()
            values_by_rep[rep] = float(est)

    if len(values_by_rep) < 2:
        return {"warning": "Insufficient replicates to assess convergence"}

    final = values_by_rep[max(values_by_rep.keys())]
    if final == 0:
        final = 1e-6

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
