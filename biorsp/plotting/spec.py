"""
Plot Specification and Standardization for BioRSP.

This module defines the canonical plot specification used across all simulation
benchmarks and kidney case studies. It ensures consistent semantics, colors,
cutoffs, and archetype definitions.

Key responsibilities:
- Define canonical column names for outputs
- Standardize archetype classification logic
- Provide consistent color mappings
- Define default cutoffs
- Enable consistent legend ordering

Usage:
    >>> from biorsp.plotting.spec import PlotSpec
    >>> spec = PlotSpec(c_cut=0.30, s_cut=0.15)
    >>> archetype = spec.classify(coverage=0.45, spatial_bias_score=0.08)
    >>> color = spec.get_color(archetype)
    >>> print(f"{archetype}: {color}")
    Ubiquitous: #4CAF50
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Canonical archetype color mapping (Material Design palette)
# These colors MUST match across all plots to avoid confusion
ARCHETYPE_COLORS = {
    "Ubiquitous": "#4CAF50",  # Green - high C, low S (widespread, no bias)
    "Gradient": "#2196F3",  # Blue - high C, high S (broad spatial domain)
    "Basal": "#9E9E9E",  # Gray - low C, low S (scattered/rare, no pattern)
    "Patchy": "#FF5722",  # Red-Orange - low C, high S (localized marker)
    # Special cases
    "Unknown": "#BDBDBD",  # Light gray for unclassified
    "Abstention": "#000000",  # Black for abstained scores
    "abstention_stress": "#000000",  # Black for simulation stress-test genes
}

# Reader-friendly descriptions for each archetype
ARCHETYPE_DESCRIPTIONS = {
    "Ubiquitous": "Widespread expression, no spatial bias\n(High Coverage, Low Spatial Bias Score)",
    "Gradient": "Broad spatial domain or gradient\n(High Coverage, High Spatial Bias Score)",
    "Basal": "Sparse/scattered expression\n(Low Coverage, Low Spatial Bias Score)",
    "Patchy": "Localized spatial marker\n(Low Coverage, High Spatial Bias Score)",
}

# Canonical legend order (for consistent subplot layouts)
ARCHETYPE_ORDER = ["Ubiquitous", "Gradient", "Patchy", "Basal"]

# Quadrant annotation positions (for scatter plots)
# Format: (rel_x, rel_y, label, alignment)
QUADRANT_ANNOTATIONS = [
    (0.25, 0.25, "Basal", ("center", "center")),  # Bottom-left
    (0.75, 0.25, "Ubiquitous", ("center", "center")),  # Bottom-right
    (0.25, 0.75, "Patchy", ("center", "center")),  # Top-left
    (0.75, 0.75, "Gradient", ("center", "center")),  # Top-right
]


@dataclass
class PlotSpec:
    """
    Specification for standardized BioRSP plots.

    This class defines the canonical thresholds, column names, and classification
    logic used across all plotting functions. It ensures that:
    1. Quadrant cutoff lines match the classification logic
    2. Colors are assigned consistently
    3. Column names are standardized

    Parameters
    ----------
    c_cut : float
        Coverage threshold separating high/low (default: 0.30)
    s_cut : float
        Spatial score threshold separating high/low (default: 0.15)
    coverage_col : str
        Column name for coverage in DataFrames (default: "Coverage")
    spatial_col : str
        Column name for spatial bias score (default: "Spatial_Bias_Score")
    archetype_col : str
        Column name for archetype labels (default: "Archetype")
    min_expr_cells : int
        Minimum expressing cells for valid classification (default: 10)
    """

    c_cut: float = 0.30
    s_cut: float = 0.15
    coverage_col: str = "Coverage"
    spatial_col: str = "Spatial_Bias_Score"
    archetype_col: str = "Archetype"
    min_expr_cells: int = 10
    # Color and description mappings (immutable)
    colors: Dict[str, str] = field(default_factory=lambda: ARCHETYPE_COLORS.copy())
    descriptions: Dict[str, str] = field(default_factory=lambda: ARCHETYPE_DESCRIPTIONS.copy())
    order: List[str] = field(default_factory=lambda: ARCHETYPE_ORDER.copy())

    def classify(
        self,
        coverage: float,
        spatial_bias_score: float,
        n_expr_cells: Optional[int] = None,
        abstain_flag: bool = False,
    ) -> str:
        """
        Classify a gene/replicate into an archetype based on C and S.

        Parameters
        ----------
        coverage : float
            Coverage score (fraction of cells above threshold)
        spatial_bias_score : float
            Spatial organization score (weighted RMS)
        n_expr_cells : int, optional
            Number of expressing cells (for abstention check)
        abstain_flag : bool
            Whether the scoring algorithm abstained

        Returns
        -------
        archetype : str
            One of: "Ubiquitous", "Gradient", "Patchy", "Basal", "Abstention"
        """
        if abstain_flag:
            return "Abstention"

        if n_expr_cells is not None and n_expr_cells < self.min_expr_cells:
            return "Abstention"

        # Handle NaN values
        if np.isnan(coverage) or np.isnan(spatial_bias_score):
            return "Abstention"

        # Quadrant classification (deterministic)
        high_c = coverage >= self.c_cut
        high_s = spatial_bias_score >= self.s_cut

        if high_c and not high_s:
            return "Ubiquitous"
        elif high_c and high_s:
            return "Gradient"
        elif not high_c and high_s:
            return "Patchy"
        else:  # low C, low S
            return "Basal"

    def classify_dataframe(
        self,
        df: pd.DataFrame,
        inplace: bool = False,
    ) -> pd.DataFrame:
        """
        Classify all rows in a DataFrame and add Archetype column.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with Coverage and Spatial_Bias_Score columns
        inplace : bool
            If True, modify df in place; otherwise return copy

        Returns
        -------
        df : pd.DataFrame
            DataFrame with added/updated Archetype column
        """
        if not inplace:
            df = df.copy()

        # Check for required columns
        if self.coverage_col not in df.columns:
            raise ValueError(f"Missing required column: {self.coverage_col}")
        if self.spatial_col not in df.columns:
            raise ValueError(f"Missing required column: {self.spatial_col}")

        # Apply classification
        n_expr_col = "n_expr_cells" if "n_expr_cells" in df.columns else None
        abstain_col = "abstain_flag" if "abstain_flag" in df.columns else None

        def classify_row(row):
            return self.classify(
                coverage=row[self.coverage_col],
                spatial_bias_score=row[self.spatial_col],
                n_expr_cells=row[n_expr_col] if n_expr_col else None,
                abstain_flag=row[abstain_col] if abstain_col else False,
            )

        df[self.archetype_col] = df.apply(classify_row, axis=1)
        return df

    def get_color(self, archetype: str, default: str = "#BDBDBD") -> str:
        """
        Get color for an archetype label.

        Parameters
        ----------
        archetype : str
            Archetype name
        default : str
            Default color hex code for unknown archetypes

        Returns
        -------
        color : str
            Hex color code
        """
        return self.colors.get(archetype, default)

    def get_description(self, archetype: str, default: str = "Unknown archetype") -> str:
        """
        Get plain-language description for an archetype.

        Parameters
        ----------
        archetype : str
            Archetype name
        default : str
            Default description for unknown archetypes

        Returns
        -------
        description : str
            Multi-line description text
        """
        return self.descriptions.get(archetype, default)

    def get_quadrant_bounds(self) -> Tuple[float, float]:
        """
        Get the (c_cut, s_cut) thresholds as a tuple.

        Returns
        -------
        bounds : tuple
            (coverage_threshold, spatial_threshold)
        """
        return (self.c_cut, self.s_cut)

    def get_legend_order(self) -> List[str]:
        """
        Get the canonical ordering for legend entries.

        Returns
        -------
        order : list
            List of archetype names in display order
        """
        return self.order.copy()

    def validate_dataframe(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Validate that a DataFrame has required columns and sane values.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to validate

        Returns
        -------
        report : dict
            Validation report with status and issues
        """
        issues = []
        warnings = []

        # Check required columns
        required = [self.coverage_col, self.spatial_col]
        issues.extend(
            f"Missing required column: {col}" for col in required if col not in df.columns
        )

        if issues:
            return {"status": "FAIL", "issues": issues, "warnings": warnings}

        # Check value ranges
        c_vals = df[self.coverage_col]
        s_vals = df[self.spatial_col]

        if (c_vals < 0).any() or (c_vals > 1).any():
            warnings.append(f"{self.coverage_col} values outside [0, 1]")

        if (s_vals < 0).any():
            warnings.append(f"{self.spatial_col} has negative values")

        # Check for excessive NaNs
        c_nan_frac = c_vals.isna().sum() / len(c_vals)
        s_nan_frac = s_vals.isna().sum() / len(s_vals)
        if c_nan_frac > 0.5:
            warnings.append(f"{self.coverage_col} is >50% NaN")
        if s_nan_frac > 0.5:
            warnings.append(f"{self.spatial_col} is >50% NaN")

        status = "PASS" if not issues else "FAIL"
        if warnings and status == "PASS":
            status = "WARNING"

        return {"status": status, "issues": issues, "warnings": warnings}

    def to_dict(self) -> Dict:
        """Export spec as a dictionary (for manifest files)."""
        return {
            "c_cut": self.c_cut,
            "s_cut": self.s_cut,
            "coverage_col": self.coverage_col,
            "spatial_col": self.spatial_col,
            "archetype_col": self.archetype_col,
            "min_expr_cells": self.min_expr_cells,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "PlotSpec":
        """Create PlotSpec from a dictionary."""
        return cls(
            c_cut=d.get("c_cut", 0.30),
            s_cut=d.get("s_cut", 0.15),
            coverage_col=d.get("coverage_col", "Coverage"),
            spatial_col=d.get("spatial_col", "Spatial_Bias_Score"),
            archetype_col=d.get("archetype_col", "Archetype"),
            min_expr_cells=d.get("min_expr_cells", 10),
        )


def load_spec_from_manifest(manifest_path: str) -> PlotSpec:
    """
    Load PlotSpec from a manifest.json file.

    Parameters
    ----------
    manifest_path : str
        Path to manifest.json

    Returns
    -------
    spec : PlotSpec
        PlotSpec with parameters from manifest, or defaults if not found
    """
    import json
    from pathlib import Path

    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path) as f:
        manifest = json.load(f)

    spec_dict = manifest.get("plot_spec", {})
    return PlotSpec.from_dict(spec_dict)
