"""Comprehensive tests for BioRSP IO module.

Tests cover:
- Smart index detection
- Robust load_umi_counts with Series/ndarray handling
- load_spatial_coords with validation
- align_inputs with mismatch detection
- Enhanced save_results and manifest serialization
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from biorsp.io import (
    align_inputs,
    compute_file_fingerprint,
    create_manifest,
    load_expression_matrix,
    load_spatial_coords,
    load_umi_counts,
    save_results,
)


class TestSmartIndexDetection:
    """Test smart index detection instead of blind index_col=0."""

    def test_load_expression_with_cell_ids(self, tmp_path):
        """Test loading expression matrix with cell IDs as first column."""
        csv_path = tmp_path / "expr_with_ids.csv"

        # Create CSV with cell IDs
        data = pd.DataFrame(
            {
                "cell_id": ["cell_001", "cell_002", "cell_003"],
                "gene_A": [1.5, 2.3, 0.0],
                "gene_B": [0.0, 4.1, 3.2],
            }
        )
        data.to_csv(csv_path, index=False)

        # Load
        df = load_expression_matrix(str(csv_path))

        # Should detect cell_id as index
        assert df.index.name == "cell_id"
        assert list(df.index) == ["cell_001", "cell_002", "cell_003"]
        assert list(df.columns) == ["gene_A", "gene_B"]

    def test_load_expression_without_cell_ids(self, tmp_path):
        """Test loading expression matrix with only numeric columns."""
        csv_path = tmp_path / "expr_no_ids.csv"

        # Create CSV with only numeric data (no IDs)
        data = pd.DataFrame(
            {
                "gene_A": [1.5, 2.3, 0.0],
                "gene_B": [0.0, 4.1, 3.2],
            }
        )
        data.to_csv(csv_path, index=False)

        # Load
        df = load_expression_matrix(str(csv_path))

        # Should use default integer index
        assert list(df.columns) == ["gene_A", "gene_B"]
        assert len(df) == 3


class TestLoadUMICounts:
    """Test robust load_umi_counts with Series/ndarray handling."""

    def test_load_umi_numeric_only_csv(self, tmp_path):
        """Test loading UMI counts from CSV with single numeric column (no IDs)."""
        csv_path = tmp_path / "umi_no_ids.csv"

        # Single numeric column, no IDs
        data = pd.DataFrame({"umi": [100, 200, 300]})
        data.to_csv(csv_path, index=False)

        # Load
        counts = load_umi_counts(str(csv_path))

        # Should return ndarray
        assert isinstance(counts, np.ndarray)
        assert len(counts) == 3
        np.testing.assert_array_equal(counts, [100, 200, 300])

    def test_load_umi_with_cell_id_and_column(self, tmp_path):
        """Test loading UMI counts with cell IDs."""
        csv_path = tmp_path / "umi_with_ids.csv"

        # Cell IDs + UMI column
        data = pd.DataFrame(
            {
                "cell_id": ["cell_A", "cell_B", "cell_C"],
                "umi_counts": [150, 250, 350],
            }
        )
        data.to_csv(csv_path, index=False)

        # Load
        counts = load_umi_counts(str(csv_path))

        # Should return Series with cell IDs as index
        assert isinstance(counts, pd.Series)
        assert list(counts.index) == ["cell_A", "cell_B", "cell_C"]
        assert list(counts.values) == [150, 250, 350]

    def test_load_umi_invalid_numeric(self, tmp_path):
        """Test that non-numeric UMI values raise TypeError."""
        csv_path = tmp_path / "umi_invalid.csv"

        data = pd.DataFrame({"umi": ["100", "abc", "300"]})  # String with non-numeric
        data.to_csv(csv_path, index=False)

        with pytest.raises(TypeError, match="non-numeric"):
            load_umi_counts(str(csv_path))


class TestLoadSpatialCoords:
    """Test load_spatial_coords with validation."""

    def test_load_spatial_with_cell_id_xy(self, tmp_path):
        """Test loading coordinates with cell IDs and x,y columns."""
        csv_path = tmp_path / "coords_with_ids.csv"

        data = pd.DataFrame(
            {
                "cell_id": ["cell_1", "cell_2", "cell_3"],
                "x": [10.5, 20.3, 15.7],
                "y": [5.2, 8.9, 12.1],
            }
        )
        data.to_csv(csv_path, index=False)

        coords_df = load_spatial_coords(str(csv_path))

        # Should detect cell_id as index
        assert list(coords_df.index) == ["cell_1", "cell_2", "cell_3"]
        assert list(coords_df.columns) == ["x", "y"]
        assert coords_df.shape == (3, 2)

    def test_load_spatial_xy_only(self, tmp_path):
        """Test loading coordinates with only x,y columns (no IDs)."""
        csv_path = tmp_path / "coords_no_ids.csv"

        data = pd.DataFrame(
            {
                "x": [10.5, 20.3, 15.7],
                "y": [5.2, 8.9, 12.1],
            }
        )
        data.to_csv(csv_path, index=False)

        coords_df = load_spatial_coords(str(csv_path))

        assert list(coords_df.columns) == ["x", "y"]
        assert coords_df.shape == (3, 2)

    def test_load_spatial_case_insensitive(self, tmp_path):
        """Test that X,Y columns are detected (case-insensitive)."""
        csv_path = tmp_path / "coords_uppercase.csv"

        data = pd.DataFrame(
            {
                "X": [10.5, 20.3],
                "Y": [5.2, 8.9],
            }
        )
        data.to_csv(csv_path, index=False)

        coords_df = load_spatial_coords(str(csv_path))

        # Should standardize to lowercase
        assert list(coords_df.columns) == ["x", "y"]

    def test_load_spatial_with_nan_raises(self, tmp_path):
        """Test that NaN coordinates raise ValueError."""
        csv_path = tmp_path / "coords_nan.csv"

        data = pd.DataFrame(
            {
                "x": [10.5, np.nan, 15.7],
                "y": [5.2, 8.9, 12.1],
            }
        )
        data.to_csv(csv_path, index=False)

        with pytest.raises(ValueError, match="NaN"):
            load_spatial_coords(str(csv_path))


class TestAlignInputs:
    """Test align_inputs for preventing silent mismatches."""

    def test_align_order_mismatch(self):
        """Test that align_inputs corrects order mismatches."""
        # Create expression with order: A, B, C
        expr = pd.DataFrame(
            {"gene1": [1, 2, 3], "gene2": [4, 5, 6]}, index=["cell_A", "cell_B", "cell_C"]
        )

        # Create coords with different order: C, A, B
        coords = pd.DataFrame(
            {"x": [30, 10, 20], "y": [3, 1, 2]}, index=["cell_C", "cell_A", "cell_B"]
        )

        # Align
        aligned_expr, aligned_coords, _, report = align_inputs(
            expr, coords, how="inner", verbose=False
        )

        # Should have same order now
        assert list(aligned_expr.index) == list(aligned_coords.index)
        assert aligned_expr.index.equals(aligned_coords.index)

        # Should have all 3 cells
        assert len(aligned_expr) == 3
        assert report["n_overlap"] == 3

    def test_align_low_overlap_raises(self):
        """Test that low overlap raises ValueError."""
        expr = pd.DataFrame({"gene1": [1, 2]}, index=["cell_A", "cell_B"])

        coords = pd.DataFrame(
            {"x": [30, 40], "y": [3, 4]},
            index=["cell_X", "cell_Y"],  # No overlap
        )

        with pytest.raises(ValueError, match="Insufficient overlap"):
            align_inputs(expr, coords, min_overlap=0.5, verbose=False)

    def test_align_with_umi_series(self):
        """Test alignment with UMI as Series."""
        expr = pd.DataFrame({"gene1": [1, 2, 3]}, index=["cell_A", "cell_B", "cell_C"])

        coords = pd.DataFrame(
            {"x": [10, 20, 30], "y": [1, 2, 3]}, index=["cell_A", "cell_B", "cell_C"]
        )

        umi = pd.Series([100, 200, 300], index=["cell_A", "cell_B", "cell_C"])

        aligned_expr, aligned_coords, aligned_umi, report = align_inputs(
            expr, coords, umi, verbose=False
        )

        assert isinstance(aligned_umi, pd.Series)
        assert aligned_umi.index.equals(aligned_expr.index)
        assert report["n_umi"] == 3

    def test_align_umi_ndarray_length_mismatch_raises(self):
        """Test that UMI ndarray with wrong length raises TypeError."""
        expr = pd.DataFrame({"gene1": [1, 2, 3]}, index=["cell_A", "cell_B", "cell_C"])

        coords = pd.DataFrame(
            {"x": [10, 20, 30], "y": [1, 2, 3]}, index=["cell_A", "cell_B", "cell_C"]
        )

        umi = np.array([100, 200])  # Wrong length

        with pytest.raises(TypeError, match="length"):
            align_inputs(expr, coords, umi, verbose=False)


class TestSaveResults:
    """Test enhanced save_results."""

    def test_save_results_creates_directories(self, tmp_path):
        """Test that save_results creates parent directories."""
        nested_path = tmp_path / "subdir1" / "subdir2" / "results.json"

        results = {"key": "value"}
        save_results(results, str(nested_path))

        assert nested_path.exists()
        with open(nested_path) as f:
            loaded = json.load(f)
        assert loaded["key"] == "value"

    def test_save_results_handles_paths(self, tmp_path):
        """Test that Path objects are serialized."""
        output = tmp_path / "results.json"

        results = {"input_file": Path("/some/path/file.csv"), "value": 42}
        save_results(results, str(output))

        with open(output) as f:
            loaded = json.load(f)
        assert loaded["input_file"] == "/some/path/file.csv"

    def test_save_results_handles_pandas(self, tmp_path):
        """Test that pandas Series and DataFrame are serialized."""
        output = tmp_path / "results.json"

        series = pd.Series([1, 2, 3], index=["a", "b", "c"])
        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})

        results = {
            "series": series,
            "dataframe": df,
        }
        save_results(results, str(output))

        with open(output) as f:
            loaded = json.load(f)
        assert "series" in loaded
        assert "dataframe" in loaded

    def test_save_results_handles_dataclasses(self, tmp_path):
        """Test that dataclasses are serialized."""
        output = tmp_path / "results.json"

        @dataclass
        class Config:
            param1: int
            param2: str

        config = Config(param1=42, param2="test")
        results = {"config": config}

        save_results(results, str(output))

        with open(output) as f:
            loaded = json.load(f)
        assert loaded["config"]["param1"] == 42


class TestManifest:
    """Test enhanced manifest functionality."""

    def test_manifest_serialization(self, tmp_path):
        """Test that manifest serialization handles paths, pandas, numpy."""
        from biorsp.io.manifest import save_manifest

        manifest = create_manifest(
            parameters={"B": 16, "delta_deg": 45.0, "input": Path("/data/file.csv")},
            seed=42,
            dataset_summary={"n_cells": 1000, "n_genes": 500},
        )

        output = tmp_path / "manifest.json"
        save_manifest(manifest, str(output))

        assert output.exists()
        with open(output) as f:
            data = json.load(f)

        assert data["schema_version"] == "2.0"
        assert data["random_seed"] == 42
        assert "git_dirty" in data["software_versions"]
        assert isinstance(data["parameters"]["input"], str)  # Path converted

    def test_compute_file_fingerprint_fast(self, tmp_path):
        """Test fast file fingerprinting (size + mtime)."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        fingerprint = compute_file_fingerprint(test_file, mode="fast")

        assert fingerprint["exists"] is True
        assert fingerprint["size_bytes"] == 12
        assert "mtime" in fingerprint
        assert "sha256" not in fingerprint

    def test_compute_file_fingerprint_strict(self, tmp_path):
        """Test strict file fingerprinting (sha256)."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        fingerprint = compute_file_fingerprint(test_file, mode="strict")

        assert fingerprint["exists"] is True
        assert "sha256" in fingerprint
        assert len(fingerprint["sha256"]) == 64  # SHA256 hex digest


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
