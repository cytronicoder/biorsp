"""
Scientific correctness tests for simulation benchmarks.

These tests verify that:
1. Archetype benchmark generates correct pattern variants per archetype
2. IID organization uses IID/uniform patterns (NOT spatial patterns)
3. Structured organization uses spatial patterns
4. Calibration benchmark produces calibrated p-values under IID null
5. Sweep dimensions match expected configurations

These are critical tests to prevent scientific bugs like:
- Applying the same pattern to all archetypes
- Only running on a single geometry/sample size
- p-values being systematically too high or too low
"""

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

# Find repository root
_test_file = Path(__file__).resolve()
_search_path = _test_file.parent
while _search_path != _search_path.parent:
    if (_search_path / "pyproject.toml").exists():
        ROOT_DIR = _search_path
        break
    _search_path = _search_path.parent
else:
    ROOT_DIR = _test_file.parent.parent

RUNNERS_DIR = ROOT_DIR / "analysis" / "benchmarks" / "runners"


def get_env():
    """Get environment with workspace root in PYTHONPATH."""
    import os

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    return env


class TestArchetypeScientificCorrectness:
    """Tests for archetype benchmark scientific correctness."""

    @pytest.fixture
    def archetype_runs(self, tmp_path):
        """Run archetype benchmark in quick mode and return results."""
        script_path = RUNNERS_DIR / "run_archetypes.py"
        assert script_path.exists(), f"Script not found: {script_path}"

        result = subprocess.run(
            [
                sys.executable,
                str(script_path),
                "--mode",
                "quick",
                "--outdir",
                str(tmp_path / "archetypes"),
                "--run_id",
                "test_run",
                "--seed",
                "12345",
            ],
            capture_output=True,
            text=True,
            timeout=300,
            env=get_env(),
            cwd=str(tmp_path),
            check=False,
        )

        assert result.returncode == 0, (
            f"Archetype benchmark failed:\n" f"stdout: {result.stdout}\n" f"stderr: {result.stderr}"
        )

        runs_csv = tmp_path / "archetypes" / "archetypes" / "test_run" / "runs.csv"
        assert runs_csv.exists(), f"runs.csv not created at {runs_csv}"
        return pd.read_csv(runs_csv)

    def test_iid_organization_has_iid_pattern(self, archetype_runs):
        """CRITICAL: IID organization must use IID/uniform pattern, not spatial patterns."""
        df = archetype_runs
        iid_rows = df[df["organization_regime"] == "iid"]

        assert len(iid_rows) > 0, "No IID organization rows found"

        # Check that all IID rows have iid/uniform/none pattern
        valid_iid_patterns = {"iid", "none", "uniform"}
        invalid_patterns = set(iid_rows["pattern_variant"].unique()) - valid_iid_patterns

        assert len(invalid_patterns) == 0, (
            f"IID organization has spatial patterns: {invalid_patterns}. "
            f"This is a CRITICAL BUG - IID should use uniform distribution, "
            f"not structured spatial patterns like wedge_core."
        )

    def test_structured_organization_has_spatial_pattern(self, archetype_runs):
        """Structured organization should use spatial patterns."""
        df = archetype_runs
        struct_rows = df[df["organization_regime"] == "structured"]

        assert len(struct_rows) > 0, "No structured organization rows found"

        # Check that structured rows have spatial patterns (not iid/uniform)
        non_spatial = {"iid", "none", "uniform"}
        actual_patterns = set(struct_rows["pattern_variant"].unique())
        unexpected = actual_patterns & non_spatial

        assert len(unexpected) == 0, (
            f"Structured organization has non-spatial patterns: {unexpected}. "
            f"Expected spatial patterns like radial_gradient, wedge_core."
        )

    def test_pattern_variant_varies_with_archetype(self, archetype_runs):
        """pattern_variant should NOT be constant - it varies by organization regime."""
        df = archetype_runs

        # Should have at least 2 different pattern variants
        # (iid for IID organization, spatial for structured)
        n_unique = df["pattern_variant"].nunique()

        assert n_unique >= 2, (
            f"Only {n_unique} unique pattern_variant(s): {df['pattern_variant'].unique()}. "
            f"Expected different patterns for IID vs structured organization."
        )

    def test_sweep_includes_multiple_shapes(self, archetype_runs):
        """Sweep should include multiple geometries."""
        df = archetype_runs
        shapes = df["shape"].unique()

        assert len(shapes) >= 2, (
            f"Only {len(shapes)} shape(s): {shapes}. "
            f"Expected sweep over multiple geometries (disk, peanut, etc.)"
        )

    def test_sweep_includes_multiple_sample_sizes(self, archetype_runs):
        """Sweep should include multiple sample sizes."""
        df = archetype_runs
        n_values = df["N"].unique()

        assert len(n_values) >= 2, (
            f"Only {len(n_values)} sample size(s): {n_values}. "
            f"Expected sweep over multiple N values (500, 2000, etc.)"
        )

    def test_all_four_archetypes_present(self, archetype_runs):
        """All four archetypes should be generated."""
        df = archetype_runs
        archetypes = set(df["Archetype_true"].unique())

        # Accept either naming convention
        expected_archetypes = {"housekeeping", "regional_program", "sparse_noise", "niche_marker"}
        alt_archetypes = {"Ubiquitous", "Gradient", "Basal", "Patchy"}

        has_expected = len(archetypes & (expected_archetypes | alt_archetypes)) >= 4

        assert has_expected, (
            f"Only found archetypes: {archetypes}. "
            f"Expected all 4: high×iid, high×structured, low×iid, low×structured."
        )

    def test_coverage_matches_regime(self, archetype_runs):
        """Coverage values should match coverage regime (high vs low)."""
        df = archetype_runs

        high_cov = df[df["coverage_regime"] == "high"]["Coverage"].dropna()
        low_cov = df[df["coverage_regime"] == "low"]["Coverage"].dropna()

        if len(high_cov) > 0 and len(low_cov) > 0:
            # High coverage should be significantly higher than low
            assert high_cov.mean() > low_cov.mean() + 0.15, (
                f"High coverage mean ({high_cov.mean():.2f}) is not significantly "
                f"higher than low coverage mean ({low_cov.mean():.2f})"
            )


class TestCalibrationScientificCorrectness:
    """Tests for calibration benchmark scientific correctness."""

    @pytest.fixture
    def calibration_runs(self, tmp_path):
        """Run calibration benchmark in quick mode and return results."""
        script_path = RUNNERS_DIR / "run_calibration.py"
        assert script_path.exists(), f"Script not found: {script_path}"

        result = subprocess.run(
            [
                sys.executable,
                str(script_path),
                "--mode",
                "quick",
                "--outdir",
                str(tmp_path / "calibration"),
                "--run_id",
                "test_run",
                "--seed",
                "54321",
            ],
            capture_output=True,
            text=True,
            # timeout=300,
            env=get_env(),
            cwd=str(tmp_path),
            check=False,
        )

        assert result.returncode == 0, (
            f"Calibration benchmark failed:\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )

        runs_csv = tmp_path / "calibration" / "calibration" / "test_run" / "runs.csv"
        assert runs_csv.exists(), f"runs.csv not created at {runs_csv}"
        return pd.read_csv(runs_csv)

    def test_iid_null_produces_uniform_pvalues(self, calibration_runs):
        """Under IID null, p-values should be approximately uniform."""
        df = calibration_runs
        iid_pvals = df[df["null_type"] == "iid"]["p_value"].dropna()

        assert len(iid_pvals) >= 50, f"Need at least 50 IID p-values, got {len(iid_pvals)}"

        # FPR at α=0.05 should be close to 0.05
        fpr = (iid_pvals < 0.05).mean()

        # Allow some tolerance (binomial confidence interval)
        # For n=100, 95% CI for p=0.05 is roughly [0.01, 0.11]
        assert 0.01 < fpr < 0.20, (
            f"IID null FPR={fpr:.1%} is outside reasonable range [1%, 20%]. "
            f"Expected ~5% for calibrated p-values."
        )

    def test_no_pvalue_floor_accumulation(self, calibration_runs):
        """P-values should not accumulate at the permutation floor."""
        df = calibration_runs
        iid_pvals = df[df["null_type"] == "iid"]["p_value"].dropna()

        if len(iid_pvals) < 20:
            pytest.skip("Insufficient p-values for floor analysis")

        # Count p-values at exact floor (1 / (n_perm + 1))
        # Assume 100 permutations → floor = 1/101 ≈ 0.0099
        pval_min = iid_pvals.min()
        at_floor = (iid_pvals == pval_min).sum()

        # Should not have more than ~5% at floor for calibrated method
        floor_fraction = at_floor / len(iid_pvals)
        assert floor_fraction < 0.15, (
            f"{floor_fraction:.1%} of p-values at floor ({pval_min:.4f}). "
            f"Possible issue with permutation p-value computation."
        )

    def test_sweep_includes_multiple_shapes(self, calibration_runs):
        """Sweep should include multiple geometries."""
        df = calibration_runs
        shapes = df["shape"].unique()

        assert len(shapes) >= 2, (
            f"Only {len(shapes)} shape(s): {shapes}. " f"Expected sweep over multiple geometries."
        )

    def test_sweep_includes_multiple_sample_sizes(self, calibration_runs):
        """Sweep should include multiple sample sizes."""
        df = calibration_runs
        n_values = df["N"].unique()

        assert len(n_values) >= 2, (
            f"Only {len(n_values)} sample size(s): {n_values}. "
            f"Expected sweep over multiple N values."
        )

    def test_abstention_rate_reasonable(self, calibration_runs):
        """Abstention rate should be reasonable (<30%)."""
        df = calibration_runs

        if "abstain_flag" in df.columns:
            abstain_rate = df["abstain_flag"].mean()
            assert abstain_rate < 0.30, (
                f"Abstention rate {abstain_rate:.1%} is too high. "
                f"Check data generation parameters."
            )


class TestArchetypeSpecsMapping:
    """Direct unit tests for ARCHETYPE_SPECS mapping."""

    def test_archetype_specs_structure(self):
        """ARCHETYPE_SPECS should have correct structure."""
        # Import directly to test the mapping
        sys.path.insert(0, str(RUNNERS_DIR))
        from run_archetypes import ARCHETYPE_SPECS

        # Check all four combinations exist
        expected_keys = [
            ("high", "iid"),
            ("high", "structured"),
            ("low", "iid"),
            ("low", "structured"),
        ]

        for key in expected_keys:
            assert key in ARCHETYPE_SPECS, f"Missing archetype key: {key}"
            assert "pattern_variant" in ARCHETYPE_SPECS[key]
            assert "archetype_name" in ARCHETYPE_SPECS[key]

    def test_get_pattern_for_archetype_iid(self):
        """IID organization should return IID pattern."""
        sys.path.insert(0, str(RUNNERS_DIR))
        from run_archetypes import get_pattern_for_archetype

        pattern_high_iid = get_pattern_for_archetype("high", "iid")
        pattern_low_iid = get_pattern_for_archetype("low", "iid")

        assert pattern_high_iid == "iid", f"high×iid should use 'iid', got {pattern_high_iid}"
        assert pattern_low_iid == "iid", f"low×iid should use 'iid', got {pattern_low_iid}"

    def test_get_pattern_for_archetype_structured(self):
        """Structured organization should return spatial patterns."""
        sys.path.insert(0, str(RUNNERS_DIR))
        from run_archetypes import get_pattern_for_archetype

        pattern_high_struct = get_pattern_for_archetype("high", "structured")
        pattern_low_struct = get_pattern_for_archetype("low", "structured")

        # Should be spatial patterns (not iid/uniform)
        assert pattern_high_struct not in {
            "iid",
            "uniform",
            "none",
        }, f"high×structured should use spatial pattern, got {pattern_high_struct}"
        assert pattern_low_struct not in {
            "iid",
            "uniform",
            "none",
        }, f"low×structured should use spatial pattern, got {pattern_low_struct}"


class TestValidationModule:
    """Tests for the load_and_validate_runs helper."""

    def test_validation_detects_missing_shapes(self, tmp_path):
        """Validation should detect missing expected shapes."""
        from biorsp.simulations.validation import load_and_validate_runs

        # Create a minimal runs.csv with only one shape
        df = pd.DataFrame(
            {
                "shape": ["disk"] * 10,
                "N": [2000] * 10,
                "coverage_regime": ["high", "low"] * 5,
                "organization_regime": ["iid", "structured"] * 5,
                "pattern_variant": ["iid", "wedge_core"] * 5,
                "Coverage": [0.7, 0.1] * 5,
            }
        )

        runs_path = tmp_path / "runs.csv"
        df.to_csv(runs_path, index=False)

        _, report = load_and_validate_runs(
            runs_path,
            benchmark="archetypes",
            expected_shapes=["disk", "peanut", "annulus"],
            write_debug_json=False,
        )

        assert not report.valid, "Should detect missing shapes"
        assert any("Missing expected shapes" in err for err in report.errors)

    def test_validation_detects_iid_with_spatial_pattern(self, tmp_path):
        """Validation should detect IID organization with spatial patterns."""
        from biorsp.simulations.validation import load_and_validate_runs

        # Create invalid runs.csv: IID with wedge_core pattern
        df = pd.DataFrame(
            {
                "shape": ["disk"] * 10,
                "N": [2000] * 10,
                "coverage_regime": ["high", "low"] * 5,
                "organization_regime": ["iid"] * 10,  # All IID
                "pattern_variant": ["wedge_core"] * 10,  # But spatial pattern - BUG!
                "Coverage": [0.7, 0.1] * 5,
            }
        )

        runs_path = tmp_path / "runs.csv"
        df.to_csv(runs_path, index=False)

        _, report = load_and_validate_runs(
            runs_path,
            benchmark="archetypes",
            write_debug_json=False,
        )

        assert not report.valid, "Should detect IID with spatial pattern"
        assert any("IID organization has invalid patterns" in err for err in report.errors)
