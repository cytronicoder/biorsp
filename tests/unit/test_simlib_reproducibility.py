"""Tests for simulation reproducibility and schema validation."""

import sys
import tempfile
from pathlib import Path

import pytest


def _ensure_root_on_path(project_root: Path) -> None:
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


def test_rng_reproducibility(project_root: Path):
    """Test that same seed produces identical RNG streams."""
    _ensure_root_on_path(project_root)
    from biorsp.simulations import rng

    gen1 = rng.make_rng(42, "test", "condition")
    gen2 = rng.make_rng(42, "test", "condition")

    vals1 = gen1.random(100)
    vals2 = gen2.random(100)

    assert (vals1 == vals2).all(), "Same seed should produce identical results"

    gen3 = rng.make_rng(42, "test", "other_condition")
    vals3 = gen3.random(100)

    assert not (vals1 == vals3).all(), "Different tags should produce different results"


def test_simulation_determinism(project_root: Path):
    """Test that running simulation twice with same seed produces identical results."""
    _ensure_root_on_path(project_root)
    from biorsp.simulations import expression, rng, shapes

    def run_single_sim(seed):
        gen = rng.make_rng(seed, "determinism_test")
        coords, _ = shapes.generate_coords("disk", 500, gen)
        libsize = expression.simulate_library_size(500, gen)
        counts, _ = expression.generate_confounded_null(coords, libsize, gen, "iid", {})
        return coords, counts

    coords1, counts1 = run_single_sim(42)
    coords2, counts2 = run_single_sim(42)

    assert (coords1 == coords2).all(), "Coordinates should be identical"
    assert (counts1 == counts2).all(), "Expression should be identical"


def test_schema_version_in_outputs(project_root: Path):
    """Test that schema version is included in outputs."""
    _ensure_root_on_path(project_root)
    import pandas as pd

    from biorsp.simulations import io

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        df = pd.DataFrame(
            {
                "shape": ["disk"],
                "N": [1000],
                "null_type": ["iid"],
                "replicate": [0],
                "seed": [42],
                "p_value": [0.5],
                "Spatial_Bias_Score": [0.1],
                "Coverage": [0.2],
                "abstain_flag": [False],
                "permutation_scheme": ["test"],
            }
        )

        io.write_runs_csv(df, output_dir, benchmark="calibration")

        df_read = pd.read_csv(output_dir / "runs.csv")
        assert "schema_version" in df_read.columns

        assert str(df_read["schema_version"].iloc[0]) == io.SCHEMA_VERSION


def test_manifest_includes_biorsp_config(project_root: Path):
    """Test that manifest includes serialized BioRSPConfig."""
    _ensure_root_on_path(project_root)
    import json
    import tempfile

    from biorsp import BioRSPConfig
    from biorsp.simulations import io

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        config = BioRSPConfig(B=72, delta_deg=60.0, n_permutations=100)

        io.write_manifest(
            output_dir,
            benchmark_name="test",
            params={"seed": 42},
            n_replicates=10,
            runtime_seconds=1.0,
            biorsp_config=config,
        )

        with open(output_dir / "manifest.json") as f:
            manifest = json.load(f)

        assert "biorsp_config" in manifest
        assert manifest["biorsp_config"]["B"] == 72
        assert manifest["biorsp_config"]["delta_deg"] == 60.0


def test_schema_validation_catches_missing_columns(project_root: Path):
    """Test that schema validation raises on missing columns."""
    _ensure_root_on_path(project_root)
    import pandas as pd

    from biorsp.simulations import io

    df = pd.DataFrame({"shape": ["disk"], "N": [1000]})

    with pytest.raises(ValueError, match="Schema validation failed"):
        io.validate_output_schema(df, "calibration", "runs")


def test_condition_key_stability(project_root: Path):
    """Test that condition_key produces stable identifiers."""
    _ensure_root_on_path(project_root)
    from biorsp.simulations import rng

    key1 = rng.condition_key("disk", 1000, "iid")
    key2 = rng.condition_key("disk", 1000, "iid")

    assert key1 == key2, "Same params should produce same key"

    key3 = rng.condition_key("disk", 2000, "iid")
    assert key1 != key3, "Different params should produce different key"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
