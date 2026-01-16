import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_simlib_imports():
    """Test importing simlib main package."""
    import biorsp.simulations as simlib

    assert simlib.__version__
    assert hasattr(simlib, "GeometryCache")
    assert hasattr(simlib, "make_gene_panel")


def test_simlib_submodules():
    """Test importing simlib submodules."""
    from biorsp.simulations import (
        cache,
        datasets,
        density,
        scoring,
    )

    assert cache.GeometryCache
    assert datasets.package_as_anndata
    assert density.kde_density
    assert scoring.score_dataset


def test_benchmarks_scripts_import():
    """Test importing benchmarks scripts."""

    from analysis.benchmarks.runners import (
        run_archetypes,
        run_calibration,
        run_genegene,
        run_robustness,
    )

    assert hasattr(run_calibration, "main")
    assert hasattr(run_archetypes, "main")
    assert hasattr(run_genegene, "main")
    assert hasattr(run_robustness, "main")
