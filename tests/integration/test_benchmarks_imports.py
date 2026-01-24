import sys
from pathlib import Path


def _ensure_root_on_path(project_root: Path) -> None:
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


def test_simlib_imports(project_root: Path):
    """Test importing simlib main package."""
    _ensure_root_on_path(project_root)
    import biorsp.simulations as simlib

    assert simlib.__version__
    assert hasattr(simlib, "GeometryCache")
    assert hasattr(simlib, "make_gene_panel")


def test_simlib_submodules(project_root: Path):
    """Test importing simlib submodules."""
    _ensure_root_on_path(project_root)
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


def test_benchmarks_scripts_import(project_root: Path):
    """Test importing benchmarks scripts."""
    _ensure_root_on_path(project_root)

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
