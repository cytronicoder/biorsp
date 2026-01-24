"""Shared utilities for BioRSP CLI scripts."""

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from biorsp.io.manifest import create_manifest, save_manifest
from biorsp.utils.config import BioRSPConfig
from biorsp.utils.constants import (
    B_DEFAULT,
    DELTA_DEG_DEFAULT,
    K_EXPLORATORY_DEFAULT,
)


def add_common_args(parser: argparse.ArgumentParser):
    """Add common BioRSP arguments to an ArgumentParser."""
    parser.add_argument("--adata", type=str, help="Path to AnnData (h5ad) file")
    parser.add_argument("--coords", type=str, help="Path to coords CSV (if not in adata)")
    parser.add_argument("--outdir", type=str, default="results", help="Output directory")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--B", type=int, default=B_DEFAULT, help="Number of angles")
    parser.add_argument(
        "--delta-deg",
        type=float,
        default=DELTA_DEG_DEFAULT,
        help="Sector width in degrees",
    )
    parser.add_argument("--q", type=float, default=0.90, help="Foreground quantile")
    parser.add_argument(
        "--perm-mode",
        type=str,
        default="radial",
        choices=["radial", "joint", "rt_umi", "none"],
        help="Permutation mode",
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=K_EXPLORATORY_DEFAULT,
        help="Number of permutations",
    )
    parser.add_argument(
        "--save-plots", action="store_true", default=True, dest="save_plots", help="Save plots"
    )
    parser.add_argument("--no-plots", action="store_false", dest="save_plots", help="Disable plots")
    parser.add_argument(
        "--save-profiles",
        action="store_true",
        default=True,
        dest="save_profiles",
        help="Save profiles",
    )
    parser.add_argument(
        "--no-profiles", action="store_false", dest="save_profiles", help="Disable profiles"
    )
    parser.add_argument("--features", type=str, help="Comma-separated list of features to analyze")
    parser.add_argument(
        "--features-file", dest="features_file", type=str, help="Path to file with feature names"
    )


def config_from_args(args: argparse.Namespace) -> BioRSPConfig:
    """Construct a BioRSPConfig from parsed CLI arguments."""
    return BioRSPConfig(
        B=args.B,
        delta_deg=args.delta_deg,
        foreground_quantile=args.q,
        perm_mode=args.perm_mode,
        n_permutations=args.n_permutations,
        seed=args.seed,
        save_plots=args.save_plots,
        save_profiles=args.save_profiles,
    )


def ensure_outdir(outdir: Union[str, Path]) -> Path:
    """Ensure the output directory exists and return its Path."""
    path = Path(outdir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_features_to_run(args: argparse.Namespace) -> Optional[List[str]]:
    """Extract feature list from CLI arguments."""
    if args.features:
        return [f.strip() for f in args.features.split(",")]
    if args.features_file:
        with open(args.features_file) as f:
            return [line.strip() for line in f if line.strip()]
    return None


def save_run_manifest(
    outdir: Path,
    config: BioRSPConfig,
    dataset_summary: Dict[str, Any],
    timings: Optional[Dict[str, float]] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
):
    """Create and save a reproducibility manifest."""
    manifest = create_manifest(
        parameters=config.to_dict(),
        seed=config.seed,
        dataset_summary=dataset_summary,
        timings=timings,
        extra_metadata=extra_metadata,
    )
    save_manifest(manifest, str(outdir / "run_metadata.json"))
