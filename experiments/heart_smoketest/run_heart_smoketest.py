#!/usr/bin/env python3
"""CI sanity wrapper for the heart case-study pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
import matplotlib

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Force headless plotting backend for CI jobs.
matplotlib.use("Agg")

from biorsp.pipeline import run_case_study


def _str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    x = str(value).strip().lower()
    if x in {"1", "true", "t", "yes", "y"}:
        return True
    if x in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run CI sanity check for heart case study.")
    p.add_argument("--h5ad", required=True, help="Input .h5ad path.")
    p.add_argument(
        "--out",
        default="experiments/heart_smoketest/outputs/heart_ci_sanity",
        help="Output directory.",
    )
    p.add_argument("--do_hierarchy", type=_str2bool, default=False)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bins", type=int, default=48)
    p.add_argument("--n_perm", type=int, default=20)
    p.add_argument("--layer", default=None)
    p.add_argument("--use_raw", action="store_true")
    p.add_argument("--donor_key", default="hubmap_id")
    p.add_argument("--cluster_key", default="azimuth_id")
    p.add_argument("--celltype_key", default="azimuth_label")
    p.add_argument("--min_cells_per_cluster", type=int, default=300)
    p.add_argument("--min_cells_per_mega", type=int, default=500)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(args.h5ad)
    run_case_study(
        adata=adata,
        outdir=outdir,
        do_hierarchy=bool(args.do_hierarchy),
        layer=args.layer,
        use_raw=bool(args.use_raw),
        bins=int(args.bins),
        n_perm=int(args.n_perm),
        seed=int(args.seed),
        min_cells_per_cluster=int(args.min_cells_per_cluster),
        min_cells_per_mega=int(args.min_cells_per_mega),
        donor_key=args.donor_key,
        cluster_key=args.cluster_key,
        celltype_key=args.celltype_key,
    )
    print(f"Heart smoke sanity complete. Outputs: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
