#!/usr/bin/env python3
"""Canonical heart case-study runner for BioRSP."""

from __future__ import annotations

import argparse
import importlib.metadata as importlib_metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd

# Force headless plotting backend for deterministic batch runs.
matplotlib.use("Agg")

from biorsp import __version__
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
    p = argparse.ArgumentParser(
        description=(
            "Run the BioRSP heart case study (method validation only; not subtype discovery)."
        )
    )
    p.add_argument("--h5ad", required=True, help="Input .h5ad path.")
    p.add_argument("--out", required=True, help="Output directory.")
    p.add_argument("--donor_key", default=None)
    p.add_argument("--cluster_key", default=None)
    p.add_argument("--celltype_key", default=None)
    p.add_argument("--do_hierarchy", type=_str2bool, default=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bins", type=int, default=72)
    p.add_argument("--n_perm", type=int, default=200)
    p.add_argument("--min_cells_per_cluster", type=int, default=300)
    p.add_argument("--min_cells_per_mega", type=int, default=500)
    p.add_argument("--layer", default=None)
    p.add_argument("--use_raw", action="store_true")
    p.add_argument(
        "--recompute_umap_if_missing",
        type=_str2bool,
        default=False,
        help="Allow UMAP recomputation when X_umap is missing (default false).",
    )
    return p.parse_args()


def _write_run_metadata(outdir: Path, args: argparse.Namespace, status: str) -> None:
    try:
        scanpy_version = importlib_metadata.version("scanpy")
    except Exception:
        scanpy_version = "unavailable"

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "study_type": "heart_case_study_method_validation",
        "biorsp_version": __version__,
        "python_version": platform.python_version(),
        "versions": {
            "scanpy": scanpy_version,
            "anndata": importlib_metadata.version("anndata"),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "parameters": vars(args),
    }
    out = outdir / "run_metadata.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    args = parse_args()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    status = "PASS"
    try:
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
            celltype_key=args.celltype_key,
            cluster_key=args.cluster_key,
            recompute_umap_if_missing=bool(args.recompute_umap_if_missing),
        )
    except Exception:
        status = "FAIL"
        _write_run_metadata(outdir, args, status)
        raise

    _write_run_metadata(outdir, args, status)
    print(f"Heart case study complete. Outputs: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
