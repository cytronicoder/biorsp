"""Command-line interfaces for BioRSP smoke tests."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import scanpy as sc

from biorsp.geometry import compute_angles, compute_vantage
from biorsp.moran import extract_weights, morans_i
from biorsp.permutation import perm_null_emax, plot_null_distribution
from biorsp.rsp import compute_rsp_profile, plot_rsp_polar
from biorsp.utils import ensure_dir, get_gene_vector, select_gene


def _get_gene_display_name(adata, gene: str) -> str:
    """Get Hugo symbol if available, otherwise return the gene name as-is.

    Args:
        adata: AnnData object.
        gene: Gene name to extract.

    Returns:
        Display name (Hugo symbol or gene name).
    """
    # If gene has var metadata with 'hugo_symbol', 'gene_name', or 'gene_symbol', prefer that
    if hasattr(adata, "var") and gene in adata.var_names:
        idx = list(adata.var_names).index(gene)
        if "hugo_symbol" in adata.var.columns:
            return str(adata.var.iloc[idx]["hugo_symbol"])
        elif "gene_name" in adata.var.columns:
            return str(adata.var.iloc[idx]["gene_name"])
        elif "gene_symbol" in adata.var.columns:
            return str(adata.var.iloc[idx]["gene_symbol"])
    # Otherwise assume var_names are already Hugo symbols (not ENSG IDs)
    return gene


def _select_marker(adata) -> str:
    marker_list = ["MS4A1", "LYZ", "COL1A1", "PECAM1", "EPCAM"]
    return select_gene(adata, marker_list, fallback_index=0)


def _select_housekeeping(adata) -> str:
    hk_list = ["ACTB", "GAPDH", "RPLP0"]
    try:
        return select_gene(adata, hk_list, fallback_index=1)
    except IndexError:
        return select_gene(adata, [], fallback_index=0)


def _select_random(adata, exclude: set[str], seed: int) -> str:
    var_names = list(adata.var_names)
    rng = np.random.default_rng(seed)
    for idx in rng.permutation(len(var_names)):
        gene = var_names[idx]
        if gene not in exclude:
            return gene
    return var_names[0]


def _outputs_root(outdir: str) -> Path:
    return Path(outdir) / "outputs"


def _figure_dir(outdir: str) -> Path:
    return _outputs_root(outdir) / "figures_v1"


def _table_dir(outdir: str) -> Path:
    return _outputs_root(outdir) / "tables"


def _read_adata(h5ad_path: str):
    path = Path(h5ad_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file '{h5ad_path}' not found.")
    return sc.read_h5ad(path)


def _append_runlog(outdir: str, note: str) -> None:
    runlog_path = _outputs_root(outdir) / "logs" / "runlog.md"
    ensure_dir(runlog_path.parent.as_posix())
    with open(runlog_path, "a", encoding="utf-8") as fh:
        fh.write(f"{note}\n")


def smoke_rsp_main(argv: Iterable[str] | None = None) -> int:
    """Run RSP smoke test.

    Args:
        argv: Command-line arguments (if None, uses sys.argv).

    Returns:
        Exit code (0 for success).
    """
    parser = argparse.ArgumentParser(description="BioRSP RSP smoke test")
    parser.add_argument(
        "--h5ad", default="adata_embed_graph.h5ad", help="Path to .h5ad file"
    )
    parser.add_argument("--outdir", default=".", help="Output directory root")
    parser.add_argument("--bins", type=int, default=72, help="Number of angular bins")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (unused here)")
    args = parser.parse_args(list(argv) if argv is not None else None)

    adata = _read_adata(args.h5ad)
    if "X_umap" not in adata.obsm:
        raise KeyError("adata.obsm['X_umap'] is required for geometry.")

    umap_xy = adata.obsm["X_umap"]
    vantage = compute_vantage(umap_xy)
    angles = compute_angles(umap_xy, vantage)

    gene = _select_marker(adata)
    expr = get_gene_vector(adata, gene)
    E_phi, phi_max, E_max = compute_rsp_profile(expr, angles, n_bins=args.bins)

    fig_dir = _figure_dir(args.outdir)
    ensure_dir(fig_dir.as_posix())
    out_png = fig_dir / "debug_rsp_marker.png"
    gene_display = _get_gene_display_name(adata, gene)
    plot_rsp_polar(E_phi, out_png.as_posix(), title=f"RSP: {gene_display}")

    print(f"gene={gene_display}")
    print(f"E_max={E_max}")
    print(f"phi_max_rad={phi_max}")
    print(f"phi_max_deg={phi_max * 180.0 / math.pi}")
    return 0


def smoke_moran_main(argv: Iterable[str] | None = None) -> int:
    """Run Moran's I smoke test.

    Args:
        argv: Command-line arguments (if None, uses sys.argv).

    Returns:
        Exit code (0 for success).
    """
    parser = argparse.ArgumentParser(description="BioRSP Moran's I smoke test")
    parser.add_argument(
        "--h5ad", default="adata_embed_graph.h5ad", help="Path to .h5ad file"
    )
    parser.add_argument("--outdir", default=".", help="Output directory root")
    parser.add_argument("--bins", type=int, default=72, help="Unused for Moran")
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for random gene"
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    adata = _read_adata(args.h5ad)
    W = extract_weights(adata)  # row-standardized

    marker = _select_marker(adata)
    housekeeping = _select_housekeeping(adata)
    random_gene = _select_random(adata, exclude={marker, housekeeping}, seed=args.seed)

    marker_display = _get_gene_display_name(adata, marker)
    housekeeping_display = _get_gene_display_name(adata, housekeeping)
    random_display = _get_gene_display_name(adata, random_gene)

    results = []
    for gene, gene_display in [
        (marker, marker_display),
        (housekeeping, housekeeping_display),
        (random_gene, random_display),
    ]:
        x = get_gene_vector(adata, gene).astype(float)
        moran_i = morans_i(x, W, row_standardize=False)
        results.append({"gene": gene_display, "moran_I": moran_i})

    table_dir = _table_dir(args.outdir)
    ensure_dir(table_dir.as_posix())
    out_csv = table_dir / "debug_moran.csv"
    pd.DataFrame(results).to_csv(out_csv.as_posix(), index=False)

    marker_I = next(r["moran_I"] for r in results if r["gene"] == marker_display)
    random_I = next(r["moran_I"] for r in results if r["gene"] == random_display)
    if marker_I > random_I:
        note = "marker_vs_random_note=marker I > random I"
    else:
        note = "marker_vs_random_note=marker I <= random I (not guaranteed)"
    print(note)

    _append_runlog(args.outdir, "morans_I_weights=row_standardized")
    return 0


def smoke_perm_main(argv: Iterable[str] | None = None) -> int:
    """Run donor-stratified permutation smoke test.

    Args:
        argv: Command-line arguments (if None, uses sys.argv).

    Returns:
        Exit code (0 for success).
    """
    parser = argparse.ArgumentParser(description="BioRSP permutation smoke test")
    parser.add_argument(
        "--h5ad", default="adata_embed_graph.h5ad", help="Path to .h5ad file"
    )
    parser.add_argument("--outdir", default=".", help="Output directory root")
    parser.add_argument("--bins", type=int, default=72, help="Number of angular bins")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--n-perm", type=int, default=100, help="Number of permutations"
    )
    parser.add_argument(
        "--donor-col",
        default="donor",
        help="Column name in adata.obs for donor/subject ID",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    adata = _read_adata(args.h5ad)
    if "X_umap" not in adata.obsm:
        raise KeyError("adata.obsm['X_umap'] is required for geometry.")
    if args.donor_col not in adata.obs:
        raise KeyError(
            f"adata.obs['{args.donor_col}'] missing; pseudoreplication risk without donor stratification."
        )

    umap_xy = adata.obsm["X_umap"]
    vantage = compute_vantage(umap_xy)
    angles = compute_angles(umap_xy, vantage)
    donor_ids = np.asarray(adata.obs[args.donor_col])

    marker = _select_marker(adata)
    housekeeping = _select_housekeeping(adata)

    marker_display = _get_gene_display_name(adata, marker)
    housekeeping_display = _get_gene_display_name(adata, housekeeping)

    records = []
    for gene, gene_display in [(marker, marker_display), (housekeeping, housekeeping_display)]:
        expr = get_gene_vector(adata, gene)
        null_emax, E_max_obs, phi_max_obs, p = perm_null_emax(
            expr,
            angles,
            donor_ids,
            n_bins=args.bins,
            n_perm=args.n_perm,
            seed=args.seed,
        )
        records.append(
            {
                "gene": gene_display,
                "E_max": E_max_obs,
                "phi_max": phi_max_obs,
                "p_perm_100": p,
            }
        )
        print(
            f"gene={gene_display} E_max={E_max_obs} phi_max_rad={phi_max_obs} p_perm_100={p}"
        )

        if gene == marker:
            fig_dir = _figure_dir(args.outdir)
            ensure_dir(fig_dir.as_posix())
            out_png = fig_dir / "null_marker.png"
            plot_null_distribution(
                null_emax,
                E_max_obs,
                out_png.as_posix(),
                title=f"Null E_max: {gene_display}",
            )

    table_dir = _table_dir(args.outdir)
    ensure_dir(table_dir.as_posix())
    pd.DataFrame(records).to_csv((table_dir / "perm_smoke.csv").as_posix(), index=False)

    hk_p = next(r["p_perm_100"] for r in records if r["gene"] == housekeeping_display)
    gate_status = "PASS" if hk_p >= 0.2 else "FAIL"
    print(f"housekeeping_p_gate={hk_p}={gate_status}")
    if gate_status == "FAIL":
        print(
            "Check donor-strat shuffle and expression threshold; housekeeping should usually be diffuse."
        )
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    """Main CLI entry point.

    Args:
        argv: Command-line arguments (if None, uses sys.argv).

    Returns:
        Exit code.
    """
    parser = argparse.ArgumentParser(description="BioRSP CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("smoke-rsp", help="Run RSP smoke test")
    sub.add_parser("smoke-moran", help="Run Moran's I smoke test")
    sub.add_parser("smoke-perm", help="Run permutation smoke test")

    args, remainder = parser.parse_known_args(list(argv) if argv is not None else None)
    if args.command == "smoke-rsp":
        return smoke_rsp_main(remainder)
    if args.command == "smoke-moran":
        return smoke_moran_main(remainder)
    if args.command == "smoke-perm":
        return smoke_perm_main(remainder)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
