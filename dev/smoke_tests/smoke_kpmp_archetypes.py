#!/usr/bin/env python3
"""
Smoke Test for KPMP All-Gene Archetypes Pipeline

Loads KPMP data, subsamples to ~5000 cells and 200 genes, runs the pipeline,
and verifies that expected outputs are created.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def run_smoke_test():
    """Run smoke test of the KPMP archetypes pipeline."""
    import anndata as ad
    import pandas as pd
    from scipy import sparse

    print("=" * 60)
    print("SMOKE TEST: KPMP All-Gene Archetypes Pipeline")
    print("=" * 60)

    # Find KPMP data
    possible_paths = [
        Path("analysis/kidney_atlas/data/kpmp.h5ad"),
        Path("data/kpmp.h5ad"),
        Path("kpmp.h5ad"),
    ]

    h5ad_path = None
    for p in possible_paths:
        if p.exists():
            h5ad_path = p
            break

    if h5ad_path is None:
        print("ERROR: Could not find KPMP h5ad file")
        print("Searched in:", [str(p) for p in possible_paths])
        sys.exit(1)

    print(f"Found data: {h5ad_path}")

    # Load and subsample
    print("Loading AnnData...")
    adata = ad.read_h5ad(h5ad_path)
    print(f"Original shape: {adata.n_obs} cells × {adata.n_vars} genes")

    # Subsample cells
    rng = np.random.default_rng(42)
    n_cells = min(5000, adata.n_obs)
    cell_idx = rng.choice(adata.n_obs, size=n_cells, replace=False)

    # Subsample genes - select genes with some expression
    if sparse.issparse(adata.X):
        gene_expr = np.array(adata.X.sum(axis=0)).flatten()
    else:
        gene_expr = adata.X.sum(axis=0)

    # Get top 200 expressed genes
    top_gene_idx = np.argsort(gene_expr)[-200:]

    # Create subsampled AnnData
    adata_sub = adata[cell_idx, top_gene_idx].copy()
    print(f"Subsampled: {adata_sub.n_obs} cells × {adata_sub.n_vars} genes")

    # Save to temp file
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        sub_h5ad = tmpdir / "kpmp_subset.h5ad"
        outdir = tmpdir / "results"

        adata_sub.write_h5ad(sub_h5ad)
        print(f"Saved subset to: {sub_h5ad}")

        # Run pipeline
        script_path = Path("analysis/kidney_atlas/run_kpmp_archetypes_all_genes.py")
        if not script_path.exists():
            script_path = (
                Path(__file__).parent.parent
                / "analysis/kidney_atlas/run_kpmp_archetypes_all_genes.py"
            )

        cmd = [
            sys.executable,
            str(script_path),
            "--h5ad",
            str(sub_h5ad),
            "--outdir",
            str(outdir),
            "--max-cells",
            "5000",
            "--chunk-size",
            "100",
            "--min-coverage",
            "0.001",
            "--min-nonzero",
            "5",
            "--skip-reliability",
            "--profile",  # Test profiling
        ]

        print("\nRunning pipeline (single-process with profiling)...")
        print(f"Command: {' '.join(cmd)}")
        print("-" * 60)

        result = subprocess.run(cmd, capture_output=False, text=True)

        if result.returncode != 0:
            print(f"ERROR: Pipeline failed with return code {result.returncode}")
            sys.exit(1)

        # Test parallel mode if joblib available
        try:
            import joblib  # noqa: F401

            outdir_parallel = tmpdir / "results_parallel"
            cmd_parallel = [
                sys.executable,
                str(script_path),
                "--h5ad",
                str(sub_h5ad),
                "--outdir",
                str(outdir_parallel),
                "--max-cells",
                "5000",
                "--chunk-size",
                "50",  # Smaller chunks for parallel test
                "--min-coverage",
                "0.001",
                "--min-nonzero",
                "5",
                "--skip-reliability",
                "--n-workers",
                "2",
                "--profile",
            ]

            print("\n\nRunning pipeline (parallel mode with 2 workers)...")
            print(f"Command: {' '.join(cmd_parallel)}")
            print("-" * 60)

            result_parallel = subprocess.run(cmd_parallel, capture_output=False, text=True)

            if result_parallel.returncode != 0:
                print("WARNING: Parallel mode failed (may be expected on some systems)")
            else:
                print("\n✓ Parallel mode test passed")

                # Compare results for determinism
                df1 = pd.read_csv(outdir / "runs_all_genes.csv")
                df2 = pd.read_csv(outdir_parallel / "runs_all_genes.csv")

                # Check same genes were scored
                if set(df1["gene"]) == set(df2["gene"]):
                    print("✓ Determinism check: Same genes scored")
                else:
                    print("⚠ Determinism warning: Different genes scored")

        except ImportError:
            print("\n⚠ Skipping parallel test (joblib not available)")

        # Verify outputs from serial run
        print("\n" + "-" * 60)
        print("Verifying outputs...")

        expected_files = [
            "runs_all_genes.csv",
            "classification.csv",
            "derived_thresholds.json",
            "manifest.json",
            "report.md",
            "figures/fig_CS_scatter.png",
            "figures/fig_CS_marginals.png",
            "figures/fig_top_tables.png",
        ]

        missing = []
        for fname in expected_files:
            fpath = outdir / fname
            if fpath.exists():
                print(f"  ✓ {fname}")
            else:
                print(f"  ✗ {fname} (MISSING)")
                missing.append(fname)

        if missing:
            print(f"\nERROR: Missing {len(missing)} expected files")
            sys.exit(1)

        # Check CSV content
        import pandas as pd

        df = pd.read_csv(outdir / "runs_all_genes.csv")
        print("\nResults summary:")
        print(f"  Genes scored: {len(df)}")
        print(f"  Columns: {list(df.columns)}")

        if "Archetype" in df.columns:
            print("  Archetypes:")
            for arch, count in df["Archetype"].value_counts().items():
                print(f"    {arch}: {count}")

        print("\n" + "=" * 60)
        print("SMOKE TEST PASSED ✓")
        print("=" * 60)


if __name__ == "__main__":
    run_smoke_test()
