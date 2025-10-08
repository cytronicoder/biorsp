"""
KPMP Gene Expression Analysis with Foreground/Background Comparison.

This script demonstrates:
1. Loading KPMP single-nucleus RNA-seq data
2. Scanning a single gene with foreground vs background comparison
3. Creating publication-quality RSP plots
"""

import anndata
import os
import numpy as np
import matplotlib.pyplot as plt
from biorsp.radar_scan import ScanParams, RadarScanner
from biorsp.plotting import plot_rsp_summary

# Set up paths
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "data", "kpmp_sn.h5ad")
output_dir = os.path.join(script_dir, "output")
os.makedirs(output_dir, exist_ok=True)

print("=" * 80)
print("KPMP Gene Expression Analysis - Foreground vs Background Comparison")
print("=" * 80)

# Load data
print("\n1. Loading data...")
kpmp_adata = anndata.read_h5ad(data_path)
print(f"   Loaded: {kpmp_adata.n_obs} cells × {kpmp_adata.n_vars} genes")

coords = kpmp_adata.obsm["X_umap"]

# Configure scanner with threshold-based binarization
print("\n2. Configuring RadarScanner...")
print(
    "   Using threshold_mode='positive' to compare expressing vs non-expressing cells"
)

params = ScanParams(
    B=180,
    widths_deg=(15, 30, 60, 90, 120, 180),
    radial_mode="quantile",
    n_bands=2,
    standardize="rank",
    residualize="ols",
    density_correction="2d",
    var_mode="binomial",
    overdispersion=0.0,
    null_model="within_batch_rotation",
    R=500,
    random_state=0,
    threshold_mode="positive",  # Cells with expression > 0 = foreground
)

scanner = RadarScanner(params).fit(
    coords,
    covariates=kpmp_adata.obs[["nCount_RNA", "percent.mt", "percent.er"]].to_numpy(),
    batches=kpmp_adata.obs["donor_id"].to_numpy(),
)
print(f"   Scanner fitted with {scanner.N} cells")

# Manually specify gene to analyze
gene_name = "UMOD"  # Change this to analyze different genes

print(f"\n3. Analyzing gene: {gene_name}")

if "feature_name" in kpmp_adata.var.columns:
    # Find gene index
    gene_names = kpmp_adata.var["feature_name"].tolist()

    if gene_name in gene_names:
        gene_idx = gene_names.index(gene_name)

        # Get gene expression
        if hasattr(kpmp_adata.X, "toarray"):
            gene_expr = kpmp_adata.X[:, gene_idx].toarray().ravel()
        else:
            gene_expr = kpmp_adata.X[:, gene_idx].ravel()

        # Print gene statistics
        n_expressing = np.sum(gene_expr > 0)
        pct_expressing = 100 * n_expressing / len(gene_expr)
        mean_expr = np.mean(gene_expr[gene_expr > 0]) if n_expressing > 0 else 0

        print(f"   Cells expressing (>0): {n_expressing} ({pct_expressing:.1f}%)")
        print(f"   Mean expression in expressing cells: {mean_expr:.3f}")

        # Scan the gene
        print("\n4. Scanning gene...")
        result = scanner.scan_feature(gene_expr, name=gene_name)

        print(f"   Z_max: {result.Z_max:.3f}")
        print(f"   p_value: {result.p_value:.6f}")
        print(f"   Peak angle: {np.degrees(result.phi_star):.1f}°")
        print(f"   Enrichment ratio: {result.ER:.3f}")

        # Generate visualization
        print("\n5. Creating summary plot...")
        fig = plot_rsp_summary(result, coords, gene_expr, figsize=(14, 6))
        output_path = os.path.join(output_dir, f"summary_{gene_name}.png")
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"   Saved to: {output_path}")

        # Print summary
        print("\n" + "=" * 80)
        print("Analysis Complete!")
        print("=" * 80)
        print(f"Gene: {gene_name}")
        print(f"  - Foreground: {n_expressing} cells expressing (value > 0)")
        print(
            f"  - Background: {len(gene_expr) - n_expressing} cells not expressing (value = 0)"
        )
        print(f"  - Z_max: {result.Z_max:.3f}")
        print(f"  - p_value: {result.p_value:.6f}")
        print(f"\nOutput saved to: {output_path}")
        print("=" * 80)

    else:
        print(f"\n   ERROR: Gene '{gene_name}' not found in dataset")
        print(f"   Available genes (first 20): {gene_names[:20]}")

else:
    print("\n   ERROR: 'feature_name' column not found in .var")
    print(f"   Available columns: {list(kpmp_adata.var.columns)}")

print("\nDone!")
