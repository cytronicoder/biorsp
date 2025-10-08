"""
Enhanced KPMP example with gene expression scanning and RSP visualization.

This script demonstrates:
1. Loading KPMP single-nucleus RNA-seq data
2. Scanning donor, class, and gene expression features
3. Creating publication-quality RSP plots
"""

import anndata
import os
import numpy as np
import matplotlib.pyplot as plt
from biorsp.radar_scan import ScanParams, RadarScanner
from biorsp.plotting import plot_rsp_grid, plot_rsp_summary, save_top_results

# Set up paths
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "data", "kpmp_sn.h5ad")
output_dir = os.path.join(script_dir, "output")
os.makedirs(output_dir, exist_ok=True)

print("=" * 60)
print("KPMP Single-Nucleus RNA-seq Analysis with BioRSP")
print("=" * 60)

# Load data
print("\n1. Loading data...")
kpmp_adata = anndata.read_h5ad(data_path)
print(f"   Loaded: {kpmp_adata.n_obs} cells × {kpmp_adata.n_vars} genes")
print(f"   Metadata columns: {list(kpmp_adata.obs.columns[:10])}...")
print(f"   Gene metadata: {list(kpmp_adata.var.columns)}")

coords = kpmp_adata.obsm["X_umap"]

# Configure scanner
print("\n2. Configuring RadarScanner...")
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
)

scanner = RadarScanner(params).fit(
    coords,
    covariates=kpmp_adata.obs[["nCount_RNA", "percent.mt", "percent.er"]].to_numpy(),
    batches=kpmp_adata.obs["donor_id"].to_numpy(),
)
print(f"   Scanner fitted with {scanner.N} cells")

# Scan metadata features
print("\n3. Scanning metadata features...")

# Donors
print("\n   a) Scanning donor features...")
donor_results = []
for donor in kpmp_adata.obs["donor_id"].unique():
    feat = (kpmp_adata.obs["donor_id"].to_numpy() == donor).astype(float)
    res = scanner.scan_feature(feat, name=f"donor={donor}")
    donor_results.append(res)
    print(f"      {res.name}: Z_max={res.Z_max:.2f}, p={res.p_value:.4f}")

# Percent cortex
print("\n   b) Scanning percent.cortex...")
feat = kpmp_adata.obs["percent.cortex"].to_numpy()
res_cortex = scanner.scan_feature(feat, name="percent.cortex")
print(
    f"      {res_cortex.name}: Z_max={res_cortex.Z_max:.2f}, p={res_cortex.p_value:.4f}"
)

# Cell classes
print("\n   c) Scanning cell class features...")
class_results = []
for ct in kpmp_adata.obs["class"].unique():
    feat = (kpmp_adata.obs["class"].to_numpy() == ct).astype(float)
    res = scanner.scan_feature(feat, name=f"class={ct}")
    class_results.append(res)
    print(f"      {res.name}: Z_max={res.Z_max:.2f}, p={res.p_value:.4f}")

# Scan gene expression features
print("\n4. Scanning gene expression features...")

if "feature_name" in kpmp_adata.var.columns:
    gene_names = kpmp_adata.var["feature_name"].tolist()
    print(f"   Total genes available: {len(gene_names)}")

    # Select top variable genes
    print("   Computing gene variance...")
    if hasattr(kpmp_adata.X, "toarray"):
        X_dense = kpmp_adata.X.toarray()
    else:
        X_dense = kpmp_adata.X

    gene_vars = np.var(X_dense, axis=0)
    top_gene_indices = np.argsort(gene_vars)[::-1][:50]  # Top 50 most variable

    print(f"   Scanning top 50 most variable genes...")
    gene_results = []

    for i, idx in enumerate(top_gene_indices):
        gene_name = gene_names[idx]

        # Get gene expression
        if hasattr(kpmp_adata.X, "toarray"):
            gene_expr = kpmp_adata.X[:, idx].toarray().ravel()
        else:
            gene_expr = kpmp_adata.X[:, idx].ravel()

        res = scanner.scan_feature(gene_expr, name=f"{gene_name}")
        gene_results.append(res)

        if (i + 1) % 10 == 0:
            print(f"      Progress: {i+1}/50 genes scanned")

    print("   Completed gene scanning!")

    # Print top results
    print("\n   Top 10 genes by p-value:")
    gene_results_sorted = sorted(gene_results, key=lambda x: x.p_value)
    for i, res in enumerate(gene_results_sorted[:10]):
        print(f"      {i+1}. {res.name}: Z_max={res.Z_max:.2f}, p={res.p_value:.6f}")

    # Generate visualizations
    print("\n5. Generating visualizations...")

    # Plot grid of top genes
    print("   a) Creating grid plot for top 12 genes...")
    fig = plot_rsp_grid(
        gene_results,
        ncols=4,
        max_plots=12,
        sort_by="p_value",
        suptitle="Top 12 Most Significant Genes - RSP Heatmaps",
        show_peaks=True,
    )
    output_path = os.path.join(output_dir, "top_genes_grid.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"      Saved to: {output_path}")

    # Plot grid of cell classes
    print("   b) Creating grid plot for cell classes...")
    fig = plot_rsp_grid(
        class_results,
        ncols=3,
        sort_by="p_value",
        suptitle="Cell Class Features - RSP Heatmaps",
        show_peaks=True,
    )
    output_path = os.path.join(output_dir, "cell_classes_grid.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"      Saved to: {output_path}")

    # Create summary plot for top gene
    print("   c) Creating detailed summary for top gene...")
    best_gene_result = gene_results_sorted[0]
    best_gene_idx = gene_names.index(best_gene_result.name)

    if hasattr(kpmp_adata.X, "toarray"):
        best_gene_expr = kpmp_adata.X[:, best_gene_idx].toarray().ravel()
    else:
        best_gene_expr = kpmp_adata.X[:, best_gene_idx].ravel()

    fig = plot_rsp_summary(best_gene_result, coords, best_gene_expr, figsize=(16, 5))
    output_path = os.path.join(output_dir, f"summary_{best_gene_result.name}.png")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"      Saved to: {output_path}")

    # Save individual plots for top 5 genes
    print("   d) Saving individual plots for top 5 genes...")
    saved_paths = save_top_results(
        gene_results,
        output_dir=output_dir,
        top_n=5,
        sort_by="p_value",
        prefix="gene_rsp",
        dpi=200,
    )
    print(f"      Saved {len(saved_paths)} individual plots")

    # Create summary statistics CSV
    print("\n6. Saving summary statistics...")
    import pandas as pd

    # Combine all results
    all_results = gene_results + class_results + [res_cortex]

    summary_data = []
    for res in all_results:
        summary_data.append(
            {
                "feature": res.name,
                "Z_max": res.Z_max,
                "p_value": res.p_value,
                "peak_angle_deg": np.degrees(res.phi_star),
                "width_idx": res.width_idx,
                "center_idx": res.center_idx,
                "enrichment_ratio": res.ER,
                "concentration": res.R_conc,
            }
        )

    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values("p_value")

    csv_path = os.path.join(output_dir, "rsp_summary_statistics.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"   Saved summary statistics to: {csv_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)
    print(f"Total features scanned: {len(all_results)}")
    print(
        f"Significant features (p < 0.05): {sum(r.p_value < 0.05 for r in all_results)}"
    )
    print(f"\nOutputs saved to: {output_dir}")
    print("  - top_genes_grid.png: Grid of top 12 genes")
    print("  - cell_classes_grid.png: Grid of cell classes")
    print(f"  - summary_{best_gene_result.name}.png: Detailed view of top gene")
    print("  - gene_rsp_*.png: Individual plots for top 5 genes")
    print("  - rsp_summary_statistics.csv: All results statistics")
    print("=" * 60)

else:
    print("\n   WARNING: 'feature_name' column not found in .var")
    print("   Available columns:", list(kpmp_adata.var.columns))
    print("   Skipping gene expression analysis.")

print("\nDone!")
