"""
KPMP TAL Cell Gene Expression Analysis - Repair & Injury Ma)

scanner = RadarScanner(params).fit(
    coords,
    batches=tal_adata.obs["donor_id"].values,
)
This script analyzes specific genes associated with TAL cell trajectories and repair:

TAL Function & Regulation: SLC12A1, ESRRB, EGF
Early Repair Markers: PROM1, DCDC1, HAVCR1, SPP1
Adaptive States (aPT/aTAL): VCAM1, DCDC2, HAVCR1, ITGB6
aTAL Specific (human): PROM1, DCDC2
Degenerative/Injury Markers: SPP1, CST3, CLU, IGFBP7, DEFB1
Stromal/Microenvironment: CDH11

Analysis steps:
1. Filters KPMP data to only TAL (Thick Ascending Limb) cells
2. Scans curated list of TAL repair and injury marker genes
3. Saves results to CSV and generates visualizations for all markers
"""

import anndata
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.sparse as sp
from biorsp.radar_scan import ScanParams, RadarScanner
from biorsp.plotting import plot_rsp_summary

script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "data", "kpmp_sn.h5ad")
output_dir = os.path.join(script_dir, "outputs", "tal_analysis")
os.makedirs(output_dir, exist_ok=True)

print("=" * 80)
print("KPMP TAL Cell Gene Expression Analysis - Repair & Injury Markers")
print("=" * 80)

print("\n1. Loading data...")
kpmp_adata = anndata.read_h5ad(data_path)
print(f"   Loaded: {kpmp_adata.n_obs} cells × {kpmp_adata.n_vars} genes")

print("\n2. Filtering for TAL cells...")
tal_mask = kpmp_adata.obs["subclass.l1"] == "TAL"
tal_adata = kpmp_adata[tal_mask, :].copy()
print(
    f"   TAL cells: {tal_adata.n_obs} ({100 * tal_adata.n_obs / kpmp_adata.n_obs:.1f}%)"
)

coords = tal_adata.obsm["X_umap"]

print("\n3. Configuring RadarScanner...")
params = ScanParams(
    B=360,
    widths_deg=(15, 30, 60, 90, 120, 180),
    radial_mode="width",
    n_bands=5,
    standardize="rank",
    residualize="none",
    density_correction="none",
    null_model="within_batch_rotation",
    R=500,
    threshold_mode="positive",
)

scanner = RadarScanner(params).fit(
    coords,
    covariates=tal_adata.obs[["nCount_RNA", "percent.mt", "percent.er"]].to_numpy(),
    batches=tal_adata.obs["donor_id"].to_numpy(),
)
print(f"   Scanner fitted with {scanner.N} TAL cells")

# Define curated gene lists for TAL repair and injury
print("\n4. Defining TAL marker genes...")
tal_marker_genes = {
    "TAL Function & Regulation": ["SLC12A1", "ESRRB", "EGF"],
    "Early Repair": ["PROM1", "DCDC1", "HAVCR1", "SPP1"],
    "Adaptive States (aPT/aTAL)": ["VCAM1", "DCDC2", "HAVCR1", "ITGB6"],
    "aTAL Specific (human)": ["PROM1", "DCDC2"],
    "Degenerative/Injury": ["SPP1", "CST3", "CLU", "IGFBP7", "DEFB1"],
    "Stromal/Microenvironment": ["CDH11"],
}

# Create flat list of unique genes
all_genes = set()
for category_genes in tal_marker_genes.values():
    all_genes.update(category_genes)
genes_to_scan = sorted(list(all_genes))

print(f"   Total unique marker genes to analyze: {len(genes_to_scan)}")
for category, genes in tal_marker_genes.items():
    print(f"   - {category}: {', '.join(genes)}")

print("\n5. Preparing gene expression data...")
X = tal_adata.X
is_sparse = sp.issparse(X)
if not is_sparse:
    gene_expr_matrix = np.asarray(X)
else:
    gene_expr_matrix = None
    print("   Detected sparse matrix; extracting per-gene vectors on demand")

if "feature_name" in tal_adata.var.columns:
    gene_names = tal_adata.var["feature_name"].tolist()
else:
    gene_names = tal_adata.var_names.tolist()

gene_name_to_idx = {name: idx for idx, name in enumerate(gene_names)}

print("\n6. Scanning marker genes...")
results = []
feature_results = {}
genes_found = []
genes_not_found = []
zero_expression_genes = []

for gene_name in genes_to_scan:
    # Find gene in dataset
    if gene_name in gene_name_to_idx:
        gene_idx = gene_name_to_idx[gene_name]
        genes_found.append(gene_name)
    else:
        genes_not_found.append(gene_name)
        print(f"   ⚠ Gene not found in dataset: {gene_name}")
        continue

    if is_sparse:
        gene_slice = X[:, gene_idx]
        if sp.issparse(gene_slice):
            gene_expr = gene_slice.toarray().ravel()
        else:
            gene_expr = np.asarray(gene_slice).ravel()
    else:
        gene_expr = gene_expr_matrix[:, gene_idx].ravel()

    if np.all(gene_expr == 0):
        zero_expression_genes.append(gene_name)
        print(f"   ⚠ Gene has zero expression: {gene_name}")
        continue

    # Scan the gene
    result = scanner.scan_feature(gene_expr, name=gene_name)

    # Calculate expression statistics
    n_expressing = np.sum(gene_expr > 0)
    pct_expressing = 100 * n_expressing / len(gene_expr)
    mean_expr = np.mean(gene_expr[gene_expr > 0]) if n_expressing > 0 else 0

    # Determine which categories this gene belongs to
    categories = [cat for cat, genes in tal_marker_genes.items() if gene_name in genes]

    results.append(
        {
            "gene": gene_name,
            "categories": "; ".join(categories),
            "Z_max": result.Z_max,
            "p_value": result.p_value,
            "peak_angle_deg": np.degrees(result.phi_star),
            "enrichment_ratio": result.ER,
            "n_expressing": n_expressing,
            "pct_expressing": pct_expressing,
            "mean_expr_in_expressing": mean_expr,
            "width_idx": result.width_idx,
            "R_conc": result.R_conc,
        }
    )
    
    # Store the full result object and gene expression for plotting
    feature_results[gene_name] = {
        "result": result,
        "gene_expr": gene_expr,
    }
    
    print(f"   ✓ Scanned: {gene_name} (p={result.p_value:.2e}, Z={result.Z_max:.2f})")

print(f"\n   Genes found and scanned: {len(genes_found)}")
if genes_not_found:
    print(f"   Genes not found in dataset: {', '.join(genes_not_found)}")
if zero_expression_genes:
    print(f"   Genes skipped due to zero detected expression: {', '.join(zero_expression_genes)}")

print("\n7. Saving results...")
if not results:
    print("   No marker genes with detectable expression; skipping result export.")
    sys.exit(0)

df = pd.DataFrame(results)
df = df.sort_values("p_value")

csv_path = os.path.join(output_dir, "tal_marker_genes_results.csv")
df.to_csv(csv_path, index=False)
print(f"   Saved results to: {csv_path}")

print("\n" + "=" * 80)
print("Analysis Summary - TAL Repair & Injury Markers")
print("=" * 80)
print(f"Total marker genes analyzed: {len(df)}")
print(f"Significant genes (p < 0.05): {(df['p_value'] < 0.05).sum()}")
print(f"Significant genes (p < 0.01): {(df['p_value'] < 0.01).sum()}")
print(f"Significant genes (p < 0.001): {(df['p_value'] < 0.001).sum()}")

print("\nAll marker genes by p-value:")
print("-" * 80)
for i, row in df.iterrows():
    print(
        f"{row['gene']:15s} | {row['categories']:40s} | p={row['p_value']:.2e} | "
        f"Z={row['Z_max']:.2f} | angle={row['peak_angle_deg']:.1f}° | {row['pct_expressing']:.1f}%"
    )

print("\n8. Generating plots for all marker genes...")
for i, (idx, row) in enumerate(df.iterrows()):
    gene_name = row["gene"]
    stored = feature_results[gene_name]
    result = stored["result"]
    gene_expr = stored["gene_expr"]

    fig = plot_rsp_summary(result, coords, gene_expr, figsize=(14, 6), scanner=scanner)
    plot_path = os.path.join(output_dir, f"marker_{i+1:02d}_{gene_name}.png")
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"   Saved plot {i+1}/{len(df)}: {gene_name}")

print("\n" + "=" * 80)
print("Analysis Complete!")
print("=" * 80)
print(f"Results saved to: {output_dir}")
print(f"- Full results CSV: {csv_path}")
print(f"- Marker gene plots: marker_01_*.png through marker_{len(df):02d}_*.png")
print("=" * 80)
