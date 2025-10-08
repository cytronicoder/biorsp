import anndata
import os
import numpy as np
import matplotlib.pyplot as plt
from biorsp.radar_scan import ScanParams, RadarScanner

script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "data", "kpmp_sn.h5ad")

kpmp_adata = anndata.read_h5ad(data_path)
print(kpmp_adata)

coords = kpmp_adata.obsm["X_umap"]

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

print("\nScanning donor features...")
donor_results = []
for donor in kpmp_adata.obs["donor_id"].unique():
    feat = (kpmp_adata.obs["donor_id"].to_numpy() == donor).astype(float)
    res = scanner.scan_feature(feat, name=f"donor={donor}")
    donor_results.append(res)
    print(f"  {res.name}: Z_max={res.Z_max:.2f}, p={res.p_value:.4f}")

print("\nScanning percent.cortex...")
feat = kpmp_adata.obs["percent.cortex"].to_numpy()
res_cortex = scanner.scan_feature(feat, name="percent.cortex")
print(f"  {res_cortex.name}: Z_max={res_cortex.Z_max:.2f}, p={res_cortex.p_value:.4f}")

print("\nScanning class features...")
class_results = []
for ct in kpmp_adata.obs["class"].unique():
    feat = (kpmp_adata.obs["class"].to_numpy() == ct).astype(float)
    res = scanner.scan_feature(feat, name=f"class={ct}")
    class_results.append(res)
    print(f"  {res.name}: Z_max={res.Z_max:.2f}, p={res.p_value:.4f}")

# Scan genes from var dataframe
print("\nScanning gene expression features...")
print(f"Available columns in .var: {list(kpmp_adata.var.columns)}")

# Get gene names from feature_name column
if "feature_name" in kpmp_adata.var.columns:
    gene_names = kpmp_adata.var["feature_name"].tolist()
    print(f"Total genes: {len(gene_names)}")

    # Select a subset of interesting genes to scan (top variable genes or specific markers)
    # For demonstration, let's select genes with highest variance
    gene_vars = np.var(
        kpmp_adata.X.toarray() if hasattr(kpmp_adata.X, "toarray") else kpmp_adata.X,
        axis=0,
    )
    top_gene_indices = np.argsort(gene_vars)[::-1][:20]  # Top 20 most variable genes

    print(f"\nScanning top 20 most variable genes...")
    gene_results = []
    for idx in top_gene_indices:
        gene_name = gene_names[idx]
        # Get gene expression for this gene
        if hasattr(kpmp_adata.X, "toarray"):
            gene_expr = kpmp_adata.X[:, idx].toarray().ravel()
        else:
            gene_expr = kpmp_adata.X[:, idx].ravel()

        res = scanner.scan_feature(gene_expr, name=f"gene={gene_name}")
        gene_results.append(res)
        print(f"  {res.name}: Z_max={res.Z_max:.2f}, p={res.p_value:.4f}")

    # Create RSP plots for top significant genes
    print("\nGenerating RSP plots...")

    # Sort by p-value and plot top 6
    gene_results_sorted = sorted(gene_results, key=lambda x: x.p_value)
    top_results = gene_results_sorted[:6]

    if len(top_results) > 0:
        fig, axes = plt.subplots(
            2, 3, figsize=(15, 10), subplot_kw=dict(projection="polar")
        )
        axes = axes.ravel()

        for i, res in enumerate(top_results):
            ax = axes[i]

            if res.Z_heat is not None:
                # Z_heat is shape (J, B) where J is number of widths/bands and B is number of angular bins
                # Create polar heatmap
                J, B = res.Z_heat.shape

                # Create angular and radial coordinates
                theta = np.linspace(0, 2 * np.pi, B, endpoint=False)
                r = np.arange(J + 1)  # J bands

                # Create meshgrid for pcolormesh
                theta_grid, r_grid = np.meshgrid(theta, r)

                # Plot heatmap
                im = ax.pcolormesh(
                    theta_grid,
                    r_grid,
                    res.Z_heat,
                    cmap="RdBu_r",
                    shading="flat",
                    vmin=-np.abs(res.Z_heat).max(),
                    vmax=np.abs(res.Z_heat).max(),
                )

                # Mark the peak
                peak_angle = res.phi_star
                ax.plot([peak_angle, peak_angle], [0, J], "g-", linewidth=2, alpha=0.7)

                # Set title with statistics
                ax.set_title(
                    f"{res.name}\nZ={res.Z_max:.2f}, p={res.p_value:.4f}",
                    fontsize=10,
                    pad=10,
                )

                # Colorbar
                plt.colorbar(im, ax=ax, label="Z-score", pad=0.1, fraction=0.046)
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No heatmap data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title(res.name, fontsize=10)

        plt.tight_layout()
        output_path = os.path.join(script_dir, "gene_rsp_plots.png")
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved RSP plots to: {output_path}")
        plt.close()

    # Also create individual plots for the most significant gene
    if len(gene_results_sorted) > 0:
        best_result = gene_results_sorted[0]

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="polar")

        if best_result.Z_heat is not None:
            J, B = best_result.Z_heat.shape
            theta = np.linspace(0, 2 * np.pi, B, endpoint=False)
            r = np.arange(J + 1)
            theta_grid, r_grid = np.meshgrid(theta, r)

            im = ax.pcolormesh(
                theta_grid,
                r_grid,
                best_result.Z_heat,
                cmap="RdBu_r",
                shading="flat",
                vmin=-np.abs(best_result.Z_heat).max(),
                vmax=np.abs(best_result.Z_heat).max(),
            )

            # Mark the peak
            peak_angle = best_result.phi_star
            ax.plot(
                [peak_angle, peak_angle],
                [0, J],
                "g-",
                linewidth=3,
                alpha=0.8,
                label="Peak direction",
            )

            ax.set_title(
                f"{best_result.name}\nZ-score={best_result.Z_max:.2f}, "
                f"p-value={best_result.p_value:.6f}\n"
                f"Peak angle={np.degrees(best_result.phi_star):.1f}°",
                fontsize=14,
                pad=20,
            )

            plt.colorbar(im, ax=ax, label="Z-score", pad=0.1)
            ax.legend(loc="upper left", bbox_to_anchor=(1.2, 1.0))

            output_path = os.path.join(
                script_dir, f"best_gene_rsp_{best_result.name.replace('gene=', '')}.png"
            )
            plt.savefig(output_path, dpi=200, bbox_inches="tight")
            print(f"Saved detailed RSP plot for top gene to: {output_path}")
            plt.close()

print("\n" + "=" * 60)
print("Analysis complete!")
