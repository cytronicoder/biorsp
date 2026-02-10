#!/usr/bin/env python3
"""Check donor distribution per cell type stratum."""

import scanpy as sc
import pandas as pd

# Load the data
adata = sc.read_h5ad("data/processed/HT_pca_umap.h5ad")

# Check specific cell types
strata_checks = [
    ("Vascular mural", ["Pericyte", "Smooth Muscle"]),
    ("Capillary EC", ["Capillary Endothelial"]),
    ("Arterial EC", ["Arterial Endothelial"]),
    ("Venous EC", ["Venous Endothelial"]),
    ("Endocardial", ["Endocardial"]),
    ("Lymphatic EC", ["Lymphatic Endothelial"]),
    ("Lymphoid", ["T", "NK", "B", "Mast"]),
]

min_cells_per_donor = 200  # Default from script
min_donors = 3  # Default from script

for stratum_name, labels in strata_checks:
    print(f"\n{'='*70}")
    print(f"Stratum: {stratum_name}")
    print(f"Labels: {labels}")

    # Get cells matching labels
    mask = adata.obs["azimuth_label"].isin(labels)
    n_total_cells = mask.sum()
    print(f"Total cells: {n_total_cells}")

    if n_total_cells == 0:
        print("  -> No cells found")
        continue

    # Check donor distribution
    donor_counts = adata.obs[mask]["hubmap_id"].value_counts()
    print(f"\nDonor distribution:")
    print(donor_counts)

    # Filter by min cells per donor
    keep_donors = donor_counts[donor_counts >= min_cells_per_donor]
    print(f"\nDonors with >= {min_cells_per_donor} cells: {len(keep_donors)}")
    if len(keep_donors) > 0:
        print(keep_donors)

    if len(keep_donors) < min_donors:
        print(f"  -> INSUFFICIENT: {len(keep_donors)} < {min_donors} donors")
    else:
        print(f"  -> SUFFICIENT: {len(keep_donors)} >= {min_donors} donors")
