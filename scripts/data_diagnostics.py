#!/usr/bin/env python3
"""Diagnose cell type labels and donor information in the dataset."""

import scanpy as sc
import pandas as pd

# Load the data
adata = sc.read_h5ad("data/processed/HT_pca_umap.h5ad")

print(f"Dataset shape: {adata.shape[0]} cells x {adata.shape[1]} genes")
print("\n" + "="*70)

# Check available columns
print("\nAvailable obs columns:")
print(adata.obs.columns.tolist())

# Check cell type labels
print("\n" + "="*70)
print("\nCell type labels (azimuth_label):")
if "azimuth_label" in adata.obs.columns:
    cell_type_counts = adata.obs["azimuth_label"].value_counts()
    print(cell_type_counts)
else:
    print("Column 'azimuth_label' not found!")

# Check donor information
print("\n" + "="*70)
print("\nDonor information (hubmap_id):")
if "hubmap_id" in adata.obs.columns:
    donor_counts = adata.obs["hubmap_id"].value_counts()
    print(f"Number of unique donors: {donor_counts.shape[0]}")
    print(donor_counts)
    
    # Cross-tabulation
    print("\n" + "="*70)
    print("\nCells per cell type per donor:")
    if "azimuth_label" in adata.obs.columns:
        crosstab = pd.crosstab(adata.obs["azimuth_label"], adata.obs["hubmap_id"])
        print(crosstab)
else:
    print("Column 'hubmap_id' not found!")
