#!/usr/bin/env python3
"""Quick script to explore KPMP metadata structure."""

import anndata as ad

print("Loading KPMP data...")
adata = ad.read_h5ad("data/kpmp.h5ad")

print(f"\nDataset: {adata.n_obs} cells × {adata.n_vars} genes")
print("\nAll metadata columns:")
for col in adata.obs.columns:
    print(f"  - {col}")

print("\n" + "=" * 80)
print("DISEASE/CONDITION COLUMNS")
print("=" * 80)

disease_cols = [
    c
    for c in adata.obs.columns
    if any(x in c.lower() for x in ["disease", "condition", "phenotype", "status"])
]
for col in disease_cols:
    print(f"\nColumn: '{col}'")
    print(f"Unique values: {adata.obs[col].nunique()}")
    value_counts = adata.obs[col].value_counts()
    for val, count in value_counts.items():
        print(f"  {val}: {count:,} cells")

print("\n" + "=" * 80)
print("CELL TYPE COLUMNS")
print("=" * 80)

celltype_cols = [
    c
    for c in adata.obs.columns
    if any(x in c.lower() for x in ["cell", "type", "annotation", "subclass", "cluster"])
]
for col in celltype_cols[:5]:
    print(f"\nColumn: '{col}'")
    print(f"Unique values: {adata.obs[col].nunique()}")
    if adata.obs[col].nunique() <= 30:
        value_counts = adata.obs[col].value_counts()
        for val, count in value_counts.head(20).items():
            print(f"  {val}: {count:,} cells")
    else:
        print("  (Too many to display)")

print("\n" + "=" * 80)
print("DONOR/SAMPLE COLUMNS")
print("=" * 80)

donor_cols = [
    c
    for c in adata.obs.columns
    if any(x in c.lower() for x in ["donor", "sample", "patient", "library"])
]
for col in donor_cols:
    print(f"\nColumn: '{col}'")
    n_unique = adata.obs[col].nunique()
    print(f"Unique values: {n_unique}")
    if n_unique <= 10:
        print(f"Values: {list(adata.obs[col].unique())}")

print("\n" + "=" * 80)
print("EMBEDDINGS")
print("=" * 80)
print(f"Available in obsm: {list(adata.obsm.keys())}")
