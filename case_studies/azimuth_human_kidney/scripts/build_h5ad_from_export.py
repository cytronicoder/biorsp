#!/usr/bin/env python3
import sys
from pathlib import Path

import anndata as ad
import pandas as pd
import scipy.io
import scipy.sparse as sp

if len(sys.argv) < 3:
    print("Usage: build_h5ad_from_export.py <export_dir> <out_h5ad>")
    sys.exit(1)
export_dir = Path(sys.argv[1])
out_path = Path(sys.argv[2])

# load files
data_mtx = export_dir / "data.mtx"
counts_mtx = export_dir / "counts.mtx"
obs_csv = export_dir / "obs.csv"
var_csv = export_dir / "var.csv"

if not data_mtx.exists():
    raise FileNotFoundError(data_mtx)

mat = scipy.io.mmread(str(data_mtx))
# mat is genes x cells (from Seurat), convert to cells x genes
if sp.issparse(mat):
    X = mat.T.tocsr()
else:
    X = mat.T

obs = pd.read_csv(obs_csv, index_col=0)
var = pd.read_csv(var_csv, index_col=0)
# Determine gene names robustly
if var.shape[1] == 0:
    var_names = var.index.astype(str)
elif "gene" in var.columns:
    var_names = var["gene"].astype(str)
else:
    var_names = var.iloc[:, 0].astype(str)
var.index = var_names

adata = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=var.index))

if counts_mtx.exists():
    counts = scipy.io.mmread(str(counts_mtx))
    if sp.issparse(counts):
        layers_counts = counts.T.tocsr()
    else:
        layers_counts = counts.T
    adata.layers["counts"] = layers_counts

# Load embeddings
for csv_file in export_dir.glob("*.csv"):
    if csv_file.name in ["obs.csv", "var.csv"]:
        continue
    # Assume other CSVs are embeddings
    emb_name = csv_file.stem
    try:
        emb_df = pd.read_csv(csv_file, index_col=0)
        # Ensure index matches obs
        if len(emb_df) == len(adata):
            # Align with adata.obs_names
            emb_df = emb_df.reindex(adata.obs_names)
            # Use X_ prefix for convention if not present
            key = f"X_{emb_name}" if not emb_name.startswith("X_") else emb_name
            adata.obsm[key] = emb_df.values
            print(f"Added embedding {emb_name} as {key}")
    except Exception as e:
        print(f"Failed to add embedding {emb_name}: {e}")

# Save
out_path.parent.mkdir(parents=True, exist_ok=True)

adata.write_h5ad(str(out_path))
print("Wrote", out_path)
