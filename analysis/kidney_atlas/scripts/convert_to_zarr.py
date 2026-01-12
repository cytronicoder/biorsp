#!/usr/bin/python
import sys

import scanpy as sc

adata = sc.read_h5ad(sys.argv[1])
del adata.raw
adata.var.index = adata.var.index.astype("str")
adata.obs.index = adata.obs.index.astype("str")
adata.var_names = adata.var_names.astype(str)
adata.X = adata.X.tocsc()
adata.write_zarr(sys.argv[2], [adata.shape[0], 10])
