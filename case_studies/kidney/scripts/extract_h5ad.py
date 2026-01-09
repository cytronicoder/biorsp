#!/bin/python3

import numpy as np
import scanpy as sc
from scipy import io, sparse

adata = sc.read("data/Mature_Full_v3.h5ad")
adata.obs.to_csv("data/kidney_demo_metadata.csv")
np.savetxt("data/kidney_demo_features.csv", np.array(adata.var.index), delimiter=",", fmt="%s")
np.savetxt("data/kidney_demo_cells.csv", np.array(adata.obs.index), delimiter=",", fmt="%s")
mat = adata.layers["counts"].todense()
io.mmwrite("data/kidney_demo_expression.mtx", sparse.csr_matrix(mat))
