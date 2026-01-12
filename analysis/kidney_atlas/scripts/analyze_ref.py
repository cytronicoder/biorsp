import re
import sys
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp

if len(sys.argv) < 2:
    print("Usage: analyze_ref.py <ref.h5ad>")
    sys.exit(1)
ref_path = Path(sys.argv[1])
adata = ad.read_h5ad(str(ref_path))
print("Loaded:", ref_path, "shape:", adata.shape)


def get_umis(adata):
    candidates = [
        c
        for c in adata.obs.columns
        if c.lower().startswith("ncount_") or "total_counts" in c.lower() or "n_counts" in c.lower()
    ]
    if len(candidates) > 0:
        col = candidates[0]
        print("Using UMI column:", col)
        return pd.to_numeric(adata.obs[col], errors="coerce").fillna(0).astype(float).values

    if sp.issparse(adata.X):
        return np.asarray(adata.X.sum(axis=1)).reshape(-1)
    else:
        return np.asarray(adata.X.sum(axis=1)).reshape(-1)


umis = get_umis(adata)
print(
    "UMI stats (min, median, mean, max):",
    np.min(umis),
    np.median(umis),
    np.mean(umis),
    np.max(umis),
)

outdir = Path("examples")
outdir.mkdir(exist_ok=True)
plt.figure(figsize=(6, 3))
plt.hist(umis, bins=50, color="C0", alpha=0.8)
plt.xlabel("UMI counts")
plt.ylabel("Cells")
plt.title("UMI distribution")
plt.tight_layout()
plt.savefig(outdir / "umi_distribution.png")
print("Wrote umi_distribution.png")


def is_tal_label(s: str) -> bool:
    if s is None:
        return False
    s = str(s).lower()
    if "thick" in s and "ascending" in s:
        return True
    if "thick ascending" in s:
        return True
    import re

    return bool(
        re.search(r"\bcortical .*thick.*ascending\b", s)
        or re.search(r"\bmedullary .*thick.*ascending\b", s)
    )


candidate_keys = [
    k for k in ["annotation.l1", "annotation.l2", "annotation.l3"] if k in adata.obs.columns
]
print("Candidate keys:", candidate_keys)

tal_suggestions = []
for k in candidate_keys:
    vc = adata.obs[k].astype(str).value_counts()
    matches = [str(c) for c in vc.index if is_tal_label(c)]
    if matches:
        tal_suggestions.extend([(k, m) for m in matches])

if tal_suggestions:
    chosen_key = tal_suggestions[0][0]
    proposed = list(dict.fromkeys([m for (_, m) in tal_suggestions]))
    print("Auto-selecting celltype_key =", chosen_key)
    print("Auto-setting tal_labels =", proposed)
    celltype_key = chosen_key
    tal_labels = proposed
else:
    print(
        "No TAL labels detected automatically; attempting marker-based detection using UMOD and SLC12A1"
    )
    celltype_key = None
    tal_labels = []

mask = None
if celltype_key is not None:
    pattern = "|".join([re.escape(str(x)) for x in tal_labels]) if len(tal_labels) > 0 else ""
    mask = adata.obs[celltype_key].astype(str).str.contains(pattern, case=False, na=False)
    print("Found TAL count by annotation:", int(mask.sum()))

if mask is None or mask.sum() == 0:
    genes = list(adata.var_names.astype(str))
    markers = ["UMOD", "SLC12A1"]
    found = [g for g in markers if g in genes]
    print("Markers found in var_names:", found)
    if len(found) == 0:
        print("No canonical TAL markers found; cannot auto-detect TAL cells.")
        mask = np.zeros(adata.n_obs, dtype=bool)
    else:
        expr = {}
        for g in found:
            x = adata[:, g].X
            x = np.asarray(x.todense()).reshape(-1) if sp.issparse(x) else np.asarray(x).reshape(-1)
            expr[g] = x

        m_any = np.zeros(adata.n_obs, dtype=bool)
        for x in expr.values():
            m_any = m_any | (x > 0)

        mask = np.zeros(adata.n_obs, dtype=bool) if m_any.sum() == 0 else m_any
        print("Found TAL count by marker heuristic:", int(mask.sum()))

if mask.sum() == 0:
    print("No TAL cells found after heuristics; aborting TAL-specific stats")
else:
    adata_tal = adata[mask].copy()

    if sp.issparse(adata_tal.X):
        is_expr = adata_tal.X > 0
        prevalence = np.asarray(is_expr.sum(axis=0)).reshape(-1) / float(adata_tal.n_obs)
        sums = np.asarray(adata_tal.X.sum(axis=0)).reshape(-1)
        means = sums / float(adata_tal.n_obs)
    else:
        x = np.asarray(adata_tal.X)
        prevalence = (x > 0).sum(axis=0) / float(adata_tal.n_obs)
        means = x.mean(axis=0)
    genes = np.asarray(adata_tal.var_names)
    df_stats = pd.DataFrame({"gene": genes, "prevalence": prevalence, "mean_expr": means})
    df_stats.sort_values("mean_expr", ascending=False, inplace=True)
    out = outdir / "tal_gene_stats.csv"
    df_stats.to_csv(out, index=False)
    print("Wrote", out)

    try:
        emb = None
        for k in adata.obsm:
            if "umap" in k.lower():
                emb = adata.obsm[k][:, :2]
                break
        if emb is None and "UMAP_1" in adata.obs.columns and "UMAP_2" in adata.obs.columns:
            emb = adata.obs[["UMAP_1", "UMAP_2"]].values
        if emb is not None:
            plt.figure(figsize=(6, 6))
            tal_mask = mask
            plt.scatter(emb[~tal_mask, 0], emb[~tal_mask, 1], c="lightgray", s=6, alpha=0.5)
            plt.scatter(emb[tal_mask, 0], emb[tal_mask, 1], c="red", s=10, alpha=0.9)
            plt.axis("off")
            plt.title("Embedding: TAL highlighted")
            plt.savefig(outdir / "tal_embedding.png")
            print("Wrote tal_embedding.png")
    except Exception as e:
        print("Could not create embedding plot:", e)

print("Analysis complete.")
