# Outputs And Reproducibility

## Output Root

```text
outputs/heart_case_study/<run_name>/
  metadata.json
  run_metadata.json
  logs/runlog.md
  tables/
    gene_summary.csv
    marker_panel_found_missing.csv
    cluster_celltype_counts.csv
  plots/
    qc/
    meta/
    pairs/
    rsp/
  hierarchy/
    hierarchy_summary.json
    global/
    mega/
    clusters/
```

## Required Tables

- `tables/gene_summary.csv`
  Includes: `scope`, `gene`, `panel_group`, `auto_gene`, `prevalence`, `anisotropy`, `peak_direction_rad`, `p_T`, `q_T`, `moran_continuous`, `moran_binary`, `qc_risk`, `qc_like_flag`.
- `tables/marker_panel_found_missing.csv`
  Includes found/missing marker resolution details and auto-gene labels.
- `tables/cluster_celltype_counts.csv`
  Global contingency table for cluster/celltype counts (when keys are available).

## Reproducibility Controls

Determinism is controlled by:

- fixed `seed`
- fixed `bins` and `n_perm`
- deterministic mega split (`KMeans(..., random_state=seed)`)
- deterministic plotting style and file naming

## Run Comparison Checklist

1. Compare `run_metadata.json` and `metadata.json` (`versions`, parameters, resolved keys).
2. Compare `hierarchy/hierarchy_summary.json` (processed/skipped scopes).
3. Compare `tables/gene_summary.csv` and `tables/marker_panel_found_missing.csv`.
4. Check `logs/runlog.md` warnings and inference-limitation notes.
