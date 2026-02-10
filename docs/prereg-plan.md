# BioRSP preregistration-style evaluation plan

## Objective

Quantify directional enrichment using BioRSP (E(phi), E_max, phi_max) within cell-type strata and corroborate with graph autocorrelation (Moran’s I) while stress-testing against embedding instability, ambient RNA, dropout threshold artifacts, batch/integration distortion, and donor-driven pseudoreplication. Donor is the unit of inference.

## Experimental units and strata

- Primary unit: donor (or donor × condition if applicable).
- Primary strata (inferential): Ventricular cardiomyocytes, Fibroblasts, Vascular mural (Pericyte + Smooth muscle), Endothelial subtypes (Capillary, Arterial, Venous, Endocardial, Lymphatic), Myeloid (Macrophage, Monocyte/cDC), Lymphoid (T, NK, B, Mast).
- Secondary strata (exploratory): Mesothelial, Adipocyte, Neuronal if minimum-data rule met; Atrial CM exploratory; ILC ignored.
- Minimum-data rule: a stratum is inferential only if ≥3 donors and ≥200 cells per donor; otherwise descriptive plots only, no significance claims.

## Hypotheses

H1: Canonical within-stratum markers show significant directional enrichment (E_max) with donor-stratified p-values and positive Moran’s I above random-gene baseline.
H2: Donor-aware null prevents pseudoreplication; donor-skewed synthetic signals lose significance under donor-strat permutation and/or donor jackknife.
H3: Robust signals are stable to embedding perturbations (≥5 UMAP seeds + PCA2D) with low circular SD of phi_max and stable E_max ranks.
H4 (optional): Ambient-prone/stress genes do not dominate after QC/correction; if they appear, they fail donor-aware or Moran corroboration checks.

## Phase A — Preprocessing & QA

- QC per donor with thresholds (min genes, min counts, max mito fraction).
- Ambient RNA handling: if correction tools unavailable, generate ambient-flag list and run stress QC mode with relaxed thresholds.
- Normalization: log-normalized expression for Moran’s I and module scores; BioRSP foreground default expr>0.
- Deliverable: preprocessing report (CSV+Markdown) with per-donor metrics before/after QC.

## Phase B — Core BioRSP per stratum

For each inferential stratum:

- Build stratum AnnData, neighbors graph (connectivities), and embeddings (UMAP with 5 seeds + PCA2D).
- For each feature (gene/module): compute E(phi), E_max, phi_max for each embedding and thresholds; Moran’s I on graph; donor-strat permutation on E_max (≥1000 perms for final).

## Phase C — Robustness & uncertainty

- Donor jackknife on reference embedding (seed0) to estimate E_max uncertainty and donor sensitivity.
- Threshold sensitivity (t=0, q90, q95) and embedding stability (phi_max circular SD across embeddings).
- Graph corroboration: Moran’s I compared to random-gene baseline.

## Phase D — Falsification stress tests

- Synthetic null calibration with donor-preserving random foregrounds; expect uniform p-values and ≤5% significant at alpha=0.05 after correction.
- Donor-driven artifact simulation: signal in one donor; donor-strat permutation removes significance vs global shuffle.
- Embedding instability test: phi_max across UMAP seeds + PCA2D.
- Multi-modal probe: detect >1 local maximum in E(phi) and report peak angles.
- Ambient probe: evaluate ambient-prone genes for robustness failure.

## Phase E — Baselines and ablations

- Moran’s I-only ranking and overlap with BioRSP hits.
- Minimal DE baseline via leiden + rank_genes_groups for overlap assessment.
- Ablation: global shuffle vs donor-strat permutation; cleaned vs stress QC modes.

## Decision rules (robust hit)

A feature is a robust directional signal if all are met:

- BH-FDR ≤ 0.05 for donor-strat p_perm on E_max within stratum.
- phi_max stability across embeddings: circular SD < 20–30°.
- Moran’s I above random baseline (>95th percentile).
- Donor jackknife robustness (no single donor drives signal).

If synthetic null calibration fails (inflated small p-values), halt with guidance.

## Deliverables

- outputs/results/biorsp_stratum_results.csv
- outputs/results/null_calibration.csv (+ QQ plots/CSV)
- outputs/figures/ for Ventricular CM, Fibroblast, Capillary EC (UMAP overlays, RSP polar plots, null histograms, phi stability, QQ plots)
- outputs/results/preprocessing_report.csv/.md (per-donor QC + detection rates)
- outputs/results/batch_summary.csv if batch metadata exists
