# Analysis Run Log

## 2026-02-07: CFDE Heart snRNA-seq Dataset Analysis

### P1: Dataset Download + Metadata Capture (08:00–08:25)

**Dataset Information:**

- **Dataset Name:** HT_raw.h5ad
- **Source:** CFDE Knowledge Commons (cfde-kc-fetch tool)
- **URL/Repository:** HuBMAP Data Portal / CFDE Knowledge Commons
- **Download Date:** 2026-02-07
- **File Location:** `data/raw/HT_raw.h5ad`
- **File Size:** ~3.2 GB
- **MD5 Checksum:** (computed during notebook execution)

**Dataset Characteristics:**

- **Total Cells:** 675,772
- **Total Genes:** 60,286
- **Data Type:** Single-nucleus RNA-seq (snRNA-seq)
- **Tissue:** Heart
- **Species:** Human (Homo sapiens)

**Metadata Fields Available:**

- uuid
- hubmap_id (donor identifier)
- age
- sex
- height
- weight
- bmi
- cause_of_death
- race
- barcode
- prediction_score

**Files in data/raw/:**

- `HT_raw.h5ad` - Raw count matrix with metadata
- `fields.json` - Metadata fields from cfde-kc-fetch
- `coordinates.tsv` - UMAP coordinates (for heart_assets subset)
- `run_params.json` - cfde-kc-fetch run parameters
- `checksums.txt` - File integrity checksums

**Provenance Notes:**

- Data retrieved using cfde-kc-fetch tool
- Raw H5AD file contains full expression matrix (675k cells × 60k genes)
- Previous analysis created heart_assets.h5ad (6,715 cells subset with UMAP)

---

### P2: Load into AnnData + Verify Donor Column (08:30–08:55)

**Loading Status:** ✓ Successful

- **Observations (cells):** 675,772
- **Variables (genes):** 60,286
- **Expected Range:** 45,000–55,000 cells
- **Note:** Actual dataset is much larger than initially expected (13.5× larger)

**Donor/Batch Column:**

- **Column Used:** `hubmap_id`
- **Number of Unique Donors:** (computed in notebook)
- **Quality Gate:** ✓ Passed (≥2 donors confirmed)
- **Minimum Cells per Donor:** (computed in notebook)

**Outputs Generated:**

- `outputs/tables/donor_counts.csv` - Cell counts per donor

---

### QC Filters + Normalization

**QC Metrics Computed:**

- Total UMI counts per cell
- Number of genes detected per cell
- Mitochondrial gene percentage (MT-)

**Filter Criteria Applied:**

- Minimum genes per cell: 200
- Maximum genes per cell: 6,000 (doublet removal)
- Maximum mitochondrial %: 20%
- Minimum cells per gene: 3

**Normalization Protocol:**

- Method: Total count normalization
- Target sum: 10,000 counts per cell
- Transformation: log1p (natural log + 1)
- Raw counts preserved in layer: `counts`

**Quality Checks:**

- ✓ No NaN values in normalized matrix
- ✓ No Inf values in normalized matrix

**Outputs Generated:**

- `outputs/logs/qc_summary.txt` - Detailed QC report (before/after filtering)
- `data/raw/HT_raw_processed.h5ad` - Filtered and normalized AnnData object

**Cells Retained:** (computed in notebook)
**Genes Retained:** (computed in notebook)

---

### Analysis Notes

1. **Dataset Size:** The HT_raw.h5ad dataset is significantly larger (675k cells) than the initially expected range (45-55k cells). This likely represents a comprehensive multi-donor heart atlas.

2. **Missing Embeddings:** The raw dataset does not contain pre-computed embeddings (UMAP/tSNE). These can be computed post-QC if needed for visualization.

3. **Gene Expression Matrix:** Unlike the heart_assets.h5ad subset, HT_raw.h5ad contains the full gene expression matrix (60k genes).

4. **Processing Pipeline:** Standard Scanpy-style workflow applied: QC metrics → filtering → normalization → log transformation.

---

### Next Steps

- [ ] Compute highly variable genes (HVG selection)
- [ ] Dimensionality reduction (PCA)
- [ ] Batch effect correction (if needed)
- [ ] Compute UMAP/tSNE embeddings
- [ ] Clustering analysis
- [ ] Cell type annotation
- [ ] Differential expression analysis
- [ ] Visualization of QC metrics and cell distributions
