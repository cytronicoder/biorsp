# Disease-Stratified Analysis: Practical Usage Examples

## KPMP Dataset Metadata

The KPMP (Kidney Precision Medicine Project) AnnData file uses these specific metadata labels:

| Property | Column Name | Available Values |
|----------|------------|------------------|
| Disease condition | `disease` or `disease_category` | `AKI`, `CKD`, `Healthy_living_donor`, `Healthy_stone_donor` |
| Cell type | `subclass.l1` | `TAL`, `PT`, `DCT`, `PC`, etc. |
| Donor ID | `donor_id` | Unique donor identifiers |

## Example 1: All Cells, Three Disease Conditions

Analyze all cells across healthy, acute, and chronic kidney disease:

```bash
python analysis/kidney_atlas/run_disease_stratified_analysis.py \
  --ref_data analysis/kidney_atlas/data/kpmp.h5ad \
  --outdir results/all_cells_disease_stratified \
  --n_workers 4
```

By default, this will analyze all three conditions found in the data:

- **Healthy reference** (Healthy_living_donor + Healthy_stone_donor combined)
- **Acute kidney injury** (AKI)
- **Chronic kidney disease** (CKD)

**Output structure:**

```
results/all_cells_disease_stratified/
├── healthy_reference/
│   ├── gene_scores.csv
│   ├── gene_classes.csv
│   ├── radar_plots/
│   └── embedding_plots/
├── acute_kidney_injury/
│   ├── gene_scores.csv
│   ├── ...
└── chronic_kidney_disease/
    ├── gene_scores.csv
    ├── ...
```

---

## Example 2: TAL Cells Only, Disease Comparison

Focus on Thick Ascending Limb (TAL) cells across disease conditions:

```bash
python analysis/kidney_atlas/run_disease_stratified_analysis.py \
  --ref_data analysis/kidney_atlas/data/kpmp.h5ad \
  --outdir results/tal_disease_stratified \
  --celltype_key subclass.l1 \
  --celltype_filter TAL \
  --n_workers 4
```

This replicates your original TAL analysis but now stratified by disease.

---

## Example 3: Using Exact KPMP Labels

If you want to explicitly specify which disease conditions to analyze:

```bash
python analysis/kidney_atlas/run_disease_stratified_analysis.py \
  --ref_data analysis/kidney_atlas/data/kpmp.h5ad \
  --outdir results/aki_vs_healthy \
  --diseases AKI Healthy_living_donor \
  --n_workers 4
```

**Note**: The script will normalize these to `acute_kidney_injury` and `healthy_reference` in the output folder names.

---

## Example 4: Using Case-Insensitive Shortcuts

For convenience, you can use lowercase shortcuts:

```bash
python analysis/kidney_atlas/run_disease_stratified_analysis.py \
  --ref_data analysis/kidney_atlas/data/kpmp.h5ad \
  --outdir results/disease_comparison \
  --diseases normal aki ckd \
  --n_workers 4
```

This is equivalent to using the exact labels but easier to type.

---

## Example 5: TAL Cells with Control Genes

Analyze TAL cells with specific control genes for quality checking:

```bash
python analysis/kidney_atlas/run_disease_stratified_analysis.py \
  --ref_data analysis/kidney_atlas/data/kpmp.h5ad \
  --outdir results/tal_with_controls \
  --celltype_key subclass.l1 \
  --celltype_filter TAL \
  --controls "SLC12A1,UMOD,EGF,SLC5A2" \
  --n_workers 4
```

Control genes will be:

- Processed first
- Highlighted in plots with special markers
- Useful for validating expected TAL markers

---

## Example 6: Quick Pilot Test

Test the workflow on a small subset before running the full analysis:

```bash
python analysis/kidney_atlas/run_disease_stratified_analysis.py \
  --ref_data analysis/kidney_atlas/data/kpmp.h5ad \
  --outdir results/pilot_test \
  --subsample 1000 \
  --max_genes 50 \
  --n_permutations 50 \
  --n_workers 2
```

This will:

- Use 1000 cells per disease (faster)
- Analyze only top 50 genes
- Use 50 permutations (less precise p-values)
- Run in ~5-10 minutes instead of hours

---

## Example 7: Full Production Analysis

Complete analysis with all genes and robust statistics:

```bash
python analysis/kidney_atlas/run_disease_stratified_analysis.py \
  --ref_data analysis/kidney_atlas/data/kpmp.h5ad \
  --outdir results/production_full \
  --celltype_key subclass.l1 \
  --celltype_filter TAL \
  --max_genes 2000 \
  --n_permutations 1000 \
  --do_genegene \
  --n_workers 8
```

This will:

- Analyze up to 2000 genes per disease
- Use 1000 permutations for precise p-values
- Include gene-gene interaction analysis
- Utilize 8 CPU cores
- Take several hours to complete

---

## Example 8: Multiple Cell Types in Parallel

Analyze several cell types using a loop:

```bash
# Run analysis for TAL, PT (proximal tubule), and DCT (distal convoluted tubule)
for celltype in TAL PT DCT; do
  echo "Processing $celltype..."
  python analysis/kidney_atlas/run_disease_stratified_analysis.py \
    --ref_data analysis/kidney_atlas/data/kpmp.h5ad \
    --outdir results/${celltype}_disease \
    --celltype_key subclass.l1 \
    --celltype_filter $celltype \
    --max_genes 500 \
    --n_workers 4
done
```

---

## Comparing Results Across Diseases

After running any of the above analyses, compare results using:

```bash
python analysis/kidney_atlas/compare_disease_results.py \
  --input results/tal_disease_stratified \
  --output results/tal_disease_comparison \
  --reference healthy_reference
```

This generates:

- Heatmaps showing RSP score changes across diseases
- Scatter plots comparing disease pairs
- Statistical summaries of disease-specific effects
- Markdown report with top changing genes

---

## Understanding Output Folders

Each disease folder contains:

| File/Folder | Description |
|-------------|-------------|
| `gene_scores.csv` | RSP scores, p-values, classifications for all genes |
| `gene_classes.csv` | Summary of gene pattern classifications |
| `metadata.json` | Analysis parameters and cell counts |
| `radar_plots/` | Radar plots for top genes (HTML + PNG) |
| `embedding_plots/` | UMAP/embedding visualizations |
| `gene_pairs/` | Gene-gene interaction results (if `--do_genegene`) |

---

## Troubleshooting

### "No cells found matching filter"

Check available cell types in your data:

```python
import anndata as ad
adata = ad.read_h5ad("analysis/kidney_atlas/data/kpmp.h5ad")
print(adata.obs['subclass.l1'].unique())
```

### "Disease column not found"

Manually specify the column:

```bash
--disease_key disease_category
```

### Out of memory errors

Reduce computational load:

```bash
--subsample 5000 \
--max_genes 200 \
--n_permutations 100 \
--n_workers 2
```

### Very slow execution

Enable parallel processing:

```bash
--n_workers 8  # Use 8 CPU cores
```

Skip gene-gene analysis for faster results:

```bash
# Remove --do_genegene flag
```

---

## Next Steps

1. **Start with a pilot test** (Example 6) to validate the workflow
2. **Run cell type-specific analysis** (Example 2) for focused results
3. **Compare across diseases** using `compare_disease_results.py`
4. **Scale to production** (Example 7) when ready

For more details, see:

- [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - Command cheat sheet
- [compare_disease_results.py](compare_disease_results.py) - Comparison tool
- [run_tal_analysis.py](run_tal_analysis.py) - Original single-condition analysis
