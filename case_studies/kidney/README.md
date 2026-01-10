# Spatial Gene Analysis: TAL Cells in Human Kidney

Understand which genes are spatially organized in the **Thick Ascending Limb (TAL)** cells of the human kidney. This analysis identifies canonical TAL markers and discovers genes with interesting spatial patterns.

## What You Get

Running this analysis produces:

1. **Ranked Gene Results Table** (`tal_gene_results.csv`)
   - Coverage: What fraction of cells express each gene?
   - Spatial Organization: Is expression clustered or uniform?
   - Statistical significance with multiple testing correction
   - Gene archetypes for easy interpretation

2. **Archetype Landscape Plot** (`tal_archetypes_scatter.png`)
   - Visual overview of all genes plotted by coverage vs spatial organization
   - Shows which genes are localized, uniform, rare-but-clustered, or sparse

3. **Per-Gene Visualizations** (`plots/`)
   - Cell embedding with expression overlay
   - Spatial distribution shown as a radar plot
   - Summary statistics box

4. **Gene-Gene Relationships** (optional via `--do_genegene`)
   - Identifies genes with similar spatial patterns
   - Helps find functional gene groups

## Quick Start

### Prerequisites

```bash
pip install biorsp anndata scanpy pandas numpy scipy matplotlib tqdm
```

### Fast Pilot Run (< 2 minutes)

Test the pipeline with just a handful of genes to make sure everything works:

```bash
python run_tal_analysis.py \
  --ref_data data/kpmp.h5ad \
  --outdir results/tal_pilot \
  --controls "SLC12A1,UMOD,EGF" \
  --max_genes 10 \
  --n_permutations 100
```

Expect this to complete in 1-2 minutes.

### Publication-Grade Run (30-60 minutes)

Comprehensive analysis with hundreds of genes and gene-gene relationships:

```bash
python run_tal_analysis.py \
  --ref_data data/kpmp.h5ad \
  --outdir results/tal_full \
  --controls "SLC12A1,UMOD,EGF" \
  --max_genes 500 \
  --n_permutations 1000 \
  --do_genegene \
  --n_workers 4
```

This gives you robust statistics and optional gene-gene discovery. Expect 30-60 minutes.

## Understanding the Output

### The Two Main Scores

| Score | What It Measures | How to Interpret |
|-------|-----------------|------------------|
| **Coverage** ($C_g$) | How many cells express this gene? | 0 = nobody expresses it, 1 = everybody expresses it |
| **Spatial Score** ($S_g$) | Is expression clustered in regions, or spread out? | 0 = evenly distributed, higher values = more clustered |

**Simple Example**: If SLC12A1 has C_g=0.9 and S_g=0.25, that means ~90% of TAL cells express it, and the expression shows moderate spatial clustering (not perfectly uniform).

### Four Gene Types: The Archetype Quadrants

Genes naturally fall into four categories based on coverage and spatial organization:

```
          ┌─── HIGH spatial organization ───┐
          │                                 │
    Niche Biomarkers         Localized Programs
    (rare but clustered)     (common AND clustered)
          │                         │
          │    Gene Space           │
          │      Divided            │
          │      by Median          │
          │                         │
    Sparse Genes            Housekeeping Genes
    (rare, scattered)       (common, uniform)
          │                         │
          │                         │
          └─── LOW spatial organization ───┘

     LOW coverage              HIGH coverage
      (few cells)              (many cells)
```

### What Each Quadrant Means

**Localized Programs** (High Coverage + High Spatial Score)
- These genes are the cell type's "signature programs"
- Found in most cells of the tissue region, but expression clusters in specific zones
- Example: SLC12A1 (Na-K-2Cl cotransporter) in proximal TAL
- **Interpretation**: Core TAL functional program that varies by micro-location

**Housekeeping Genes** (High Coverage + Low Spatial Score)
- Widely expressed across all cells, no location preference
- Likely involved in basic cellular function needed everywhere
- Example: EEF1A1 (elongation factor) or ribosomal proteins
- **Interpretation**: General cell maintenance—less informative about tissue organization

**Niche Biomarkers** (Low Coverage + High Spatial Score)
- Expressed in just a fraction of cells, but those cells cluster together
- Potential markers of specialized sub-populations or functional niches
- Could indicate cell-cell interaction zones
- **Interpretation**: May reveal micro-anatomical structure or cell state transitions

**Sparse Genes** (Low Coverage + Low Spatial Score)
- Rarely expressed and scattered randomly
- Likely either noise, very rare cell states, or genes needing specific stimuli
- **Interpretation**: Less reliable for understanding tissue organization

### A Word on Confidence: `coverage_bg`

The `coverage_bg` column tells you how confident we can be in a gene's spatial score:

- **>0.9** (green flag 🟢): We could evaluate the gene across nearly all directions. Confidence: **high**
- **0.7–0.9** (yellow flag 🟡): Some blind spots due to sampling, but generally reliable
- **<0.7** (red flag 🔴): Limited angular coverage. Spatial score may be misleading
- **<0.5**: We'd flag this with a warning; take the spatial score with a grain of salt

**Why does this matter?** If your tissue region has an odd shape or the cells aren't evenly distributed, we can't measure spatial organization in all directions equally. `coverage_bg` tells you where we have good data.

## Output Files

```
results/tal_full/
├── tal_gene_results.csv        # Complete ranked gene table
├── tal_top_genes.txt           # Top genes for quick reference
├── tal_archetypes_scatter.png  # Primary story figure
├── tal_gene_pairs.csv          # Pairwise relationships (if --do_genegene)
├── tal_gene_pairs_heatmap.png  # Copattern heatmap (if --do_genegene)
├── run_metadata.json           # Full provenance and configuration
└── plots/
    ├── plot_SLC12A1.png       # Per-gene radar plots
    ├── plot_UMOD.png
    └── ...
```

### Results Table Columns Explained

| Column | What It Means |
|--------|---------------|
| `gene` | Gene ID from your dataset |
| `gene_symbol` | Gene name (e.g., SLC12A1) |
| `coverage_expr` | Coverage score: fraction of cells expressing this gene |
| `spatial_score` | Spatial score: how clustered vs uniform is the expression? |
| `spatial_sign` | Direction (+1 = one pattern, -1 = opposite pattern, 0 = balanced) |
| `r_mean_bg` | Average radar intensity across reliably measured directions |
| `coverage_bg` | **Confidence metric**: What fraction of directions could we measure? |
| `coverage_fg` | Technical metric: measurement quality in foreground |
| `p_value` | How likely is this pattern by random chance? |
| `q_value` | FDR-corrected p-value (accounts for testing many genes) |
| `archetype` | Archetype: localized_program, housekeeping_uniform, niche_biomarker, or sparse_presence |
| `warnings` | Any quality flags (e.g., "low confidence in spatial score") |

## Key Parameters

### Geometry: Tuning Angular Resolution

| Parameter | Default | What It Does |
|-----------|---------|---------------|
| `--B` | 72 | Divide the tissue into how many angular "slices"? (72 = 5° each) |
| `--delta_deg` | 60 | Sector "width" in degrees—how much angular smoothing? |

⚠️ **Note on `--delta_deg`**: 
- Use **30–90°** to detect localized "wedge" patterns (most common)
- **180°** means comparing opposite hemispheres—good for coarse tissue structure but loses fine spatial detail
- Larger values = smoother but less detailed spatial maps

### Expression Thresholds: What Counts as "Expressed"?

| Parameter | Default | What It Controls |
|-----------|---------|------------------|
| `--expr_threshold_mode` | detect | How to determine the expression threshold: *detect* (automatic), *fixed* (you specify), or *nonzero_quantile* |
| `--expr_threshold_value` | — | If using `fixed` mode, set it here |
| `--foreground_quantile` | 0.90 | Which cells count as "expressed" for the radar plot? (0.90 = top 10% of expressers) |
| `--empty_fg_policy` | zero | If a sector has no expressed cells, what do we report? *zero* (no signal) or *nan* (unmeasurable)? |

### Statistics: P-Values and Multiple Testing

| Parameter | Default | What It Does |
|-----------|---------|---------------|
| `--n_permutations` | 200 | How many randomizations to estimate background? (200 = fast, 1000+ = publication-ready) |
| `--fdr_cut` | 0.05 | Significance threshold after correcting for testing many genes |
| `--c_cut` | 0.10 | Coverage threshold for archetype classification (e.g., genes with C_g > 0.10 are "common") |
| `--s_cut` | auto | Spatial score threshold (auto = derived from p-values) |

## Known TAL Marker Genes (What to Expect)

These genes are classic TAL signatures—you should see high coverage and good spatial signal:

- **SLC12A1** (NKCC2): The definitive TAL marker. Na-K-2Cl cotransporter (active transport). **Expect**: Very high C_g, strong spatial organization
- **UMOD** (uromodulin/Tamm-Horsfall): TAL-specific secreted protein. **Expect**: High C_g, moderate to high S_g
- **CLDN16** (claudin-16): Tight junction protein crucial for TAL paracellular transport. **Expect**: High C_g
- **CLDN10** (claudin-10): Another TAL tight junction component. **Expect**: High C_g
- **EGF** (epidermal growth factor): Growth factor. **Expect**: High C_g, possibly moderate S_g depending on signaling state

**Pro tip**: Use these as sanity checks. If SLC12A1 comes back with low coverage or spatial score, something may be wrong with your data or cell type selection.

## Data Sources

### KPMP Dataset
- **Source**: Kidney Precision Medicine Project
- **Reference**: Lake et al. (2021). Nature. https://doi.org/10.1038/s41586-021-03549-5

### Azimuth Reference
- **Provenance**: https://github.com/satijalab/azimuth-references/tree/master/human_kidney
- **Reference**: Hao et al. (2021). Cell. https://doi.org/10.1016/j.cell.2021.04.048

## Troubleshooting

### "No suitable embedding found"
**Problem**: Script can't find a 2D embedding (UMAP, t-SNE, etc.)

**Solution**: Check what embeddings your file has, then tell the script explicitly:
```bash
# First, see what's available
python -c "import anndata; a = anndata.read_h5ad('data/kpmp.h5ad'); print(a.obsm.keys())"

# Then use the right key
python run_tal_analysis.py ... --embedding_key X_your_custom_key
```

### "No cells match labels"
**Problem**: Script couldn't find your cell type label in the metadata.

**Solution**: Check what metadata columns exist and what the TAL label is:
```python
import anndata
adata = anndata.read_h5ad("data/kpmp.h5ad")

# See all metadata columns
print(adata.obs.columns.tolist())

# See cell types in the column you think contains them
print(adata.obs["subclass.l1"].unique())
```

Then pass the correct labels:
```bash
python run_tal_analysis.py ... --celltype_key your_column_name --tal_labels "Label1" "Label2"
```

### All genes have low `coverage_bg` (confidence)
**Problem**: We can't measure spatial organization reliably across all directions.

**Possible causes**:
- Cell subset is too small or clustered in one region
- Embedding is sparse or elongated
- Wrong cell type labels selected

**Solutions**:
1. Try a larger sample: `--subsample 5000` (if you're subsampling)
2. Check your cell type selection—are these really TAL cells?
3. Visualize the embedding manually to see if TAL cells fill the space or cluster in one corner
4. Use a denser/higher-resolution embedding if available

### Results seem noisy or inconsistent
**Problem**: P-values or rankings look unstable between runs.

**Solution**: You're probably using too few permutations (randomizations). Increase it:
```bash
python run_tal_analysis.py ... --n_permutations 1000
```

Default is 200 for speed. For publication, use 1000+.

## How Much Computing Power Do I Need?

| Dataset Size | RAM | Typical Runtime | Recommendation |
|--------------|-----|-----------------|------------------|
| <10K cells | 4–6 GB | 5–10 min | Laptop (no subsample needed) |
| 10–50K cells | 8–16 GB | 10–30 min | Workstation or small cluster |
| 50K+ cells | 16–32 GB | 30–120 min | Use `--subsample` for testing |

**For testing new cell types**: Always start with a subsample:
```bash
python run_tal_analysis.py ... --subsample 5000 --max_genes 20 --n_permutations 100
```

Once you're happy, run full analysis.

## Citation

If you use this case study or the BioRSP framework, please cite:

```
[BioRSP citation - to be updated]
```

## Questions or Issues?

**Got unexpected results?** Check the troubleshooting section above.

**Found a bug?** Please open an issue on GitHub with:
- Your command line
- Error message (if any)
- Dataset info (# cells, # genes, embedding type)

**New analysis ideas?** We'd love to hear about successful applications of this workflow to other cell types!
