# Azimuth human_kidney (imported)

This folder contains an imported copy of the Azimuth `human_kidney` reference files and supporting scripts.

What I imported
- The upstream folder `human_kidney` from https://github.com/satijalab/azimuth-references at commit b8b07dcdfcc09816a85aad07362e7bad4de03976
- The repository's `scripts/` folder (conversion and setup utilities)
- External reference data files from the upstream `links/dropbox_links.txt` were downloaded into `reference/data/` (these are large; see list below)

Files downloaded into `reference/data/`:
- kpmp_counts.mtx, kpmp_features.tsv, kpmp_cells.tsv, kpmp_metadata.csv (KPMP dataset)
- lake_counts.mtx, lake_features.tsv, lake_cells.tsv, lake_metadata.csv (LAKE dataset)

Notes
- I added an `export_for_python.R` script to export RDS contents (matrices and metadata) into CSV/MTX files and a helper Python script `build_h5ad_from_export.py` that assembles an `h5ad` from these exported files. I used these to produce a `ref.h5ad` in `reference/`.
- The KPMP dataset and the generated `ref.h5ad` do not contain standard Azimuth cell-type annotation keys (e.g., `annotation.l2`), so the notebook's TAL detection may fall back to marker heuristics (which require known marker genes to be present).
- The files downloaded are large (many hundreds of MBs to multiple GBs); consider removing or archiving them if you don't want them stored in the repository.

Provenance
- Source: https://github.com/satijalab/azimuth-references/tree/master/human_kidney
- Commit: b8b07dcdfcc09816a85aad07362e7bad4de03976

How to reproduce the `ref.h5ad` locally
1. Install R packages `Seurat` and `SeuratDisk` (or use the provided `export_for_python.R` + `build_h5ad_from_export.py` pipeline)
2. Run `Rscript scripts/setup_demo.R reference/ref.Rds` to create a demo `ref.Rds` if desired
3. Run `Rscript scripts/convert_to_h5ad.R ref.Rds fullref.Rds out.h5ad` or use the Python helper to assemble from exported files

If you want, I can (a) make the `setup_demo.R` pipeline produce richer metadata, or (b) run a marker-driven clustering to identify TAL cells automatically — tell me which you'd prefer.
