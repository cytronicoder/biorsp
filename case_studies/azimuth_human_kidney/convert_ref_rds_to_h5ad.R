#!/usr/bin/env Rscript
# Robust conversion script using intermediate export to avoid SeuratDisk/HDF5 issues.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  cat("Usage: convert_ref_rds_to_h5ad.R <ref.Rds> <fullref.Rds> <out.h5ad>\n")
  # We accept fullref.Rds for compatibility but will use ref.Rds as the primary source
  # unless fullref is needed. For this robust version, we'll just convert ref.Rds.
  quit(status = 1)
}

ref_rds <- args[1]
# fullref_rds <- args[2] # Ignored in this robust version
out_h5ad <- args[3]

# Paths to helper scripts
script_dir <- file.path("case_studies", "azimuth_human_kidney", "scripts")
export_script <- file.path(script_dir, "export_for_python.R")
build_script <- file.path(script_dir, "build_h5ad_from_export.py")

if (!file.exists(export_script)) {
  stop(sprintf("Export script not found at %s", export_script))
}
if (!file.exists(build_script)) {
  stop(sprintf("Build script not found at %s", build_script))
}

# Temporary directory for export
temp_dir <- tempfile(pattern = "azimuth_export_")
dir.create(temp_dir)
cat(sprintf("Using temporary directory: %s\n", temp_dir))

# Step 1: Export from R
cat("Step 1: Exporting data from R...\n")
cmd_export <- sprintf("Rscript %s %s %s", export_script, ref_rds, temp_dir)
ret_export <- system(cmd_export)
if (ret_export != 0) {
  unlink(temp_dir, recursive = TRUE)
  stop("Export step failed.")
}

# Step 2: Build H5AD in Python
cat("Step 2: Building H5AD in Python...\n")
cmd_build <- sprintf("python3 %s %s %s", build_script, temp_dir, out_h5ad)
ret_build <- system(cmd_build)
if (ret_build != 0) {
  unlink(temp_dir, recursive = TRUE)
  stop("Build step failed.")
}

# Cleanup
unlink(temp_dir, recursive = TRUE)
cat(sprintf("Conversion successful. Output: %s\n", out_h5ad))
