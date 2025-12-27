#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly = TRUE)

library(Seurat)
library(Matrix)

mat <- readMM(file = "data/kidney_demo_expression.mtx")
cells <- read.csv(file = "data/kidney_demo_cells.csv", header = FALSE)
features <- read.csv(file = "data/kidney_demo_features.csv", header = FALSE)
meta <- read.csv(file = "data/kidney_demo_metadata.csv", row.names = 1)

# Safely subset metadata to available columns (avoid failing if expected columns are absent)
wanted_cols <- c("Project", "Experiment", "celltype", "compartment", "broad_celltype")
avail <- intersect(wanted_cols, colnames(meta))
if (length(avail) > 0) {
  meta <- meta[, avail, drop = FALSE]
} else {
  warning("None of the expected metadata columns found; keeping full metadata table as-is.")
}

# If 'celltype' is missing, try to create it from common alternatives
if (!("celltype" %in% colnames(meta))) {
  candidates <- c("cell_type", "annotation.l2", "annotation.l3", "annotation.l1", "broad_celltype")
  cand <- intersect(candidates, colnames(meta))
  if (length(cand) > 0) meta$celltype <- as.character(meta[[cand[1]]])
}

# Assign row/col names robustly depending on matrix orientation
if (nrow(mat) == nrow(features) && ncol(mat) == nrow(cells)) {
  rownames(mat) <- features[, 1]
  colnames(mat) <- cells[, 1]
} else if (nrow(mat) == nrow(cells) && ncol(mat) == nrow(features)) {
  rownames(mat) <- cells[, 1]
  colnames(mat) <- features[, 1]
} else {
  if (ncol(mat) == nrow(features)) colnames(mat) <- features[, 1]
  if (nrow(mat) == nrow(cells)) rownames(mat) <- cells[, 1]
}

# Optional stewart_cells filter (if present)
if (file.exists("data/stewart_cells.csv")) {
  cells.keep <- read.csv("data/stewart_cells.csv", header = TRUE, stringsAsFactors = FALSE)
  keep_ids <- as.character(cells.keep[, 1])
  if (all(keep_ids %in% rownames(mat))) {
    mat <- mat[keep_ids, , drop = FALSE]
  } else if (all(keep_ids %in% colnames(mat))) {
    mat <- mat[, keep_ids, drop = FALSE]
  } else {
    warning("stewart_cells.csv found but IDs not present in matrix; skipping filter")
  }
} else {
  message("No stewart_cells.csv found; using all cells present in the matrix")
}

# Ensure Seurat's expected orientation (genes x cells): if rows currently look like cell IDs, transpose
if (!is.null(rownames(mat)) && length(intersect(rownames(mat), cells[, 1])) == length(rownames(mat))) {
  mat <- t(x = mat)
}

ob <- CreateSeuratObject(counts = mat, meta.data = meta)
saveRDS(object = ob, file = args[1])