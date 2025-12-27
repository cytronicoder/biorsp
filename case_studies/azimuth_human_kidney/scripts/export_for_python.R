#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  cat("Usage: export_for_python.R <input_ref.Rds> <out_dir>\n")
  quit(status = 1)
}
input <- args[1]
out_dir <- args[2]
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

library(Seurat)
library(Matrix)

ref <- readRDS(input)
assay_name <- DefaultAssay(ref)
# prefer data slot, then counts
# Try to retrieve assay layers (work with multiple Seurat versions)
assay_obj <- ref@assays[[assay_name]]
mat_data <- NULL
mat_counts <- NULL
# common layer names to look for
if (!is.null(assay_obj@layers) && length(assay_obj@layers) > 0) {
  layer_names <- names(assay_obj@layers)
  if ('data' %in% layer_names) mat_data <- assay_obj@layers[['data']]
  if ('counts' %in% layer_names) mat_counts <- assay_obj@layers[['counts']]
  # fallback to first numeric layer as data
  if (is.null(mat_data) && length(layer_names) > 0) {
    for (ln in layer_names) {
      val <- assay_obj@layers[[ln]]
      if (!is.null(val) && (inherits(val, 'dgCMatrix') || inherits(val, 'dgTMatrix') || is.matrix(val))) {
        mat_data <- val
        break
      }
    }
  }
}
# also attempt GetAssayData as an additional fallback
if (is.null(mat_data)) mat_data <- tryCatch({ GetAssayData(ref, assay = assay_name, slot = 'data') }, error = function(e) NULL)
if (is.null(mat_counts)) mat_counts <- tryCatch({ GetAssayData(ref, assay = assay_name, slot = 'counts') }, error = function(e) NULL)

# If data slot missing, fall back to counts for writing
if (is.null(mat_data) && !is.null(mat_counts)) mat_data <- mat_counts

# ensure matrices are dgCMatrix for writeMM
if (!is.null(mat_data) && inherits(mat_data, 'dgTMatrix')) mat_data <- as(mat_data, 'dgCMatrix')
if (!is.null(mat_counts) && inherits(mat_counts, 'dgTMatrix')) mat_counts <- as(mat_counts, 'dgCMatrix')

if (!is.null(mat_data)) {
  Matrix::writeMM(mat_data, file.path(out_dir, 'data.mtx'))
}
if (!is.null(mat_counts)) {
  Matrix::writeMM(mat_counts, file.path(out_dir, 'counts.mtx'))
}

# write obs and var
write.csv(as.data.frame(ref@meta.data), file = file.path(out_dir, 'obs.csv'), row.names = TRUE)
var_genes <- NULL
# try assay feature table first, then other fallbacks
try({
  if (!is.null(rownames(assay_obj@features))) var_genes <- rownames(assay_obj@features)
}, silent = TRUE)
try({
  if (!is.null(rownames(ref@assays[[assay_name]]@meta.features))) var_genes <- rownames(ref@assays[[assay_name]]@meta.features)
}, silent = TRUE)
if (is.null(var_genes) && !is.null(mat_data) && !is.null(rownames(mat_data))) var_genes <- rownames(mat_data)
if (is.null(var_genes) && !is.null(mat_counts) && !is.null(rownames(mat_counts))) var_genes <- rownames(mat_counts)
if (is.null(var_genes) && !is.null(mat_data) && !is.null(colnames(mat_data))) var_genes <- colnames(mat_data)
if (is.null(var_genes) && !is.null(mat_counts) && !is.null(colnames(mat_counts))) var_genes <- colnames(mat_counts)
if (is.null(var_genes)) {
  # use gene1..geneN where N = number of rows in data.mtx (genes)
  if (!is.null(mat_data)) {
    ng <- tryCatch(nrow(mat_data), error = function(e) NA)
    if (!is.na(ng)) var_genes <- paste0('gene', seq_len(ng))
  }
  if (is.null(var_genes)) var_genes <- character(0)
}
write.csv(data.frame(gene=var_genes), file = file.path(out_dir, 'var.csv'), row.names = FALSE)

# write embeddings (UMAP or refUMAP if present)
reds <- names(ref@reductions)
for (r in reds) {
  emb <- tryCatch(Embeddings(ref, reduction = r), error = function(e) NULL)
  if (!is.null(emb)) {
    write.csv(as.data.frame(emb), file = file.path(out_dir, paste0(r, '.csv')), row.names = TRUE)
  }
}

cat('Exported data to', out_dir, '\n')
