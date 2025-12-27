#!/usr/bin/env Rscript
# Wrapper/convenience script: prefer an official conversion script from the imported Azimuth human_kidney
# folder when available, otherwise fall back to the embedded conversion behavior.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  cat("Usage: convert_ref_rds_to_h5ad.R <ref.Rds> <fullref.Rds> <out.h5ad>\n")
  quit(status = 1)
}

ref_rds <- args[1]
fullref_rds <- args[2]
out_h5ad <- args[3]

# Path to imported official conversion script
official_script <- file.path('examples', 'azimuth_human_kidney', 'scripts', 'convert_to_h5ad.R')
if (file.exists(official_script)) {
  cat(sprintf("Using official conversion script at %s\n", official_script))
  ret <- system2('Rscript', args = c(official_script, ref_rds, fullref_rds, out_h5ad), stdout = TRUE, stderr = TRUE)
  cat(paste(ret, collapse = '\n'), '\n')
  quit(status = 0)
}

# Fallback: if official script isn't present, perform an in-place conversion using Seurat/SeuratDisk
suppressPackageStartupMessages({
  library(Seurat)
  library(SeuratDisk)
})

cat('Official conversion script not found; running fallback conversion in this script.\n')
ref <- readRDS(file = ref_rds)
fullref <- readRDS(file = fullref_rds)
fullref <- subset(x = fullref, cells = Cells(x = ref))
if (!is.null(ref[['refUMAP']])) {
  fullref[['umap']] <- ref[['refUMAP']]
  Key(object = fullref[['umap']]) <- 'umap_'
  DefaultAssay(object = fullref[['umap']]) <- 'RNA'
}
DefaultAssay(object = fullref) <- 'RNA'
fullref <- NormalizeData(object = fullref)
fullref <- DietSeurat(object = fullref, dimreducs = 'umap', assays = 'RNA')
for (i in colnames(x = fullref[[]])) fullref[[i]] <- NULL
fullref <- AddMetaData(object = fullref, metadata = ref[[]])
if (!is.null(fullref[['umap']])) try({ Misc(object = fullref[['umap']], slot = 'model') <- NULL }, silent = TRUE)
fullref <- RenameCells(object = fullref, new.names = paste0('cell', 1:ncol(x = fullref)))
for (i in colnames(x = fullref[[]])) if (is.factor(x = fullref[[i, drop = TRUE]])) fullref[[i]] <- as.character(x = fullref[[i, drop = TRUE]])

# Add per-cell counts/feature metadata
nc_name <- 'nCount_refAssay'
nf_name <- 'nFeature_refAssay'
counts_mat <- NULL
try({ counts_mat <- GetAssayData(fullref, assay = 'RNA', slot = 'counts') }, silent = TRUE)
if (!is.null(counts_mat) && length(counts_mat) > 0) {
  ncount_vals <- as.numeric(colSums(counts_mat))
  nfeature_vals <- as.integer(colSums(counts_mat > 0))
} else {
  data_mat <- GetAssayData(fullref, assay = 'RNA', slot = 'data')
  ncount_vals <- as.numeric(colSums(data_mat))
  nfeature_vals <- as.integer(colSums(data_mat > 0))
}
if (!(nc_name %in% colnames(fullref[[]]))) fullref[[nc_name]] <- ncount_vals
if (!(nf_name %in% colnames(fullref[[]]))) fullref[[nf_name]] <- nfeature_vals

tmp_h5seurat <- tempfile(fileext = '.h5seurat')
SaveH5Seurat(object = fullref, file = tmp_h5seurat, overwrite = TRUE)
Convert(tmp_h5seurat, dest = 'h5ad', overwrite = TRUE, filename = out_h5ad)
cat('Fallback conversion complete.\n')
if (length(args) < 2) {
    cat("Usage: Rscript convert_ref_rds_to_h5ad.R <input_ref.Rds> <output_file.h5ad> [--strict] [--inspect]\n")
    cat("Example: Rscript convert_ref_rds_to_h5ad.R ref.Rds azimuth_kidney_ref.h5ad --strict --inspect\n")
    quit(status = 1)
}

input_path <- args[1]
output_path <- args[2]

cat(sprintf("Loading Seurat object from %s...\n", input_path))
seurat_obj <- readRDS(input_path)

cmd_args <- commandArgs(trailingOnly = FALSE)
file_arg <- tryCatch(sub("--file=", "", cmd_args[grep("--file=", cmd_args)]), error = function(e) NA)
cat(sprintf("Invocation args (trailingOnly): %s\n", paste(commandArgs(trailingOnly = TRUE), collapse = " ")))
cat(sprintf("R reports this script as: %s\n", ifelse(!is.na(file_arg) && nzchar(file_arg), file_arg, "(not reported)")))
cat(sprintf("Ref RDS mtime: %s\n", as.character(file.info(input_path)$mtime)))
cat(sprintf("Now: %s\n", as.character(Sys.time())))

cat("Extracting data...\n")

library(jsonlite)

assay_name <- DefaultAssay(seurat_obj)
cat(sprintf("Using default assay: %s\n", assay_name))

cat("Inspecting Seurat object schema...\n")
schema_ok <- FALSE
tryCatch(
    {
        schema <- list()
        schema$class <- as.character(class(seurat_obj))[1]
        schema$n_cells <- as.integer(ncol(seurat_obj))
        schema$n_genes <- as.integer(nrow(seurat_obj))

        assays_avail <- tryCatch(Assays(seurat_obj), error = function(e) character(0))
        schema$assays <- list()
        if (length(assays_avail) > 0) {
            for (a in assays_avail) {
                aobj <- tryCatch(seurat_obj[[a]], error = function(e) NULL)
                n_features <- NA_integer_
                n_cells_a <- NA_integer_
                has_data <- FALSE
                has_counts <- FALSE
                has_scale_data <- FALSE
                counts_sum <- NA_real_
                data_sum <- NA_real_
                if (!is.null(aobj)) {
                    if (!is.null(rownames(aobj))) n_features <- length(rownames(aobj))
                    if (!is.null(colnames(aobj))) n_cells_a <- length(colnames(aobj))
                    has_data <- !is.null(aobj@data) && length(aobj@data) > 0
                    has_counts <- !is.null(aobj@counts) && length(aobj@counts) > 0
                    has_scale_data <- !is.null(aobj@scale.data) && length(aobj@scale.data) > 0
                    if (has_counts) {
                        counts_sum <- tryCatch({
                            ct <- aobj@counts
                            # sparse-aware summation (avoid coercing large sparse to dense)
                            if (inherits(ct, "dgCMatrix") || inherits(ct, "dgTMatrix") || inherits(ct, "sparseMatrix")) {
                                sum(ct@x)
                            } else {
                                sum(as.numeric(ct))
                            }
                        }, error = function(e) NA_real_)
                    }
                    if (has_data) {
                        data_sum <- tryCatch({
                            dt <- aobj@data
                            if (inherits(dt, "dgCMatrix") || inherits(dt, "dgTMatrix") || inherits(dt, "sparseMatrix")) {
                                sum(dt@x)
                            } else {
                                sum(as.numeric(dt))
                            }
                        }, error = function(e) NA_real_)
                    }
                }
                schema$assays[[length(schema$assays) + 1]] <- list(
                    name = a,
                    n_features = n_features,
                    n_cells = n_cells_a,
                    has_data = has_data,
                    has_counts = has_counts,
                    has_scale_data = has_scale_data,
                    counts_sum = counts_sum,
                    data_sum = data_sum
                )
            }
        }

        reds <- names(seurat_obj@reductions)
        schema$reductions <- lapply(reds, function(r) {
            emb <- tryCatch(Embeddings(seurat_obj, reduction = r), error = function(e) NULL)
            if (!is.null(emb)) {
                list(name = r, dims = dim(emb))
            } else {
                list(name = r, dims = NULL)
            }
        })

        schema$obs_columns <- tryCatch(colnames(seurat_obj@meta.data), error = function(e) character(0))
        var_meta <- NULL
        if (!is.null(assay_name) && assay_name %in% assays_avail) {
            vm_tmp <- tryCatch(seurat_obj[[assay_name]]@meta.features, error = function(e) NULL)
            if (!is.null(vm_tmp)) {
                cols_vm <- tryCatch(colnames(vm_tmp), error = function(e) NULL)
                if (!is.null(cols_vm)) var_meta <- cols_vm
            }
        }
        schema$var_columns <- if (!is.null(var_meta)) var_meta else character(0)

        lower_obs <- tolower(schema$obs_columns)
        schema$possible_celltype_keys <- schema$obs_columns[grepl("annot|celltype|cell_type|l1|l2|subclass|cluster", lower_obs)]
        schema$possible_donor_keys <- schema$obs_columns[grepl("donor|sample|orig.ident|patient|batch", lower_obs)]

        for (k in head(schema$possible_celltype_keys, 5)) {
            vals <- tryCatch(unique(as.character(seurat_obj@meta.data[[k]])), error = function(e) NULL)
            schema[[paste0("sample_values_", k)]] <- if (!is.null(vals)) head(vals, 20) else NULL
        }

        schema$has_raw <- FALSE
        if ("RNA" %in% assays_avail) {
            rna_assay <- tryCatch(seurat_obj@assays$RNA, error = function(e) NULL)
            if (!is.null(rna_assay) && !is.null(rna_assay@counts)) schema$has_raw <- TRUE
        }
        schema$seurat_version <- tryCatch(as.character(packageVersion("Seurat")), error = function(e) NA)
        schema$generated_at <- as.character(Sys.time())
# Check for Azimuth package availability
azimuth_available <- requireNamespace("Azimuth", quietly = TRUE)
if (!azimuth_available) {
    cat("Note: R package 'Azimuth' is not installed; Azimuth-specific validation will be limited. Install Azimuth to run full checks.\n")
}
        # Azimuth-specific checks (scRNA and scATAC reference expectations)
        az_checks <- list()

        # 1) refAssay (SCTAssay named 'refAssay' with SCTModel.list containing 'refmodel')
        az_checks$refAssay <- list(present = FALSE, has_refmodel = FALSE)
        if ('refAssay' %in% assays_avail) {
            az_checks$refAssay$present <- TRUE
            r_assay <- tryCatch(seurat_obj[['refAssay']], error = function(e) NULL)
            if (!is.null(r_assay)) {
                smn <- tryCatch({ names(r_assay@SCTModel.list) }, error = function(e) NULL)
                az_checks$refAssay$has_refmodel <- !is.null(smn) && ('refmodel' %in% smn)
            }
        }

        # 2) refUMAP and refDR reductions and requirements
        reds_names <- names(seurat_obj@reductions)
        az_checks$reductions <- list(refUMAP = list(present = 'refUMAP' %in% reds_names, has_model = FALSE),
                                     refDR = list(present = 'refDR' %in% reds_names, dims = NA_integer_, associated_assay = NA_character_))
        if (az_checks$reductions$refUMAP$present) {
            umap_misc <- tryCatch({ seurat_obj@reductions$refUMAP@misc }, error = function(e) NULL)
            az_checks$reductions$refUMAP$has_model <- !is.null(umap_misc) && !is.null(umap_misc$model)
        }
        if (az_checks$reductions$refDR$present) {
            emb <- tryCatch(Embeddings(seurat_obj, reduction = 'refDR'), error = function(e) NULL)
            if (!is.null(emb)) az_checks$reductions$refDR$dims <- ncol(as.matrix(emb))
            # try to detect assay association for refDR
            assoc <- tryCatch({ DefaultAssay(seurat_obj[['refDR']]) }, error = function(e) NULL)
            if (!is.null(assoc)) az_checks$reductions$refDR$associated_assay <- assoc
        }

        # 3) neighbors (refdr.annoy.neighbors)
        az_checks$neighbors <- list(present = FALSE)
        nb_names <- names(seurat_obj@neighbors)
        if (!is.null(nb_names) && ('refdr.annoy.neighbors' %in% nb_names)) az_checks$neighbors$present <- TRUE

        # 4) tools -> AzimuthReference
        az_checks$tools <- list(AzimuthReference = list(present = FALSE, plotref_has_metadata = FALSE, colormap_ok = FALSE))
        if ('AzimuthReference' %in% names(seurat_obj@tools)) {
            az_checks$tools$AzimuthReference$present <- TRUE
            az_obj <- tryCatch(seurat_obj@tools$AzimuthReference, error = function(e) NULL)
            if (!is.null(az_obj)) {
                pr <- tryCatch({ az_obj$plotref }, error = function(e) NULL)
                if (!is.null(pr)) {
                    pm <- tryCatch({ pr@misc$plot.metadata }, error = function(e) NULL)
                    az_checks$tools$AzimuthReference$plotref_has_metadata <- !is.null(pm) && is.data.frame(pm)

                    # colormap check
                    cm <- tryCatch({ az_obj$colormap }, error = function(e) NULL)
                    if (!is.null(cm) && is.list(cm) && is.data.frame(pm)) {
                        cols <- colnames(pm)
                        ok <- TRUE
                        for (c in cols) {
                            if (!(c %in% names(cm))) { ok <- FALSE; break }
                        }
                        az_checks$tools$AzimuthReference$colormap_ok <- ok
                    }

                    # If plot.metadata missing or colormap missing, attempt a conservative synthetic fallback
                    if (!az_checks$tools$AzimuthReference$plotref_has_metadata || !az_checks$tools$AzimuthReference$colormap_ok) {
                        cat("AzimuthReference: plot.metadata or colormap missing or incomplete. Attempting to synthesize a minimal plot.metadata and colormap for exploration (will be noted in schema).\n")
                        # pick up to two candidate metadata columns to include (prefer annotation.l2, annotation.l1, annotation.l3)
                        candidate_cols <- intersect(c('annotation.l2','annotation.l1','annotation.l3'), colnames(seurat_obj@meta.data))
                        if (length(candidate_cols) == 0) candidate_cols <- head(colnames(seurat_obj@meta.data), 2)
                        pm_df <- seurat_obj@meta.data[, candidate_cols, drop = FALSE]
                        # ensure factors
                        for (c in colnames(pm_df)) pm_df[[c]] <- as.factor(pm_df[[c]])
                        # attach to plotref misc
                        tryCatch({ pr@misc$plot.metadata <- pm_df; seurat_obj@tools$AzimuthReference$plotref <- pr }, error = function(e) cat("Could not attach synthetic plot.metadata (", conditionMessage(e), ")\n"))
                        # build simple colormap using RColorBrewer (fallback to viridis if many categories)
                        cm_out <- list()
                        for (c in colnames(pm_df)) {
                            vals <- levels(pm_df[[c]])
                            nvals <- length(vals)
                            colors <- NULL
                            if (requireNamespace('RColorBrewer', quietly = TRUE) && nvals <= 12) {
                                colors <- RColorBrewer::brewer.pal(max(3, min(12, nvals)), 'Set3')[1:nvals]
                            } else if (requireNamespace('viridisLite', quietly = TRUE)) {
                                colors <- viridisLite::viridis(nvals)
                            } else {
                                colors <- grDevices::rainbow(nvals)
                            }
                            names(colors) <- vals
                            cm_out[[c]] <- colors
                        }
                        tryCatch({ seurat_obj@tools$AzimuthReference$colormap <- cm_out }, error = function(e) cat("Could not attach synthetic colormap (", conditionMessage(e), ")\n"))
                        az_checks$tools$AzimuthReference$plotref_has_metadata <- TRUE
                        az_checks$tools$AzimuthReference$colormap_ok <- TRUE
                        az_checks$tools$AzimuthReference$plotref_synthesized <- TRUE
                    }
                }
            }
        }

        # 5) metadata checks (ori.index present, candidate celltype keys are factors)
        az_checks$metadata <- list(ori_index = FALSE, celltype_keys_factored = list())
        if ('ori.index' %in% schema$obs_columns) az_checks$metadata$ori_index <- TRUE
        for (k in schema$possible_celltype_keys) {
            is_fact <- FALSE
            if (k %in% schema$obs_columns) {
                is_fact <- is.factor(seurat_obj@meta.data[[k]])
            }
            az_checks$metadata$celltype_keys_factored[[k]] <- is_fact
        }

        # 6) scATAC-specific checks for extended references
        az_checks$scATAC <- list(has_Bridge = 'Bridge' %in% assays_avail, has_ATAC = 'ATAC' %in% assays_avail,
                                 reductions = list(lap = 'lap' %in% reds_names, slsi = 'slsi' %in% reds_names, ref_refDR = 'ref.refDR' %in% reds_names))

        # Summarize pass/fail
        az_checks$summary_ok <- TRUE
        # Requirements for scRNA azimuth reference
        if (!az_checks$refAssay$present || !az_checks$refAssay$has_refmodel) az_checks$summary_ok <- FALSE
        if (!az_checks$reductions$refUMAP$present || !az_checks$reductions$refUMAP$has_model) az_checks$summary_ok <- FALSE
        if (!az_checks$reductions$refDR$present || is.na(az_checks$reductions$refDR$dims) || az_checks$reductions$refDR$dims < 50) az_checks$summary_ok <- FALSE
        if (!az_checks$neighbors$present) az_checks$summary_ok <- FALSE
        if (!az_checks$tools$AzimuthReference$present || !az_checks$tools$AzimuthReference$plotref_has_metadata) az_checks$summary_ok <- FALSE
        if (!az_checks$metadata$ori_index) az_checks$summary_ok <- FALSE

        schema$azimuth_checks <- az_checks

        # write schema summary and print a human-readable summary
        schema_path <- paste0(tools::file_path_sans_ext(output_path), ".schema.json")
        cat(sprintf("Writing schema summary to %s\n", schema_path))
        write_json(schema, schema_path, pretty = TRUE, auto_unbox = TRUE)

        cat("\nAzimuth reference validation summary:\n")
        cat(sprintf(" refAssay present: %s, has refmodel: %s\n", az_checks$refAssay$present, az_checks$refAssay$has_refmodel))
        cat(sprintf(" refUMAP present: %s, has model: %s\n", az_checks$reductions$refUMAP$present, az_checks$reductions$refUMAP$has_model))
        cat(sprintf(" refDR present: %s, dims: %s, associated_assay: %s\n", az_checks$reductions$refDR$present, as.character(az_checks$reductions$refDR$dims), az_checks$reductions$refDR$associated_assay))
        cat(sprintf(" neighbors (refdr.annoy.neighbors) present: %s\n", az_checks$neighbors$present))
        cat(sprintf(" AzimuthReference tool present: %s, plot.metadata present: %s, colormap ok: %s\n", az_checks$tools$AzimuthReference$present, az_checks$tools$AzimuthReference$plotref_has_metadata, az_checks$tools$AzimuthReference$colormap_ok))
        cat(sprintf(" metadata ori.index present: %s\n", az_checks$metadata$ori_index))
        cat(sprintf(" scATAC (Bridge present, ATAC present, lap,slsi,ref.refDR): %s\n", paste(c(az_checks$scATAC$has_Bridge, az_checks$scATAC$has_ATAC, az_checks$scATAC$reductions$lap, az_checks$scATAC$reductions$slsi, az_checks$scATAC$reductions$ref_refDR), collapse = ", ")))

        if (!az_checks$summary_ok) {
            cat("\nWARNING: This reference does not meet all Azimuth reference requirements. Some Azimuth features/analyses may not function properly. See schema .azimuth_checks for details.\n")
            if (strict_mode) {
                cat("Strict mode enabled (--strict): exiting with non-zero status due to failed Azimuth checks.\n")
                quit(status = 2)
            }
        } else {
            cat("\nAzimuth reference validation: ALL CHECKS PASSED\n")
        }

        cat(sprintf("Class: %s; cells=%d; genes=%d\n", schema$class, schema$n_cells, schema$n_genes))
        cat(sprintf("Assays: %s\n", paste(assays_avail, collapse = ", ")))
        cat(sprintf("Reductions: %s\n", paste(reds, collapse = ", ")))
        cat(sprintf("Candidate celltype keys: %s\n", paste(schema$possible_celltype_keys, collapse = ", ")))
        cat(sprintf("Candidate donor/sample keys: %s\n", paste(schema$possible_donor_keys, collapse = ", ")))

        # Print per-assay sums for quick diagnostics
        cat("\nPer-assay counts/data sums (for verification):\n")
        if (length(schema$assays) > 0) {
            for (a in schema$assays) {
                cs <- ifelse(is.null(a$counts_sum) || is.na(a$counts_sum), "NA", as.character(a$counts_sum))
                ds <- ifelse(is.null(a$data_sum) || is.na(a$data_sum), "NA", as.character(a$data_sum))
                cat(sprintf("  %s: counts_sum=%s, data_sum=%s\n", a$name, cs, ds))
            }
        }

        cat("\nSample of obs (metadata) -- first 10 rows:\n")
        print(utils::head(seurat_obj@meta.data, 10))
        cat("\nSample of var (feature metadata) -- first 10 rows:\n")
        vm <- tryCatch(seurat_obj[[assay_name]]@meta.features, error = function(e) NULL)
        if (!is.null(vm)) {
            print(utils::head(vm, 10))
        } else {
            cat("(no feature metadata available)\n")
        }

        if (length(reds) > 0) {
            cat("\nReductions preview (first 5 rows each):\n")
            for (r in reds) {
                emb <- tryCatch(Embeddings(seurat_obj, reduction = r), error = function(e) NULL)
                if (!is.null(emb)) {
                    cat(sprintf("\nReduction: %s (dims: %s)\n", r, paste(dim(emb), collapse = "x")))
                    print(utils::head(as.data.frame(emb), 5))
                }
            }
        }

        cat("\nCandidate celltype key summaries:\n")
        for (k in schema$possible_celltype_keys) {
            cat(sprintf("\nKey: %s -> top values:\n", k))
            tvals <- sort(table(seurat_obj@meta.data[[k]]), decreasing = TRUE)
            print(utils::head(tvals, 10))
        }
        cat("\nCandidate donor/sample key summaries:\n")
        for (k in schema$possible_donor_keys) {
            cat(sprintf("\nKey: %s -> top values:\n", k))
            tvals <- sort(table(seurat_obj@meta.data[[k]]), decreasing = TRUE)
            print(utils::head(tvals, 10))
        }

        obs_path <- paste0(tools::file_path_sans_ext(output_path), ".obs_sample.csv")
        var_path <- paste0(tools::file_path_sans_ext(output_path), ".var_sample.csv")
        write.csv(utils::head(seurat_obj@meta.data, 200), obs_path, row.names = TRUE)
        if (!is.null(vm)) write.csv(utils::head(vm, 200), var_path, row.names = TRUE)
        cat(sprintf("\nWrote sample CSVs: %s, %s\n", obs_path, var_path))

        reds_exported <- c()
        reds_list <- tryCatch(names(seurat_obj@reductions), error = function(e) character(0))
        if (length(reds_list) > 0) {
            for (r in reds_list) {
                emb <- tryCatch(Embeddings(seurat_obj, reduction = r), error = function(e) NULL)
                if (!is.null(emb)) {
                    red_path <- paste0(tools::file_path_sans_ext(output_path), ".", r, ".csv")
                    to_save <- as.data.frame(emb)
                    if (ncol(to_save) > 50) to_save <- to_save[, 1:50, drop = FALSE]
                    write.csv(to_save, red_path, row.names = TRUE)
                    reds_exported <- c(reds_exported, red_path)
                }
            }
        }
        if (length(reds_exported) > 0) cat(sprintf("Wrote reductions to: %s\n", paste(reds_exported, collapse = ", ")))

        count_paths <- c()
        if (length(schema$possible_celltype_keys) > 0) {
            for (k in schema$possible_celltype_keys) {
                tab <- sort(table(seurat_obj@meta.data[[k]]), decreasing = TRUE)
                cp <- paste0(tools::file_path_sans_ext(output_path), ".", k, ".counts.csv")
                write.csv(as.data.frame(tab), cp, row.names = TRUE)
                count_paths <- c(count_paths, cp)
            }
        }
        if (length(count_paths) > 0) cat(sprintf("Wrote cell-type count CSVs: %s\n", paste(count_paths, collapse = ", ")))

        cat("\nSession info (R):\n")
        print(sessionInfo())

        schema_ok <- TRUE
    },
    error = function(e) {
        cat("Schema inspection failed with error:", conditionMessage(e), "\n")
        schema_ok <- FALSE
    }
)

cat(sprintf("Schema inspection success: %s\n", ifelse(schema_ok, "YES", "NO")))

# Prefer refAssay if it exists and contains data/counts, otherwise select main matrix (prefer assay "data" then counts)
prefer_assay <- assay_name
if ('refAssay' %in% assays_avail) {
    r_assay_tmp <- tryCatch(seurat_obj[['refAssay']], error = function(e) NULL)
    if (!is.null(r_assay_tmp)) {
        has_data_tmp <- !is.null(r_assay_tmp@data) && length(r_assay_tmp@data) > 0
        has_counts_tmp <- !is.null(r_assay_tmp@counts) && length(r_assay_tmp@counts) > 0
        if (isTRUE(has_data_tmp) || isTRUE(has_counts_tmp)) {
            prefer_assay <- 'refAssay'
            cat("Preferring assay 'refAssay' for main expression matrix (has_data=", has_data_tmp, ", has_counts=", has_counts_tmp, ")\n")
        }
    }
}

X <- NULL
# try data slot on preferred assay
try(
    {
        X <- GetAssayData(seurat_obj, assay = prefer_assay, layer = "data")
    },
    silent = TRUE
)
# fallback to counts slot on preferred assay
if (is.null(X)) {
    try(
        {
            X <- GetAssayData(seurat_obj, assay = prefer_assay, layer = "counts")
        },
        silent = TRUE
    )
}

# If still NULL, fall back to default/default assay slots (previous behavior)
if (is.null(X)) {
    assay_obj <- seurat_obj[[assay_name]]
    if (!is.null(assay_obj@data) && length(assay_obj@data) > 0) {
        X <- assay_obj@data
    } else if (!is.null(assay_obj@counts) && length(assay_obj@counts) > 0) {
        X <- assay_obj@counts
    } else {
        stop("Could not extract expression data from the Seurat object. Tried preferred assay, layer='data','counts' and assay slots.")
    }
}

# Scan all assays for a non-zero counts slot (prefer counts slot; fall back to data slot if counts absent)
counts_found <- FALSE
counts_assay <- NULL
counts_mat <- NULL
if (length(assays_avail) > 0) {
    for (a in assays_avail) {
        # try counts slot first
        ct <- tryCatch({
            aobj <- seurat_obj[[a]]
            if (!is.null(aobj@counts) && length(aobj@counts) > 0) aobj@counts else NULL
        }, error = function(e) NULL)
        if (!is.null(ct)) {
            # check non-zero
            s <- tryCatch({ sum(ct) }, error = function(e) NA)
            if (!is.na(s) && s > 0) {
                counts_found <- TRUE
                counts_assay <- a
                counts_mat <- ct
                break
            }
        }
    }
    # if no non-zero counts found, look for non-zero data slots as a fallback
    if (!counts_found) {
        for (a in assays_avail) {
            dt <- tryCatch({
                aobj <- seurat_obj[[a]]
                if (!is.null(aobj@data) && length(aobj@data) > 0) aobj@data else NULL
            }, error = function(e) NULL)
            if (!is.null(dt)) {
                s <- tryCatch({ sum(dt) }, error = function(e) NA)
                if (!is.na(s) && s > 0) {
                    counts_found <- TRUE
                    counts_assay <- a
                    counts_mat <- dt
                    break
                }
            }
        }
    }
}

# Annotate schema with counts info and rewrite schema summary (overwrite previous if needed)
schema$counts_present <- counts_found
schema$counts_assay <- if (counts_found) counts_assay else NULL
cat(sprintf("Counts present? %s; counts assay: %s\n", ifelse(counts_found, "YES", "NO"), ifelse(is.null(counts_assay), "(none)", counts_assay)))
if (!is.null(schema_path)) {
    write_json(schema, schema_path, pretty = TRUE, auto_unbox = TRUE)
    cat(sprintf("(Updated schema summary written to %s)\n", schema_path))
}

# Deep inspection (help locate expression/counts stored in nonstandard slots)
if (inspect_mode || !counts_found) {
    cat("\nRunning deep inspection of Seurat object to locate expression/counts...\n")
    inspection <- list(timestamp = as.character(Sys.time()), assays = list(), tools = list(), searches = list())

    # helper: summarize a matrix-like object in a sparse-safe way
    summarize_mat <- function(m) {
        out <- list()
        tryCatch({
            if (inherits(m, 'dgCMatrix') || inherits(m, 'dgTMatrix') || inherits(m, 'sparseMatrix')) {
                out$dims <- dim(m)
                out$nnz <- length(m@x)
                out$sum <- sum(m@x)
                out$sample_row_sums <- as.numeric(Matrix::rowSums(m)[1:min(10, nrow(m))])
                out$sample_col_sums <- as.numeric(Matrix::colSums(m)[1:min(10, ncol(m))])
            } else if (is.matrix(m) || inherits(m, 'Matrix')) {
                out$dims <- dim(m)
                out$nnz <- sum(m != 0)
                out$sum <- sum(as.numeric(m))
                out$sample_row_sums <- as.numeric(rowSums(m)[1:min(10, nrow(m))])
                out$sample_col_sums <- as.numeric(colSums(m)[1:min(10, ncol(m))])
            } else {
                out <- NULL
            }
        }, error = function(e) {
            out$error <- conditionMessage(e)
        })
        return(out)
    }

    # Inspect assays slots in detail
    if (length(assays_avail) > 0) {
        for (a in assays_avail) {
            aobj <- tryCatch(seurat_obj[[a]], error = function(e) NULL)
            ai <- list(name = a)
            if (!is.null(aobj)) {
                # check counts/data/scale.data and any unusual slots
                ai$has_counts <- !is.null(aobj@counts) && length(aobj@counts) > 0
                ai$has_data <- !is.null(aobj@data) && length(aobj@data) > 0
                ai$has_scaled <- !is.null(aobj@scale.data) && length(aobj@scale.data) > 0
                if (ai$has_counts) ai$counts_summary <- summarize_mat(aobj@counts)
                if (ai$has_data) ai$data_summary <- summarize_mat(aobj@data)
                if (ai$has_scaled) ai$scaled_summary <- summarize_mat(aobj@scale.data)
                # try other candidate slots
                try({ ai$SCTModel.names <- names(aobj@SCTModel.list) }, silent = TRUE)
            }
            inspection$assays[[length(inspection$assays) + 1]] <- ai
        }
    }

    # Inspect common places for matrices: tools, reductions misc, and top-level list elements
    # 1) tools
    if (length(names(seurat_obj@tools)) > 0) {
        for (tname in names(seurat_obj@tools)) {
            tobj <- tryCatch(seurat_obj@tools[[tname]], error = function(e) NULL)
            if (!is.null(tobj)) {
                # if it's a list-like object, search for matrix-like elements
                found <- list()
                if (is.list(tobj) || is.environment(tobj)) {
                    for (el in names(tobj)) {
                        val <- tryCatch(tobj[[el]], error = function(e) NULL)
                        if (!is.null(val) && (inherits(val, 'dgCMatrix') || is.matrix(val) || inherits(val, 'Matrix'))) {
                            found[[el]] <- summarize_mat(val)
                        }
                    }
                }
                inspection$tools[[tname]] <- list(class = class(tobj), found = found)
            }
        }
    }

    # 2) reductions misc (check for matrices stored in misc elements)
    if (length(names(seurat_obj@reductions)) > 0) {
        for (r in names(seurat_obj@reductions)) {
            remisc <- tryCatch(seurat_obj@reductions[[r]]@misc, error = function(e) NULL)
            if (!is.null(remisc) && is.list(remisc)) {
                found <- list()
                for (el in names(remisc)) {
                    val <- remisc[[el]]
                    if (!is.null(val) && (inherits(val, 'dgCMatrix') || is.matrix(val) || inherits(val, 'Matrix'))) {
                        found[[el]] <- summarize_mat(val)
                    }
                }
                if (length(found) > 0) inspection$searches[[r]] <- found
            }
        }
    }

    # 3) scan top-level slots (meta, assays already done)
    # 4) attempt a shallow recursive scan for any matrix-like objects in top-level fields (limited depth to avoid runaway)
    recursive_find <- function(obj, path = '', depth = 0, maxdepth = 3) {
        res <- list()
        if (depth > maxdepth) return(res)
        if (is.list(obj) || is.environment(obj) || isS4(obj)) {
            nms <- tryCatch(names(obj), error = function(e) NULL)
            if (is.null(nms) && isS4(obj)) nms <- slotNames(obj)
            if (!is.null(nms)) {
                for (nm in nms) {
                    val <- tryCatch({ if (isS4(obj)) slot(obj, nm) else obj[[nm]] }, error = function(e) NULL)
                    if (!is.null(val)) {
                        if (inherits(val, 'dgCMatrix') || is.matrix(val) || inherits(val, 'Matrix')) {
                            res[[paste0(path, '/', nm)]] <- summarize_mat(val)
                        } else if (is.list(val) || is.environment(val) || isS4(val)) {
                            sub <- recursive_find(val, paste0(path, '/', nm), depth + 1, maxdepth)
                            if (length(sub) > 0) res <- c(res, sub)
                        }
                    }
                }
            }
        }
        return(res)
    }

    cat('Running a limited recursive scan (depth=3) for matrix-like objects...\n')
    found_mats <- recursive_find(seurat_obj, path = 'seurat_obj', depth = 0, maxdepth = 3)
    inspection$recursive_found <- found_mats

    # write inspection JSON to disk for user examination
    insp_path <- paste0(tools::file_path_sans_ext(output_path), '.inspection.json')
    tryCatch({ write_json(inspection, insp_path, pretty = TRUE, auto_unbox = TRUE); cat(sprintf('Wrote deep inspection to %s\n', insp_path)) }, error = function(e) cat('Could not write inspection JSON:', conditionMessage(e), '\n'))

    # print concise human summary
    cat('\nDeep inspection summary:\n')
    for (ai in inspection$assays) {
        cat(sprintf('  Assay %s: has_counts=%s, has_data=%s, has_scaled=%s\n', ai$name, isTRUE(ai$has_counts), isTRUE(ai$has_data), isTRUE(ai$has_scaled)))
        if (!is.null(ai$counts_summary)) cat(sprintf('    counts nnz=%s, sum=%s\n', ai$counts_summary$nnz, ai$counts_summary$sum))
        if (!is.null(ai$data_summary)) cat(sprintf('    data nnz=%s, sum=%s\n', ai$data_summary$nnz, ai$data_summary$sum))
    }
    if (length(inspection$recursive_found) > 0) cat(sprintf('\nFound %d matrix-like objects in recursive scan. See %s for details.\n', length(inspection$recursive_found), insp_path)) else cat('\nNo additional matrix-like objects found in recursive scan.\n')
}


cell_names <- colnames(seurat_obj)
if (!is.null(cell_names) && length(cell_names) == ncol(X)) {
    colnames(X) <- cell_names
}

X <- Matrix::t(X)

obs <- seurat_obj@meta.data
if (!is.null(rownames(obs)) && !is.null(rownames(X)) && !all(rownames(obs) == rownames(X))) {
    common <- intersect(rownames(obs), rownames(X))
    if (length(common) == nrow(obs)) {
        obs <- obs[rownames(X), , drop = FALSE]
    } else {
        warning("Cell names in metadata and assay matrix do not perfectly match. Proceeding with existing metadata order.")
    }
} else if (is.null(rownames(obs)) && !is.null(rownames(X))) {
    rownames(obs) <- rownames(X)
}

var <- tryCatch(seurat_obj[[assay_name]]@meta.features, error = function(e) NULL)
if (is.null(var) || (is.data.frame(var) && nrow(var) == 0)) {
    var <- data.frame(row.names = colnames(X))
}

# Add nCount_<assay> and nFeature_<assay> obs columns when missing (use the preferred assay name)
try({
    nc_name <- paste0('nCount_', prefer_assay)
    nf_name <- paste0('nFeature_', prefer_assay)
    # compute per-cell totals and feature counts (X is cells x genes at this point)
    if (!is.null(X)) {
        # Use Matrix rowSums for sparse-aware sums
        ncounts_vals <- as.numeric(Matrix::rowSums(X))
        nfeatures_vals <- as.integer(rowSums(X != 0))
        if (!(nc_name %in% colnames(obs))) {
            obs[[nc_name]] <- ncounts_vals
            cat(sprintf("Added obs column '%s' (per-cell totals from %s)\n", nc_name, prefer_assay))
        }
        if (!(nf_name %in% colnames(obs))) {
            obs[[nf_name]] <- nfeatures_vals
            cat(sprintf("Added obs column '%s' (per-cell non-zero feature counts from %s)\n", nf_name, prefer_assay))
        }
        # Update schema obs columns and write schema summary again if schema_path provided
        if (!is.null(schema_path)) {
            schema$obs_columns <- colnames(obs)
            try({ write_json(schema, schema_path, pretty = TRUE, auto_unbox = TRUE); cat(sprintf('(Updated schema summary written to %s)\n', schema_path)) }, silent = TRUE)
        }
    }
}, silent = TRUE)

obsm <- list()
for (red_name in names(seurat_obj@reductions)) {
    emb_name <- paste0("X_", red_name)
    obsm[[emb_name]] <- Embeddings(seurat_obj, reduction = red_name)
}

cat("Constructing AnnData object...\n")

# Prepare layers and raw (counts) if we detected a non-zero counts/data assay
layers_list <- list()
raw_ad <- NULL
if (exists("counts_found") && counts_found && !is.null(counts_mat)) {
    # transpose counts to shape (cells x genes)
    counts_t <- Matrix::t(counts_mat)
    # try to align cell ordering if X already has row names
    if (!is.null(rownames(counts_t)) && !is.null(rownames(X)) && !all(rownames(counts_t) == rownames(X))) {
        common <- intersect(rownames(counts_t), rownames(X))
        if (length(common) == nrow(X)) {
            counts_t <- counts_t[rownames(X), , drop = FALSE]
        } else {
            warning("Cell names in counts matrix and X do not perfectly match; proceeding without reordering counts layer.")
        }
    }
    layers_list <- list(counts = counts_t)
    # set raw to counts if counts differ from X
    same_dims <- identical(dim(counts_t), dim(X))
    equal_vals <- FALSE
    if (same_dims) {
        # try a quick check (may be slow for large matrices) -- only when dims match
        try({ equal_vals <- all(counts_t == X) }, silent = TRUE)
    }
    if (!same_dims || !isTRUE(equal_vals)) {
        raw_ad <- AnnData(X = counts_t)
    }
    cat(sprintf("Exporting counts layer from assay '%s' (non-zero)\n", counts_assay))
} else {
    cat("No non-zero counts found across assays; not exporting counts layer.\n")
}

ad <- AnnData(
    X = X,
    obs = obs,
    var = var,
    obsm = obsm,
    layers = layers_list,
    raw = raw_ad
)

cat(sprintf("Writing to %s...\n", output_path))
write_h5ad(ad, output_path)

cat("Conversion complete.\n")
