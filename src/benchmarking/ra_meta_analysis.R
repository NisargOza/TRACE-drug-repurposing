#!/usr/bin/env Rscript
#
# RA consensus signature via random-effects meta-analysis.
#
# Reads the three per-cohort DE files from results/de/ra/ and runs
# DerSimonian-Laird random-effects meta-analysis (metafor::rma, REML estimator)
# across genes present in >= 2 of 3 cohorts.
#
# Justification for random-effects: RA is heterogeneous across synovial
# biopsy sites and patient populations; between-study variance tau2 is
# expected to be non-zero (Borenstein et al. 2009, Ch. 13).
#
# Outputs:
#   results/meta/ra_consensus_signature.csv
#   results/meta/ra_replication_stats.csv
#   results/meta/ra_meta_analysis_summary.png
#
# Usage:
#   Rscript src/benchmarking/ra_meta_analysis.R

suppressPackageStartupMessages({
  library(metafor)
  library(ggplot2)
  library(data.table)
})

DE_DIR      <- "results/de/ra"
OUT_DIR     <- "results/meta"
MIN_COHORTS <- 2    # gene must appear in >= 2 of 3 cohorts

dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

ACCS <- c("GSE55457", "GSE36700", "GSE77298")

# ── Load per-cohort DE files ─────────────────────────────────────────────────
load_de <- function(acc) {
  f <- file.path(DE_DIR, paste0(acc, "_ra_de_entrez.csv"))
  if (!file.exists(f)) {
    message("WARNING: missing ", f)
    return(NULL)
  }
  dt <- fread(f)
  # First column is the Entrez index written by pandas (unnamed or "entrez")
  first_col <- names(dt)[1]
  setnames(dt, first_col, "gene_id")
  dt[, gene_id := as.character(gene_id)]
  # Guard against p=0 (causes infinite z)
  dt[pvalue == 0, pvalue := .Machine$double.xmin]
  dt[, dataset := acc]
  dt[, .(gene_id, log2FoldChange, pvalue, padj, dataset)]
}

de_list <- lapply(ACCS, load_de)
de_list <- de_list[!sapply(de_list, is.null)]

if (length(de_list) < 2) {
  stop("Need at least 2 cohort DE files in results/de/ra/ — run B1_ra_geo_de.py first")
}

de_all <- rbindlist(de_list)
message(sprintf("Loaded %d cohorts, %d gene-cohort rows", length(de_list), nrow(de_all)))

# Compute standard error from p-value and LFC (normal approximation)
# SE = |LFC| / |z|  where  z = qnorm(p/2)
de_all[, z_approx := abs(qnorm(pvalue / 2))]
de_all[z_approx < 1e-10, z_approx := 1e-10]
de_all[, se := abs(log2FoldChange) / z_approx]
de_all[se == 0 | !is.finite(se), se := 1]

# ── Random-effects meta-analysis ─────────────────────────────────────────────
message("Running metafor::rma (DerSimonian-Laird REML) ...")

n_cohorts_per_gene <- de_all[, .(n = uniqueN(dataset)), by = gene_id]
genes_ok           <- n_cohorts_per_gene[n >= MIN_COHORTS, gene_id]
message(sprintf("  Genes in >= %d cohorts: %d", MIN_COHORTS, length(genes_ok)))

results <- vector("list", length(genes_ok))

for (i in seq_along(genes_ok)) {
  g   <- genes_ok[i]
  sub <- de_all[gene_id == g]

  fit <- tryCatch(
    rma(yi = sub$log2FoldChange, sei = sub$se, method = "REML", verbose = FALSE),
    error   = function(e) NULL,
    warning = function(w) suppressWarnings(
      tryCatch(rma(yi = sub$log2FoldChange, sei = sub$se, method = "DL"),
               error = function(e2) NULL))
  )
  if (is.null(fit)) next

  results[[i]] <- data.table(
    gene_id     = g,
    meta_log2FC = as.numeric(fit$beta),
    meta_SE     = as.numeric(fit$se),
    meta_z      = as.numeric(fit$zval),
    meta_pvalue = as.numeric(fit$pval),
    tau2        = as.numeric(fit$tau2),
    I2          = as.numeric(fit$I2),
    n_datasets  = nrow(sub),
    n_up        = sum(sub$log2FoldChange > 0),
    n_down      = sum(sub$log2FoldChange < 0)
  )

  if (i %% 2000 == 0) message(sprintf("  %d / %d genes done", i, length(genes_ok)))
}

meta <- rbindlist(results, fill = TRUE)
meta <- meta[!is.na(meta_pvalue)]
meta[, meta_padj       := p.adjust(meta_pvalue, method = "BH")]
meta[, frac_concordant := pmax(n_up, n_down) / n_datasets]
meta[, replicated      := frac_concordant > 0.5]
setorder(meta, meta_padj)

fwrite(meta, file.path(OUT_DIR, "ra_replication_stats.csv"))

# Use nominal p < 0.05 + replication instead of FDR < 0.05.
# With only 3 small cohorts (n = 5-16 per group), BH correction is too
# conservative and leaves ~158 genes — far too few for CMap scoring, which
# requires ~150 up + 150 down genes (Lamb 2006). Nominal p < 0.05 with the
# cross-cohort replication filter is the primary quality gate and matches
# the approach used by Sirota et al. (2011 STM) for small-cohort CMap queries.
consensus <- meta[meta_pvalue < 0.05 & replicated == TRUE]
setorder(consensus, meta_pvalue)
fwrite(consensus, file.path(OUT_DIR, "ra_consensus_signature.csv"))

message(sprintf("\nTotal genes tested:                %d", nrow(meta)))
message(sprintf("nominal p < 0.05:                  %d", sum(meta$meta_pvalue < 0.05)))
message(sprintf("nominal p < 0.05 + replicated:     %d  (consensus)", nrow(consensus)))
message(sprintf("  Up in RA:   %d", sum(consensus$meta_log2FC > 0)))
message(sprintf("  Down in RA: %d", sum(consensus$meta_log2FC < 0)))

# ── Summary plot ─────────────────────────────────────────────────────────────
png(file.path(OUT_DIR, "ra_meta_analysis_summary.png"),
    width = 1400, height = 700, res = 150)
par(mfrow = c(1, 2))

# Volcano
col_pts <- ifelse(meta$meta_padj >= 0.05, "#aaaaaa",
           ifelse(meta$meta_log2FC > 0,   "#d62728", "#1f77b4"))
plot(meta$meta_log2FC, -log10(meta$meta_pvalue),
     pch = 20, cex = 0.35, col = col_pts,
     xlab = "Meta log2FC (RA vs control)",
     ylab = "-log10(p-value)",
     main = sprintf("RA meta-analysis volcano\n(%d cohorts, DL random-effects)",
                    length(de_list)))
abline(h = -log10(0.05 / nrow(meta)), lty = 2, lwd = 0.8, col = "grey40")
abline(v = 0, lwd = 0.5)

# Forest plot for top gene
if (nrow(consensus) > 0) {
  g   <- consensus$gene_id[1]
  sub <- de_all[gene_id == g]
  fit <- rma(yi = sub$log2FoldChange, sei = sub$se, method = "REML")
  forest(fit,
         slab = sub$dataset,
         xlab = "log2 Fold Change",
         main = paste0("Forest plot — Entrez ", g, " (top consensus gene)"))
}
dev.off()

message("\nOutputs:")
message("  results/meta/ra_consensus_signature.csv")
message("  results/meta/ra_replication_stats.csv")
message("  results/meta/ra_meta_analysis_summary.png")
message("\nNext: python src/benchmarking/B2_ra_network_propagation.py")