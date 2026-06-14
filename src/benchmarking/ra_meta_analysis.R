#!/usr/bin/env Rscript
#
# RA consensus signature via random-effects meta-analysis — Step B1b.
#
# Uses metafor::rma (DerSimonian-Laird random-effects, REML estimator)
# to combine per-cohort DE results.  Genes must appear in >= 2 of 3 cohorts.
#
# Outputs:
#   results/meta/ra_consensus_signature.csv     — ranked consensus gene list
#   results/meta/ra_meta_analysis_summary.png   — forest plot + volcano
#
# Usage:
#   Rscript src/benchmarking/ra_meta_analysis.R

suppressPackageStartupMessages({
  library(metafor)
  library(ggplot2)
  library(data.table)
})

DE_DIR  <- "results/de/ra"
OUT_DIR <- "results/meta"
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

ACCS        <- c("GSE55457", "GSE36700", "GSE77298")
MIN_COHORTS <- 2    # gene must appear in >=2 of 3 cohorts

# ── Load per-cohort DE ───────────────────────────────────────────────────────
load_de <- function(acc) {
  f <- file.path(DE_DIR, paste0(acc, "_ra_de_entrez.csv"))
  if (!file.exists(f)) {
    message("WARNING: missing ", f)
    return(NULL)
  }
  dt <- fread(f)
  # Column may be named 'entrez' or be the row index
  if ("entrez" %in% names(dt)) {
    setnames(dt, "entrez", "gene_id")
  } else {
    setnames(dt, names(dt)[1], "gene_id")
  }
  dt[, gene_id := as.character(gene_id)]
  # Replace p=0 with machine epsilon
  dt[pvalue == 0, pvalue := .Machine$double.xmin]
  dt[, dataset := acc]
  dt[, .(gene_id, log2FoldChange, pvalue, padj, dataset)]
}

de_list <- lapply(ACCS, load_de)
de_list <- de_list[!sapply(de_list, is.null)]
de_all  <- rbindlist(de_list)

# SE from p-value and LFC (normal approximation)
# SE = |LFC| / |z|, where z = qnorm(p/2)
de_all[, z_approx := abs(qnorm(pvalue / 2))]
de_all[z_approx < 1e-10, z_approx := 1e-10]
de_all[, se := abs(log2FoldChange) / z_approx]
de_all[se == 0 | !is.finite(se), se := 1]   # guard against degenerate SE

# ── Random-effects meta-analysis via metafor ─────────────────────────────────
message("Running random-effects meta-analysis (DerSimonian-Laird REML)...")

genes     <- de_all[, unique(gene_id)]
n_cohorts <- de_all[, .(n = uniqueN(dataset)), by = gene_id]
genes_ok  <- n_cohorts[n >= MIN_COHORTS, gene_id]
message(sprintf("  Genes in >= %d cohorts: %d", MIN_COHORTS, length(genes_ok)))

results <- vector("list", length(genes_ok))

for (i in seq_along(genes_ok)) {
  g   <- genes_ok[i]
  sub <- de_all[gene_id == g]

  # rma: random-effects model, yi = observed LFC, sei = SE
  fit <- tryCatch(
    rma(yi = sub$log2FoldChange, sei = sub$se, method = "REML", verbose = FALSE),
    error = function(e) NULL
  )
  if (is.null(fit)) next

  results[[i]] <- data.table(
    gene_id          = g,
    meta_log2FC      = as.numeric(fit$beta),
    meta_SE          = fit$se,
    meta_z           = fit$zval,
    meta_pvalue      = fit$pval,
    tau2             = fit$tau2,    # between-study heterogeneity
    I2               = fit$I2,      # I² statistic
    n_datasets       = nrow(sub),
    n_up             = sum(sub$log2FoldChange > 0),
    n_down           = sum(sub$log2FoldChange < 0)
  )
}

meta <- rbindlist(results, fill = TRUE)
meta <- meta[!is.na(meta_pvalue)]

# BH FDR correction
meta[, meta_padj := p.adjust(meta_pvalue, method = "BH")]

# Direction concordance
meta[, frac_concordant := pmax(n_up, n_down) / n_datasets]
meta[, replicated := frac_concordant > 0.5]

setorder(meta, meta_padj)

# Save full results
fwrite(meta, file.path(OUT_DIR, "ra_replication_stats.csv"))

# Consensus: FDR < 0.05 AND direction replicated
consensus <- meta[meta_padj < 0.05 & replicated == TRUE]
setorder(consensus, meta_padj)
fwrite(consensus, file.path(OUT_DIR, "ra_consensus_signature.csv"))

message(sprintf("\n  Total genes tested:         %d", nrow(meta)))
message(sprintf("  FDR < 0.05:                 %d", sum(meta$meta_padj < 0.05)))
message(sprintf("  FDR < 0.05 + replicated:    %d (consensus)", nrow(consensus)))
message(sprintf("    Up in RA:   %d", sum(consensus$meta_log2FC > 0)))
message(sprintf("    Down in RA: %d", sum(consensus$meta_log2FC < 0)))

# ── Forest plot for top 10 RA genes ─────────────────────────────────────────
top10 <- head(consensus, 10)$gene_id

png(file.path(OUT_DIR, "ra_meta_analysis_summary.png"),
    width = 1400, height = 900, res = 150)
par(mfrow = c(1, 2))

# Volcano
all_lfc  <- meta$meta_log2FC
all_logp <- -log10(meta$meta_pvalue)
sig_mask <- meta$meta_padj < 0.05
col_pts  <- ifelse(!sig_mask, "#aaaaaa",
             ifelse(meta$meta_log2FC > 0, "#d62728", "#1f77b4"))
plot(all_lfc, all_logp, pch = 20, cex = 0.4, col = col_pts,
     xlab = "Meta log2FC (RA vs control)",
     ylab = "-log10(meta p-value)",
     main = "RA meta-analysis volcano\n(random-effects, 3 synovial cohorts)")
abline(h = -log10(0.05), lty = 2, lwd = 0.8)
abline(v = 0, lwd = 0.5)

# Forest plot for top gene
if (length(top10) > 0) {
  g <- top10[1]
  sub <- de_all[gene_id == g]
  fit <- rma(yi = sub$log2FoldChange, sei = sub$se, method = "REML")
  forest(fit,
         slab  = sub$dataset,
         xlab  = "log2 Fold Change",
         main  = paste0("Forest plot — gene ", g,
                        " (top RA consensus gene)"))
}
dev.off()

message("\nOutputs:")
message("  results/meta/ra_consensus_signature.csv")
message("  results/meta/ra_replication_stats.csv")
message("  results/meta/ra_meta_analysis_summary.png")
message("\nNext: python src/benchmarking/B2_ra_network_propagation.py")