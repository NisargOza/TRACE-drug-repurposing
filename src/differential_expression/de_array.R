
# Limma DE for a single microarray dataset.
# Handles probe->Entrez mapping internally per platform.
# Args: acc  gpl  expr_csv  meta_csv  out_csv

args <- commandArgs(trailingOnly = TRUE)
acc      <- args[1]
gpl      <- args[2]
expr_csv <- args[3]
meta_csv <- args[4]
out_csv  <- args[5]

suppressPackageStartupMessages(library(limma))

get_probe_map <- function(gpl) {
  if (gpl == "GPL5175") {
    if (!requireNamespace("huex10sttranscriptcluster.db", quietly = TRUE)) {
      if (!requireNamespace("BiocManager", quietly = TRUE))
        install.packages("BiocManager", repos = "https://cloud.r-project.org")
      BiocManager::install("huex10sttranscriptcluster.db", ask = FALSE, update = FALSE)
    }
    suppressPackageStartupMessages(library(huex10sttranscriptcluster.db))
    eg <- AnnotationDbi::toTable(huex10sttranscriptclusterENTREZID)
    probe_map <- setNames(eg$gene_id, eg$probe_id)
    cat(sprintf("  GPL5175: %d probe->Entrez mappings\n", length(probe_map)))
    return(probe_map)
  } else if (gpl == "GPL6480") {
    # Pre-built by 05_differential_expression.py (streaming Python parse of soft file)
    pmap_csv <- file.path("results/de", "GPL6480_probe_map.csv")
    if (!file.exists(pmap_csv)) stop(paste("GPL6480 probe map not found:", pmap_csv))
    tbl <- read.csv(pmap_csv, stringsAsFactors = FALSE)
    probe_map <- setNames(as.character(tbl$gene_id), as.character(tbl$probe_id))
    cat(sprintf("  GPL6480: %d probe->Entrez mappings\n", length(probe_map)))
    return(probe_map)
  } else { stop(paste("Unsupported GPL:", gpl)) }
}

cat(sprintf("  Loading expression matrix for %s...\n", acc))
expr <- read.csv(expr_csv, row.names = 1, check.names = FALSE)
expr <- as.matrix(expr)

meta <- read.csv(meta_csv, row.names = 1, stringsAsFactors = FALSE)
meta <- meta[meta$condition %in% c("IPF", "control"), , drop = FALSE]
common <- intersect(colnames(expr), rownames(meta))
expr <- expr[, common, drop = FALSE]
meta <- meta[common, , drop = FALSE]
cat(sprintf("  Samples: IPF=%d  control=%d\n",
            sum(meta$condition=="IPF"), sum(meta$condition=="control")))

meta$condition <- factor(meta$condition, levels = c("control", "IPF"))
design <- model.matrix(~ condition, data = meta)
fit    <- lmFit(expr, design)
fit2   <- eBayes(fit)
tt     <- topTable(fit2, coef = "conditionIPF", number = Inf,
                   sort.by = "none", adjust.method = "BH")
tt$probe_id <- rownames(tt)

probe_map  <- get_probe_map(gpl)
tt$gene_id <- probe_map[tt$probe_id]
tt         <- tt[!is.na(tt$gene_id) & tt$gene_id != "", ]
tt         <- tt[order(tt$adj.P.Val), ]
tt         <- tt[!duplicated(tt$gene_id), ]

out <- data.frame(
  gene_id        = tt$gene_id,
  log2FoldChange = tt$logFC,
  stat           = tt$t,
  pvalue         = tt$P.Value,
  padj           = tt$adj.P.Val,
  dataset        = acc,
  row.names      = tt$gene_id
)
write.csv(out, out_csv)
sig <- sum(out$padj < 0.05 & abs(out$log2FoldChange) > 1, na.rm = TRUE)
cat(sprintf("  Genes: %d total, %d sig (padj<0.05, |LFC|>1)\n", nrow(out), sig))
cat(sprintf("  Results -> %s\n", out_csv))
