# Build Entrez-indexed STRING network for RA propagation.
# Reuses the same STRING files downloaded for IPF pipeline.
# Outputs: data/raw/string_entrez_edges_700.csv.gz
suppressPackageStartupMessages({
  library(org.Hs.eg.db)
  library(AnnotationDbi)
  library(data.table)
})

info_path  <- "data/raw/string_human_info.txt.gz"
ppi_path   <- "data/raw/string_human_ppi.txt.gz"
out_path   <- "data/raw/string_entrez_edges_700.csv.gz"

if (file.exists(out_path)) {
  cat("Already built:", out_path, "\n")
  quit(save = "no")
}

cat("Loading STRING info file...\n")
info <- fread(info_path, sep = "\t", select = c("#string_protein_id", "preferred_name"))
setnames(info, c("string_id", "gene_symbol"))

cat("Mapping gene symbols -> Entrez via org.Hs.eg.db...\n")
syms <- unique(info$gene_symbol)
res  <- suppressMessages(
  AnnotationDbi::select(org.Hs.eg.db, keys = syms,
                        columns = "ENTREZID", keytype = "SYMBOL")
)
res <- res[!is.na(res$ENTREZID) & !duplicated(res$SYMBOL), ]
sym2entrez <- setNames(res$ENTREZID, res$SYMBOL)

info$entrez <- sym2entrez[info$gene_symbol]
info <- info[!is.na(info$entrez)]
str2entrez <- setNames(info$entrez, info$string_id)

cat(sprintf("  Mapped %d / %d STRING proteins to Entrez\n",
            nrow(info), nrow(fread(info_path, select = 1))))

cat("Loading STRING PPI (score >= 700)...\n")
ppi <- fread(ppi_path, sep = " ")
ppi <- ppi[combined_score >= 700]

ppi$entrez1 <- str2entrez[ppi$protein1]
ppi$entrez2 <- str2entrez[ppi$protein2]
ppi <- ppi[!is.na(entrez1) & !is.na(entrez2)]
ppi <- ppi[, .(entrez1, entrez2)]
ppi <- unique(ppi)

cat(sprintf("  %d edges after Entrez mapping\n", nrow(ppi)))
fwrite(ppi, out_path)
cat("Saved:", out_path, "\n")
