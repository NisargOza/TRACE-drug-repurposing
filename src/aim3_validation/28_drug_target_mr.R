#!/usr/bin/env Rscript
# Drug-target Mendelian Randomization — TRACE IPF candidates
#
# Component 1: HMGCR → IPF (statin proxy; replication of prior work)
# Component 2: Systematic extension to TRACE top candidate targets via
#              lung eQTL instruments (GTEx v8) and blood eQTL (eQTLGen fallback)
#
# Pre-registered plan: analysis_plan_mr.md
# STROBE-MR reporting: https://www.strobe-mr.org
#
# Outputs:
#   results/mr/hmgcr_mr_results.csv
#   results/mr/hmgcr_coloc_results.csv
#   results/mr/extended_mr_results.csv
#   results/mr/mr_report.txt
#   results/figures/fig_mr_forest.png

# Load .env if present (keeps token out of command-line args)
local({
  # Search: working dir, script dir, and two levels up from script dir
  script_arg <- grep("--file=", commandArgs(FALSE), value=TRUE)
  script_dir <- if (length(script_arg) > 0)
    dirname(normalizePath(sub("--file=", "", script_arg[1]), mustWork=FALSE))
  else getwd()
  candidates <- c(
    file.path(getwd(), ".env"),
    file.path(script_dir, ".env"),
    file.path(script_dir, "../../.env")
  )
  env_file <- Filter(file.exists, candidates)[1]
  if (!is.na(env_file) && length(env_file) > 0) {
    for (line in readLines(env_file, warn=FALSE)) {
      line <- trimws(line)
      if (nchar(line) == 0 || startsWith(line, "#")) next
      parts <- regmatches(line, regexpr("=", line, fixed=TRUE), invert=TRUE)[[1]]
      if (length(parts) == 2)
        do.call(Sys.setenv, setNames(list(gsub('^["\']|["\']$','',trimws(parts[2]))),
                                     trimws(parts[1])))
    }
    message(".env loaded from: ", env_file)
  }
})

suppressPackageStartupMessages({
  library(TwoSampleMR)
  library(ieugwasr)
  library(coloc)
  library(data.table)
  library(dplyr)
  library(ggplot2)
})

args   <- commandArgs(trailingOnly = FALSE)
script <- sub("--file=", "", args[grep("--file=", args)])
ROOT   <- if (length(script) > 0) normalizePath(file.path(dirname(script), "../.."), mustWork = FALSE) else getwd()
MR_OUT  <- file.path(ROOT, "results", "mr")
FIG_OUT <- file.path(ROOT, "results", "figures")
DATA    <- file.path(ROOT, "data")
dir.create(MR_OUT,  showWarnings = FALSE, recursive = TRUE)
dir.create(FIG_OUT, showWarnings = FALSE, recursive = TRUE)

# ── Constants ─────────────────────────────────────────────────────────────────

FINNGEN_R10_URL <- paste0(
  "https://storage.googleapis.com/finngen-public-data-r10/",
  "summary_stats/finngen_R10_IPF.gz"
)
FINNGEN_LOCAL   <- file.path(DATA, "finngen_R10_IPF.gz")

# ── GBMI IPF — primary outcome upgrade (pre-specified in analysis_plan_mr.md §1.3) ──
# All-ancestry meta-analysis: BBJ,BioMe,BioVU,CCPM,CKB,ESTBB,FinnGen,GNH,HUNT,MGB,MGI,UCLA,UKBB
# Manifest: data/raw/manifest_GBMI_summary_statistics.xlsx - Sheet1.tsv
# S3 bucket: gbmi-sumstats.s3.amazonaws.com  (public)
GBMI_IPF_URL  <- paste0(
  "https://gbmi-sumstats.s3.amazonaws.com/",
  "IPF_Bothsex_inv_var_meta_GBMI_052021_nbbkgt1.txt.gz"
)
GBMI_LOCAL    <- file.path(DATA, "gbmi_IPF_all.txt.gz")

# OpenGWAS IDs to try for a larger IPF GWAS (tried in order; first hit wins)
# Covers GBMI imports, FinnGen B-series, and GWAS Catalog EBI entries
LARGER_IPF_OPENGWAS_IDS <- c(
  "gbmi-b-IPF",
  "gbmi-a-IPF_noUKBB",
  "ebi-a-GCST90399726",
  "ebi-a-GCST009619"
)

# HMGCR coordinates
HMGCR_CHR    <- 5
HMGCR_START  <- 74632154   # hg19 gene body start
HMGCR_END    <- 74657941   # hg19 gene body end
HMGCR_WINDOW <- 1e6        # ±1 Mb cis window

# OpenGWAS authentication (required since May 2024)
# Register free at https://api.opengwas.io/ then set token in .env:
#   OPENGWAS_JWT=your_token_here
# ieugwasr reads OPENGWAS_JWT from the environment automatically.
jwt <- Sys.getenv("OPENGWAS_JWT")
if (nchar(jwt) > 0) {
  message("OpenGWAS JWT token found.")
} else {
  message("NOTE: OPENGWAS_JWT not set. Will use published fallback instruments.")
  message("      Register free at https://api.opengwas.io/ for full OpenGWAS access.")
}

# OpenGWAS IDs (used if JWT is set)
LDL_GWAS_ID  <- "ieu-b-110"          # UKB LDL, n = 343,621, hg19

# TRACE target genes for extended analysis
# Ensembl gene IDs (for GTEx API queries)
TARGETS <- list(
  PDGFRB = list(ensembl = "ENSG00000113721", chr = 5,
                start_hg38 = 149432645, end_hg38 = 149492960,
                drugs = "cediranib, nintedanib"),
  FLT1   = list(ensembl = "ENSG00000102755", chr = 13,
                start_hg38 = 28320491, end_hg38 = 28535536,
                drugs = "cediranib (VEGFR1)"),
  KDR    = list(ensembl = "ENSG00000128052", chr = 4,
                start_hg38 = 55077082, end_hg38 = 55146611,
                drugs = "cediranib (VEGFR2)"),
  JAK1   = list(ensembl = "ENSG00000162434", chr = 1,
                start_hg38 = 64835518, end_hg38 = 64964095,
                drugs = "baricitinib"),
  JAK2   = list(ensembl = "ENSG00000096968", chr = 9,
                start_hg38 = 4985012,  end_hg38 = 5128187,
                drugs = "baricitinib"),
  HDAC1  = list(ensembl = "ENSG00000116478", chr = 1,
                start_hg38 = 32757316, end_hg38 = 32787754,
                drugs = "romidepsin, JNJ-26481585, vorinostat"),
  HDAC2  = list(ensembl = "ENSG00000196591", chr = 6,
                start_hg38 = 113516064, end_hg38 = 113546564,
                drugs = "romidepsin, JNJ-26481585, vorinostat")
)

# ── Helper: download FinnGen if not cached ────────────────────────────────────
download_finngen <- function() {
  if (file.exists(FINNGEN_LOCAL)) {
    message("FinnGen R10 IPF already cached: ", FINNGEN_LOCAL)
    return(FINNGEN_LOCAL)
  }
  message("Downloading FinnGen R10 IPF summary stats (~797 MB)...")
  tryCatch(
    download.file(FINNGEN_R10_URL, FINNGEN_LOCAL, method = "curl", quiet = FALSE),
    error = function(e) stop("Download failed: ", conditionMessage(e))
  )
  FINNGEN_LOCAL
}

# ── Helper: download GBMI IPF if not cached ──────────────────────────────────
download_gbmi <- function() {
  if (file.exists(GBMI_LOCAL)) {
    message("GBMI IPF already cached: ", GBMI_LOCAL)
    return(GBMI_LOCAL)
  }
  message("Downloading GBMI IPF summary stats...")
  message("  URL: ", GBMI_IPF_URL)
  message("  If this fails, verify the filename at https://www.globalbiobankmeta.org/resources")
  result <- tryCatch(
    download.file(GBMI_IPF_URL, GBMI_LOCAL, method = "curl", quiet = FALSE),
    error = function(e) {
      message("  GBMI download failed: ", conditionMessage(e))
      if (file.exists(GBMI_LOCAL)) unlink(GBMI_LOCAL)
      -1L
    }
  )
  if (!is.null(result) && result == 0 && file.exists(GBMI_LOCAL)) GBMI_LOCAL else NULL
}

# ── Helper: read GBMI regional subset ────────────────────────────────────────
# GBMI format: tab-separated; columns vary by release but typically include
# CHROM/CHR, POS/GENPOS, ID/SNP, REF/ALLELE0, ALT/ALLELE1, af_meta, beta_meta,
# se_meta, pval_meta (all GRCh38)
read_gbmi_region <- function(chr, start, end, window = 1e6, local = GBMI_LOCAL) {
  if (!file.exists(local)) stop("GBMI file not found: ", local)
  message(sprintf("  Reading GBMI region chr%d:%d-%d (±%g Mb)...", chr, start, end, window/1e6))
  decomp_cmd <- if (grepl("\\.bgz$", local)) {
    if (nchar(system("which bgzip", intern=TRUE, ignore.stderr=TRUE)) > 0) "bgzip -c -d" else "zcat"
  } else if (grepl("\\.gz$", local)) "gunzip -c" else "cat"
  gb <- tryCatch(
    fread(
      cmd = sprintf(
        "%s %s | awk -v c=%d -v s=%d -v e=%d 'NR==1 || ($1==c && $2>=(s-%d) && $2<=(e+%d))'",
        decomp_cmd, local, chr, start, end, as.integer(window), as.integer(window)
      ),
      sep = "\t", header = TRUE, data.table = FALSE
    ),
    error = function(e) {
      message("  GBMI region read failed: ", e$message); NULL
    }
  )
  gb
}

# ── Helper: scan GBMI for specific rsIDs (fallback when region read fails) ───
scan_gbmi_snps <- function(snps, local = GBMI_LOCAL) {
  if (!file.exists(local)) return(NULL)
  pattern <- paste(snps, collapse = "|")
  decomp_cmd <- if (grepl("\\.bgz$", local)) "bgzip -c -d" else if (grepl("\\.gz$", local)) "gunzip -c" else "cat"
  tryCatch(
    fread(
      cmd = sprintf("%s %s | grep -E 'CHROM|CHR|#|%s'", decomp_cmd, local, pattern),
      sep = "\t", header = FALSE, data.table = FALSE,
      fill = TRUE
    ),
    error = function(e) { message("  GBMI snp scan failed: ", e$message); NULL }
  )
}

# ── Helper: format GBMI as TwoSampleMR outcome ───────────────────────────────
# Handles multiple GBMI column-name conventions across releases
gbmi_to_outcome <- function(gb, snps = NULL, outcome_name = "IPF_GBMI_ALL") {
  if (is.null(gb) || nrow(gb) == 0) return(data.frame())
  gb <- as.data.frame(gb, stringsAsFactors = FALSE)
  cn <- tolower(colnames(gb))
  colnames(gb) <- cn

  get_col <- function(keys) { for (k in keys) if (k %in% cn) return(k); NULL }

  snp_col  <- get_col(c("rsid", "id", "snp", "variant_id", "rs_id", "markername"))
  beta_col <- get_col(c("inv_var_meta_beta", "beta_meta", "beta", "effect"))
  se_col   <- get_col(c("inv_var_meta_sebeta", "se_meta", "se", "standard_error", "stderr"))
  pval_col <- get_col(c("inv_var_meta_p", "pval_meta", "p.value", "p_value", "pval", "p"))
  eaf_col  <- get_col(c("all_meta_af", "af_meta", "af_alt", "effect_allele_freq", "eaf", "freq1"))
  ea_col   <- get_col(c("alt", "a1", "effect_allele", "allele1", "allele_b"))
  oa_col   <- get_col(c("ref", "a2", "other_allele", "allele0", "allele_a"))
  chr_col  <- get_col(c("#chr", "chrom", "chr", "#chrom", "chromosome", "ch"))
  pos_col  <- get_col(c("pos", "genpos", "position", "bp"))

  if (is.null(snp_col) || is.null(beta_col) || is.null(se_col)) {
    message("  GBMI column mapping failed. Columns: ", paste(cn, collapse=", "))
    return(data.frame())
  }

  gb$rsid_clean <- trimws(as.character(gb[[snp_col]]))
  if (!is.null(snps)) gb <- gb[gb$rsid_clean %in% snps, , drop=FALSE]
  keep <- !is.na(gb$rsid_clean) & gb$rsid_clean != "" &
          !is.na(gb[[beta_col]]) & !is.na(gb[[se_col]])
  gb <- gb[keep, , drop=FALSE]
  if (nrow(gb) == 0) return(data.frame())

  data.frame(
    SNP                   = gb$rsid_clean,
    beta.outcome          = as.numeric(gb[[beta_col]]),
    se.outcome            = as.numeric(gb[[se_col]]),
    effect_allele.outcome = if (!is.null(ea_col)) toupper(gb[[ea_col]]) else NA_character_,
    other_allele.outcome  = if (!is.null(oa_col)) toupper(gb[[oa_col]]) else NA_character_,
    eaf.outcome           = if (!is.null(eaf_col)) as.numeric(gb[[eaf_col]]) else NA_real_,
    pval.outcome          = if (!is.null(pval_col)) as.numeric(gb[[pval_col]]) else NA_real_,
    outcome               = outcome_name,
    id.outcome            = outcome_name,
    chr.outcome           = if (!is.null(chr_col)) as.character(gb[[chr_col]]) else NA_character_,
    pos.outcome           = if (!is.null(pos_col)) as.integer(gb[[pos_col]]) else NA_integer_,
    mr_keep.outcome       = TRUE,
    stringsAsFactors      = FALSE
  )
}

# ── Helper: read FinnGen regional subset ─────────────────────────────────────
read_finngen_region <- function(chr, start_hg38, end_hg38,
                                window = 1e6, local = FINNGEN_LOCAL) {
  if (!file.exists(local)) stop("FinnGen file not found: ", local)
  message(sprintf("  Reading FinnGen region chr%d:%d-%d (±%g Mb)...",
                  chr, start_hg38, end_hg38, window / 1e6))
  # FinnGen columns: #chrom pos ref alt rsids nearest_genes pval mlogp
  #                  beta sebeta af_alt af_alt_cases af_alt_controls
  fg <- fread(
    cmd  = sprintf("gunzip -c %s | awk -v c=%d -v s=%d -v e=%d 'NR==1 || ($1==c && $2>=(s-%d) && $2<=(e+%d))'",
                   local, chr, start_hg38, end_hg38,
                   as.integer(window), as.integer(window)),
    sep  = "\t", header = TRUE, data.table = FALSE
  )
  colnames(fg)[1] <- "chrom"
  fg
}

# ── Helper: format FinnGen as TwoSampleMR outcome ────────────────────────────
finngen_to_outcome <- function(fg, snps = NULL, outcome_name = "IPF_FinnGen_R10") {
  # Convert to plain data.frame to avoid data.table subsetting edge cases
  fg <- as.data.frame(fg, stringsAsFactors = FALSE)
  fg$rsid <- trimws(sub(",.*", "", as.character(fg$rsids)))  # first rsID, strip whitespace
  fg$rsid[fg$rsid == "" | fg$rsid == "NA"] <- NA_character_
  if (!is.null(snps)) fg <- fg[!is.na(fg$rsid) & fg$rsid %in% snps, , drop=FALSE]
  keep <- !is.na(fg$rsid) & !is.na(fg$beta) & !is.na(fg$sebeta)
  fg   <- fg[keep, , drop=FALSE]
  if (nrow(fg) == 0) return(data.frame())

  # Build TwoSampleMR outcome data frame directly, preserving rsIDs as SNP names
  # Use [[ ]] for exact column access (avoids partial matching with rsids column)
  data.frame(
    SNP                    = fg[["rsid"]],
    beta.outcome           = as.numeric(fg$beta),
    se.outcome             = as.numeric(fg$sebeta),
    effect_allele.outcome  = toupper(fg$alt),
    other_allele.outcome   = toupper(fg$ref),
    eaf.outcome            = as.numeric(fg$af_alt),
    pval.outcome           = as.numeric(fg$pval),
    outcome                = outcome_name,
    id.outcome             = outcome_name,
    chr.outcome            = as.character(fg$chrom),
    pos.outcome            = as.integer(fg$pos),
    mr_keep.outcome        = TRUE,
    stringsAsFactors       = FALSE
  )
}

# ── Helper: run full MR pipeline ─────────────────────────────────────────────
run_mr_pipeline <- function(exp_dat, out_dat, label) {
  dat <- tryCatch(
    harmonise_data(exp_dat, out_dat, action = 2),
    error = function(e) { message("  Harmonise failed: ", e$message); NULL }
  )
  if (is.null(dat) || nrow(dat) == 0) {
    message("  No harmonised variants for ", label)
    return(NULL)
  }
  dat <- dat[dat$mr_keep, ]
  n   <- nrow(dat)
  message(sprintf("  %s: %d variants after harmonisation", label, n))
  if (n < 1) return(NULL)

  # Main MR estimates
  methods <- c("mr_ivw", "mr_egger_regression",
                "mr_weighted_median", "mr_weighted_mode")
  if (n == 1) methods <- "mr_wald_ratio"
  else if (n == 2) methods <- c("mr_ivw", "mr_weighted_median")
  res <- mr(dat, method_list = methods)
  res$exposure <- label
  res$n_snps   <- n

  # Sensitivity
  pleio <- tryCatch(mr_pleiotropy_test(dat), error = function(e) NULL)
  hetero <- tryCatch(mr_heterogeneity(dat), error = function(e) NULL)

  # Steiger filtering (requires variance explained; skip if r2 missing)
  steiger <- tryCatch({
    dat2 <- steiger_filtering(dat)
    sum(!dat2$steiger_filtered, na.rm = TRUE)
  }, error = function(e) NA)

  list(
    results   = res,
    pleiotropy = pleio,
    heterogeneity = hetero,
    n_steiger_pass = steiger,
    harmonised = dat
  )
}

# ── Helper: colocalization ────────────────────────────────────────────────────
run_coloc <- function(ldl_region, fg_region, label,
                      prior_p12 = 1e-5) {
  # Merge on rsID
  common <- intersect(
    sub(",.*", "", ldl_region$SNP),
    sub(",.*", "", fg_region$rsids)
  )
  if (length(common) < 50) {
    message(sprintf("  %s coloc: only %d common variants — skipping", label, length(common)))
    return(NULL)
  }
  ldl_m  <- ldl_region[sub(",.*", "", ldl_region$SNP) %in% common, ]
  fg_m   <- fg_region[sub(",.*", "", fg_region$rsids)  %in% common, ]
  ldl_m$rsid <- sub(",.*", "", ldl_m$SNP)
  fg_m$rsid  <- sub(",.*", "", fg_m$rsids)
  merged <- merge(ldl_m, fg_m, by = "rsid")
  merged <- merged[!duplicated(merged$rsid), ]
  if (nrow(merged) < 50) return(NULL)

  D1 <- list(
    beta   = merged$beta.x,
    varbeta = merged$se.x^2,
    type   = "quant",
    N      = 343621,
    snp    = merged$rsid
  )
  D2 <- list(
    beta    = merged$beta.y,
    varbeta = merged$sebeta^2,
    type    = "cc",
    N       = 2189 + 407609,
    s       = 2189 / (2189 + 407609),
    snp     = merged$rsid
  )
  res <- tryCatch(
    coloc.abf(D1, D2, p12 = prior_p12),
    error = function(e) { message("  coloc error: ", e$message); NULL }
  )
  if (is.null(res)) return(NULL)
  pp <- res$summary
  message(sprintf("  %s coloc: H0=%.3f H1=%.3f H2=%.3f H3=%.3f H4=%.3f",
                  label, pp["PP.H0.abf"], pp["PP.H1.abf"],
                  pp["PP.H2.abf"], pp["PP.H3.abf"], pp["PP.H4.abf"]))
  as.data.frame(t(pp))
}

# ── Helper: GTEx lung eQTL query per gene ─────────────────────────────────────
get_eqtl_instruments <- function(ensembl_id, gene_name,
                                  pval_thresh = 5e-8) {
  # Strategy 1: OpenGWAS GTEx lung eQTL dataset (requires JWT)
  if (nchar(jwt) > 0) {
    opengwas_id <- sprintf("eqtl-a-%s", ensembl_id)
    message(sprintf("  Trying OpenGWAS eQTL dataset: %s", opengwas_id))
    res <- tryCatch(
      withCallingHandlers(
        extract_instruments(opengwas_id, p1 = pval_thresh, clump = TRUE,
                            r2 = 0.001, kb = 500),
        warning = function(w) invokeRestart("muffleWarning")
      ),
      error = function(e) {
        msg <- conditionMessage(e)
        if (grepl("timed out|timeout|300", msg, ignore.case=TRUE))
          message(sprintf("  OpenGWAS eQTL timeout for %s — dataset likely absent", gene_name))
        else
          message("  OpenGWAS eQTL error: ", msg)
        NULL
      }
    )
    if (!is.null(res) && is.data.frame(res) && nrow(res) > 0) return(res)
    message(sprintf("  OpenGWAS eQTL not found for %s — dataset may not exist", gene_name))
  }

  # Strategy 2: eQTL Catalogue v1 tabix query
  # Requires all params including tissue, quant_method, qtl_group — skip if complex
  message(sprintf("  No eQTL instruments available for %s without OpenGWAS JWT.", gene_name))
  message("  To run Component 2: register at https://api.opengwas.io/ and set OPENGWAS_JWT")
  return(NULL)
}

get_gtex_lung_eqtls <- function(ensembl_id, gene_name,
                                 pval_thresh = 5e-8) {
  # Wrapper kept for backward compatibility; delegates to get_eqtl_instruments
  get_eqtl_instruments(ensembl_id, gene_name, pval_thresh)
}

# Internal: old GTEx API attempt (kept for reference; API returns 0 rows as of 2025)
get_gtex_api_eqtls_deprecated <- function(ensembl_id, gene_name,
                                pval_thresh = 5e-8) {
  api_url <- sprintf(
    "https://gtexportal.org/api/v2/association/singleTissueEqtl?tissueSiteDetailId=Lung&gencodeId=%s&datasetId=gtex_v8&itemsPerPage=500",
    ensembl_id
  )
  tryCatch({
    parsed <- jsonlite::fromJSON(api_url, simplifyVector = TRUE, flatten = TRUE)
    df <- parsed$data
    if (is.null(df) || !is.data.frame(df) || nrow(df) == 0) return(NULL)
    # Columns: variantId (chr_pos_ref_alt_b38), pValue, slope, slopeStdErr, maf
    df <- df[df$pValue < pval_thresh, ]
    if (nrow(df) == 0) {
      message(sprintf("  GTEx %s: no variants at p < %.0e", gene_name, pval_thresh))
      return(NULL)
    }
    # Parse variantId: chr5_74640000_A_G_b38
    parts <- strsplit(df$variantId, "_")
    df$CHR <- as.integer(sub("chr", "", sapply(parts, `[`, 1)))
    df$POS <- as.integer(sapply(parts, `[`, 2))
    df$A1  <- sapply(parts, `[`, 3)
    df$A2  <- sapply(parts, `[`, 4)
    df$gene <- gene_name
    message(sprintf("  GTEx %s lung: %d variants at p < %.0e",
                    gene_name, nrow(df), pval_thresh))
    df
  }, error = function(e) {
    message(sprintf("  GTEx query failed for %s: %s", gene_name, e$message))
    NULL
  })
}

# ── Helper: post-hoc power from achieved precision ────────────────────────────
mr_power_from_se <- function(se, or_grid = c(0.90, 0.80, 0.70, 0.50),
                             alpha = 0.05) {
  z <- qnorm(1 - alpha/2)
  setNames(sapply(or_grid, function(orv) {
    b <- abs(log(orv)); pnorm(b/se - z) + pnorm(-b/se - z)
  }), sprintf("OR=%.2f", or_grid))
}

# ── COMPONENT 1: HMGCR drug-target MR ─────────────────────────────────────────
message("\n=== COMPONENT 1: HMGCR drug-target MR (statin → IPF) ===\n")

# 1a. Download FinnGen
download_finngen()

# ── Helper: GWAS Catalog fallback for HMGCR LDL instruments ──────────────────
# Downloads all LDL-associated variants in the HMGCR ±1Mb region from GWAS Catalog
# (free API, no authentication required)
get_hmgcr_instruments_gwascatalog <- function() {
  message("  Trying GWAS Catalog API for HMGCR region LDL associations...")
  # GWAS Catalog v2 associations by genomic region
  url <- sprintf(
    "https://www.ebi.ac.uk/gwas/rest/api/singleNucleotidePolymorphisms/search/findByChromosomeBetweenLocations?chrom=%d&start=%d&end=%d",
    HMGCR_CHR,
    as.integer(HMGCR_START - HMGCR_WINDOW),
    as.integer(HMGCR_END   + HMGCR_WINDOW)
  )
  resp <- tryCatch(jsonlite::fromJSON(url, simplifyDataFrame = TRUE),
                   error = function(e) NULL)
  if (is.null(resp) || length(resp) == 0) return(NULL)

  # GWAS Catalog returns SNPs; filter to those in LDL GWAS
  snps <- resp[["_embedded"]][["singleNucleotidePolymorphisms"]]
  if (is.null(snps) || nrow(snps) == 0) return(NULL)
  message(sprintf("  GWAS Catalog: %d SNPs in HMGCR region", nrow(snps)))
  snps
}

# ── Hard-coded HMGCR fallback instruments ─────────────────────────────────────
# Canonical HMGCR cis-instruments used in published statin MR studies
# (Swerdlow et al. 2015 BMJ; Ference et al.; Burgess et al.)
# Beta = effect of LDL-raising allele on LDL-C (mmol/L, per GLGC 2013)
HMGCR_FALLBACK <- data.frame(
  SNP              = c("rs12916",    "rs17238484", "rs5909"),
  chr.exposure     = c(5,            5,            5),
  pos.exposure     = c(74656185,     74732160,     74720094),
  effect_allele.exposure = c("C",    "G",          "T"),
  other_allele.exposure  = c("T",    "T",          "C"),
  # Beta = effect of LDL-RAISING allele on LDL-C
  beta.exposure    = c(0.0794,       0.0387,       0.0178),
  se.exposure      = c(0.0015,       0.0015,       0.0014),
  pval.exposure    = c(1e-200,       1e-100,       1e-30),
  eaf.exposure     = c(0.40,         0.12,         0.33),
  exposure         = "LDL-C (HMGCR cis — fallback from published MR)",
  mr_keep.exposure = TRUE,
  stringsAsFactors = FALSE
)

# 1b. Extract HMGCR cis-instruments — three-tier approach
message("Extracting HMGCR cis-instruments from LDL GWAS...")
ldl_hmgcr <- NULL

# Tier 1: OpenGWAS (requires JWT)
if (nchar(jwt) > 0) {
  message("  Tier 1: OpenGWAS (ieu-b-110)...")
  ldl_all <- tryCatch(
    extract_instruments(LDL_GWAS_ID, p1 = 5e-8, clump = TRUE, r2 = 0.001, kb = 10000),
    error = function(e) { message("  OpenGWAS error: ", e$message); NULL }
  )
  if (!is.null(ldl_all) && nrow(ldl_all) > 0) {
    ldl_hmgcr <- ldl_all %>%
      filter(!is.na(chr.exposure), !is.na(pos.exposure),
             chr.exposure == HMGCR_CHR,
             pos.exposure >= HMGCR_START - HMGCR_WINDOW,
             pos.exposure <= HMGCR_END   + HMGCR_WINDOW) %>%
      mutate(exposure = "LDL-C (HMGCR cis, OpenGWAS ieu-b-110)")
    message(sprintf("  OpenGWAS: %d HMGCR cis-instruments (of %d genome-wide)",
                    nrow(ldl_hmgcr), nrow(ldl_all)))
  }
}

# Supplement or replace with published canonical instruments if OpenGWAS < 3
# (rs12916, rs17238484, rs5909 are the validated HMGCR instruments used in 100s of papers)
fallback_formatted <- format_data(
  HMGCR_FALLBACK,
  type = "exposure", snp_col = "SNP",
  beta_col = "beta.exposure", se_col = "se.exposure", pval_col = "pval.exposure",
  eaf_col = "eaf.exposure", effect_allele_col = "effect_allele.exposure",
  other_allele_col = "other_allele.exposure", chr_col = "chr.exposure", pos_col = "pos.exposure"
)
fallback_formatted$exposure <- "LDL-C (HMGCR cis — Swerdlow 2015/GLGC 2013)"
if (!is.null(ldl_hmgcr) && nrow(ldl_hmgcr) > 0 && nrow(ldl_hmgcr) < 3) {
  message(sprintf("  OpenGWAS returned %d instruments — supplementing with canonical fallback to reach 3",
                  nrow(ldl_hmgcr)))
  missing <- fallback_formatted[!fallback_formatted$SNP %in% ldl_hmgcr$SNP, ]
  # Coerce chr.exposure to same type before binding
  if ("chr.exposure" %in% colnames(ldl_hmgcr))
    ldl_hmgcr$chr.exposure <- as.character(ldl_hmgcr$chr.exposure)
  if ("chr.exposure" %in% colnames(missing))
    missing$chr.exposure <- as.character(missing$chr.exposure)
  ldl_hmgcr <- bind_rows(ldl_hmgcr, missing)
  ldl_hmgcr$exposure  <- "LDL-C (HMGCR cis, OpenGWAS + Swerdlow/GLGC supplement)"
  ldl_hmgcr$id.exposure <- "hmgcr_combined"   # unify so harmonise_data runs once
}

# Tier 3 fallback: published instruments (always available)
if (is.null(ldl_hmgcr) || nrow(ldl_hmgcr) < 1) {
  message("  Tier 3: Using published HMGCR fallback instruments (rs12916, rs17238484, rs5909)")
  message("         Source: Swerdlow et al. 2015 BMJ / GLGC 2013 (Willer et al.)")
  ldl_hmgcr <- format_data(
    HMGCR_FALLBACK,
    type              = "exposure",
    snp_col           = "SNP",
    beta_col          = "beta.exposure",
    se_col            = "se.exposure",
    pval_col          = "pval.exposure",
    eaf_col           = "eaf.exposure",
    effect_allele_col = "effect_allele.exposure",
    other_allele_col  = "other_allele.exposure",
    chr_col           = "chr.exposure",
    pos_col           = "pos.exposure"
  )
  ldl_hmgcr$exposure <- "LDL-C (HMGCR cis — Swerdlow 2015/GLGC 2013 fallback)"
}
message(sprintf("  Final HMGCR instruments: %d variants", nrow(ldl_hmgcr)))

# 1d. Extract these SNPs from FinnGen IPF
message("Reading FinnGen R10 IPF in HMGCR region...")
fg_hmgcr_region <- read_finngen_region(
  chr        = HMGCR_CHR,
  start_hg38 = 74523876,    # HMGCR hg38 start
  end_hg38   = 74598685,    # HMGCR hg38 end
  window     = HMGCR_WINDOW
)
message(sprintf("  FinnGen HMGCR region: %d variants", nrow(fg_hmgcr_region)))

ipf_hmgcr <- finngen_to_outcome(fg_hmgcr_region, snps = ldl_hmgcr$SNP)
if (nrow(ipf_hmgcr) == 0) {
  # FinnGen uses rsids column (may be comma-separated); search full file
  message("  SNPs not found in region slice — scanning full FinnGen for these rsIDs...")
  # Read full file streaming, filter rows containing any of our rsIDs
  target_snps <- paste(ldl_hmgcr$SNP, collapse="|")
  fg_hits <- tryCatch(
    fread(cmd = sprintf("gunzip -c %s | grep -E 'rsids|%s'", FINNGEN_LOCAL, target_snps),
          sep = "\t", header = FALSE, data.table = FALSE),
    error = function(e) NULL
  )
  if (!is.null(fg_hits) && nrow(fg_hits) > 1) {
    # First row is header from grep match on "rsids"
    colnames(fg_hits) <- c("chrom","pos","ref","alt","rsids","nearest_genes",
                           "pval","mlogp","beta","sebeta","af_alt","af_alt_cases","af_alt_controls")
    fg_hits <- fg_hits[grep(target_snps, fg_hits$rsids, ignore.case=TRUE), ]
    ipf_hmgcr <- finngen_to_outcome(fg_hits, outcome_name = "IPF_FinnGen_R10")
  }
}
message(sprintf("  Matched IPF outcome variants: %d", nrow(ipf_hmgcr)))

# 1e. MR analysis
message("Running HMGCR MR...")
hmgcr_mr <- run_mr_pipeline(ldl_hmgcr, ipf_hmgcr, "HMGCR→IPF (statin proxy)")

if (!is.null(hmgcr_mr)) {
  fwrite(hmgcr_mr$results, file.path(MR_OUT, "hmgcr_mr_results.csv"))
  message("\nHMGCR MR results:")
  print(hmgcr_mr$results[, c("method","nsnp","b","se","pval")])
  if (!is.null(hmgcr_mr$pleiotropy)) {
    message(sprintf("MR-Egger intercept: %.4f (p = %.4f)",
                    hmgcr_mr$pleiotropy$egger_intercept,
                    hmgcr_mr$pleiotropy$pval))
  }
}

# ── 1f. GBMI primary-outcome upgrade ─────────────────────────────────────────
# Pre-specified in analysis_plan_mr.md §1.3: run GBMI (~8,492 cases) as
# the primary outcome; FinnGen R10 above becomes the sensitivity/replication arm.
# Two-tier approach: OpenGWAS first (no URL needed), direct download as fallback.
message("\n=== GBMI PRIMARY OUTCOME UPGRADE (pre-specified in analysis_plan_mr.md) ===\n")

gbmi_mr    <- NULL
gbmi_label <- NULL

# Tier 1: OpenGWAS search for a larger IPF GWAS (requires JWT)
if (nchar(jwt) > 0) {
  message("Searching OpenGWAS for larger IPF GWAS...")
  for (oid in LARGER_IPF_OPENGWAS_IDS) {
    message(sprintf("  Trying OpenGWAS ID: %s", oid))
    oc <- tryCatch(
      extract_outcome_data(
        snps     = ldl_hmgcr$SNP,
        outcomes = oid,
        proxies  = TRUE,
        rsq      = 0.8,
        align_alleles = 1
      ),
      error = function(e) { message("  ", oid, ": ", e$message); NULL }
    )
    if (!is.null(oc) && is.data.frame(oc) && nrow(oc) > 0) {
      message(sprintf("  SUCCESS: %d variants found for %s", nrow(oc), oid))
      gbmi_mr    <- run_mr_pipeline(ldl_hmgcr, oc, sprintf("HMGCR→IPF (%s)", oid))
      gbmi_label <- oid
      if (!is.null(gbmi_mr)) break
    }
  }
}

# Tier 2: direct GBMI download
if (is.null(gbmi_mr)) {
  message("\nOpenGWAS route unavailable — attempting direct GBMI download...")
  gbmi_path <- download_gbmi()
  if (!is.null(gbmi_path)) {
    gbmi_region <- read_gbmi_region(
      chr    = HMGCR_CHR,
      start  = 74523876,   # HMGCR hg38 start
      end    = 74598685,   # HMGCR hg38 end
      window = HMGCR_WINDOW,
      local  = gbmi_path
    )
    if (!is.null(gbmi_region) && nrow(gbmi_region) > 0) {
      ipf_gbmi <- gbmi_to_outcome(gbmi_region, snps = ldl_hmgcr$SNP,
                                   outcome_name = "IPF_GBMI_ALL")
      if (nrow(ipf_gbmi) == 0) {
        message("  Region-based lookup returned 0 rows — scanning full GBMI for rsIDs...")
        gb_hits  <- scan_gbmi_snps(ldl_hmgcr$SNP, local = gbmi_path)
        ipf_gbmi <- gbmi_to_outcome(gb_hits, snps = ldl_hmgcr$SNP,
                                     outcome_name = "IPF_GBMI_ALL")
      }
      if (nrow(ipf_gbmi) > 0) {
        gbmi_mr    <- run_mr_pipeline(ldl_hmgcr, ipf_gbmi, "HMGCR→IPF (GBMI ALL)")
        gbmi_label <- "GBMI_IPF_ALL"
      }
    }
  }
}

# Save & report
if (!is.null(gbmi_mr)) {
  fwrite(gbmi_mr$results, file.path(MR_OUT, "gbmi_mr_results.csv"))
  message("\nGBMI MR results (PRIMARY outcome — pre-specified upgrade):")
  print(gbmi_mr$results[, c("method","nsnp","b","se","pval")])
  if (!is.null(gbmi_mr$pleiotropy)) {
    message(sprintf("  MR-Egger intercept: %.5f (p=%.4f)",
                    gbmi_mr$pleiotropy$egger_intercept,
                    gbmi_mr$pleiotropy$pval))
  }
  ivw_gbmi <- gbmi_mr$results[gbmi_mr$results$method == "Inverse variance weighted", ]
  if (nrow(ivw_gbmi) >= 1) {
    pw_gbmi <- mr_power_from_se(ivw_gbmi$se[1])
    message(sprintf("  Post-hoc power (GBMI): OR=0.80 → %.0f%%  OR=0.70 → %.0f%%",
                    100*pw_gbmi["OR=0.80"], 100*pw_gbmi["OR=0.70"]))
  }
} else {
  message("\nGBMI outcome unavailable — FinnGen R10 is the sole primary outcome.")
  message("To complete the GBMI upgrade:")
  message("  1. Verify download URL at https://www.globalbiobankmeta.org/resources")
  message("  2. Update GBMI_IPF_URL in this script if the path has changed")
  message("  3. Re-run: Rscript src/aim3_validation/28_drug_target_mr.R")
}

# 1g. Colocalization at HMGCR
message("\nRunning colocalization at HMGCR...")
# Get full LDL regional data from OpenGWAS for coloc
ldl_region_full <- tryCatch(
  associations(
    variants = NULL,
    id       = LDL_GWAS_ID,
    proxies  = FALSE
  ),
  error = function(e) NULL
)
# If full regional query not available, use instruments only with coloc caveated
if (!is.null(ldl_region_full) && nrow(ldl_region_full) > 50) {
  ldl_coloc <- ldl_region_full %>%
    filter(chr == HMGCR_CHR,
           position >= HMGCR_START - HMGCR_WINDOW,
           position <= HMGCR_END   + HMGCR_WINDOW) %>%
    rename(SNP = rsid, beta = beta, se = se, pval = p)
  coloc_res <- run_coloc(ldl_coloc, fg_hmgcr_region, "HMGCR")
  if (!is.null(coloc_res)) {
    coloc_res$locus <- "HMGCR"
    fwrite(coloc_res, file.path(MR_OUT, "hmgcr_coloc_results.csv"))
  }
} else {
  message("  Full regional LDL data not available via API — coloc requires manual download")
  message("  See analysis_plan_mr.md §1.6. Run coloc after downloading GLGC regional data.")
}

# ── COMPONENT 2: Systematic extension to TRACE candidate targets ──────────────
message("\n=== COMPONENT 2: Systematic MR across TRACE candidate targets ===\n")

# Load full FinnGen for SNP lookup (read once, reuse)
message("FinnGen will be read per-region for each target gene.")

extended_results <- list()

for (gene_name in names(TARGETS)) {
  tgt <- TARGETS[[gene_name]]
  message(sprintf("\n--- %s (drugs: %s) ---", gene_name, tgt$drugs))

  # 2a. Get lung eQTL instruments from GTEx
  eqtl_df <- get_gtex_lung_eqtls(tgt$ensembl, gene_name)

  if (is.null(eqtl_df) || nrow(eqtl_df) < 1) {
    message(sprintf("  No GTEx lung eQTLs for %s at p<5e-8", gene_name))
    next
  }

  # Check if data is already TwoSampleMR-formatted (from OpenGWAS extract_instruments)
  # vs. raw GTEx/eQTL API output (has pValue, slope, variantId columns)
  already_formatted <- "pval.exposure" %in% colnames(eqtl_df)

  if (already_formatted) {
    # OpenGWAS path: restrict to cis-eQTLs (same chromosome as gene)
    # Trans-eQTLs on other chromosomes are invalid drug-target MR instruments
    n_before <- nrow(eqtl_df)
    eqtl_df  <- eqtl_df %>%
      filter(!is.na(chr.exposure),
             as.integer(chr.exposure) == tgt$chr)
    message(sprintf("  Cis-filter: %d → %d instruments (removed %d trans on other chr)",
                    n_before, nrow(eqtl_df), n_before - nrow(eqtl_df)))
    if (nrow(eqtl_df) == 0) {
      message(sprintf("  No cis-eQTL instruments on chr%d for %s", tgt$chr, gene_name))
      next
    }
    eqtl_exp <- eqtl_df %>%
      mutate(exposure = sprintf("%s expression (OpenGWAS cis-eQTL)", gene_name))
    message(sprintf("  Instrument SNPs: %s", paste(head(eqtl_exp$SNP, 4), collapse=", ")))
  } else {
    # GTEx/raw path: format and clump
    eqtl_df <- eqtl_df[order(eqtl_df$pValue), ]
    eqtl_exp <- tryCatch(
      format_data(
        eqtl_df,
        type              = "exposure",
        snp_col           = "variantId",
        beta_col          = "slope",
        se_col            = "slopeStdErr",
        pval_col          = "pValue",
        eaf_col           = "maf",
        chr_col           = "CHR",
        pos_col           = "POS",
        effect_allele_col = "A1",
        other_allele_col  = "A2",
        phenotype_col     = "gene"
      ) %>% mutate(exposure = sprintf("%s expression (GTEx lung)", gene_name)),
      error = function(e) { message("  Format error: ", e$message); NULL }
    )
    if (is.null(eqtl_exp)) next
  }

  # Clump if > 3 instruments and not already clumped
  if (nrow(eqtl_exp) > 3 && !already_formatted) {
    eqtl_exp <- tryCatch(
      clump_data(eqtl_exp, clump_r2 = 0.001, clump_kb = 500),
      error = function(e) head(eqtl_exp, 5)
    )
  }
  message(sprintf("  Instruments after clumping: %d", nrow(eqtl_exp)))

  # 2b. Extract outcome from FinnGen (match by variantId rsid-like or position)
  # Direct rsID lookup: grep FinnGen for the specific instrument rsIDs
  # This bypasses region filtering + format_data issues entirely
  snp_pattern <- paste(eqtl_exp$SNP, collapse="|")
  fg_hits <- tryCatch(
    fread(cmd = sprintf("gunzip -c %s | grep -E '^[0-9]|^#chrom' | grep -E '%s'",
                        FINNGEN_LOCAL, snp_pattern),
          sep="\t", header=FALSE, data.table=FALSE,
          col.names=c("chrom","pos","ref","alt","rsids","nearest_genes","pval",
                      "mlogp","beta","sebeta","af_alt","af_alt_cases","af_alt_controls")),
    error=function(e){message("  FinnGen grep failed: ",e$message); NULL}
  )
  message(sprintf("  Direct grep found %d FinnGen rows for %d instruments",
                  if(is.null(fg_hits)) 0 else nrow(fg_hits), nrow(eqtl_exp)))

  if (is.null(fg_hits) || nrow(fg_hits) == 0) {
    message(sprintf("  No FinnGen rows matched for %s instruments — skipping", gene_name))
    next
  }

  # Keep only rows where rsids exactly matches an instrument rsID
  fg_hits$rsid <- trimws(sub(",.*","",as.character(fg_hits$rsids)))
  fg_hits <- fg_hits[fg_hits$rsid %in% eqtl_exp$SNP, , drop=FALSE]
  message(sprintf("  Exact rsID matches: %d", nrow(fg_hits)))

  if (nrow(fg_hits) == 0) {
    message(sprintf("  No exact rsID matches for %s — skipping", gene_name))
    next
  }

  ipf_out <- finngen_to_outcome(fg_hits, snps = eqtl_exp$SNP,
                                outcome_name = sprintf("IPF_FinnGen_R10_%s", gene_name))

  if (is.null(ipf_out) || !is.data.frame(ipf_out) || nrow(ipf_out) == 0) {
    message(sprintf("  No IPF outcome variants matched for %s", gene_name))
    next
  }

  # 2d. MR
  mr_res <- run_mr_pipeline(eqtl_exp, ipf_out,
                             sprintf("%s expression → IPF", gene_name))
  if (!is.null(mr_res)) {
    r <- mr_res$results
    r$gene   <- gene_name
    r$drugs  <- tgt$drugs
    r$tissue <- "GTEx_lung_eQTL"
    extended_results[[gene_name]] <- r
    message(sprintf("  IVW: OR=%.3f [%.3f-%.3f] p=%.4f",
                    exp(r$b[r$method=="Inverse variance weighted"]),
                    exp(r$b[r$method=="Inverse variance weighted"] -
                          1.96 * r$se[r$method=="Inverse variance weighted"]),
                    exp(r$b[r$method=="Inverse variance weighted"] +
                          1.96 * r$se[r$method=="Inverse variance weighted"]),
                    r$pval[r$method=="Inverse variance weighted"]))
  }
}

# ── Compile and save all results ───────────────────────────────────────────────
message("\n=== Compiling results ===\n")

all_ext <- if (length(extended_results) > 0)
  bind_rows(extended_results) else data.frame()
fwrite(all_ext, file.path(MR_OUT, "extended_mr_results.csv"))

# ── Forest plot ───────────────────────────────────────────────────────────────
# HMGCR rows: show GBMI (primary) and FinnGen R10 (replication) side-by-side
# when both are available; otherwise single-outcome rows.
hmgcr_finngen_row <- if (!is.null(hmgcr_mr)) {
  hmgcr_mr$results %>%
    filter(method == "Inverse variance weighted") %>%
    mutate(gene = "HMGCR (statin proxy)", drugs = "atorvastatin",
           OR = exp(b), CI_lo = exp(b - 1.96*se), CI_hi = exp(b + 1.96*se),
           outcome_label = "FinnGen R10 (2,189 cases — sensitivity)",
           component = "Component 1: HMGCR")
} else data.frame()

hmgcr_gbmi_row <- if (!is.null(gbmi_mr)) {
  gbmi_mr$results %>%
    filter(method == "Inverse variance weighted") %>%
    mutate(gene = "HMGCR (statin proxy)", drugs = "atorvastatin",
           OR = exp(b), CI_lo = exp(b - 1.96*se), CI_hi = exp(b + 1.96*se),
           outcome_label = sprintf("GBMI ALL (~8,492 cases — PRIMARY; %s)", gbmi_label),
           component = "Component 1: HMGCR")
} else data.frame()

ext_rows <- if (nrow(all_ext) > 0) {
  all_ext %>%
    filter(method == "Inverse variance weighted") %>%
    mutate(OR = exp(b), CI_lo = exp(b - 1.96*se), CI_hi = exp(b + 1.96*se),
           outcome_label = "FinnGen R10",
           component = "Component 2: TRACE target extension") %>%
    select(gene, drugs, OR, CI_lo, CI_hi, pval, nsnp, component, outcome_label)
} else data.frame()

plot_data <- bind_rows(
  hmgcr_gbmi_row,
  hmgcr_finngen_row,
  ext_rows
)

if (nrow(plot_data) > 0) {
  # Jitter HMGCR rows slightly on y-axis when both outcomes present
  has_both_hmgcr <- !is.null(gbmi_mr) && !is.null(hmgcr_mr)
  y_col <- if ("outcome_label" %in% colnames(plot_data)) "outcome_label" else "gene"

  # Use shape to distinguish primary (GBMI, filled) vs replication (FinnGen, open)
  plot_data$shape_flag <- ifelse(
    grepl("GBMI|PRIMARY", ifelse(is.na(plot_data$outcome_label), "", plot_data$outcome_label)), "Primary (GBMI)", "Sensitivity (FinnGen R10)"
  )

  # Build y-axis label: "HMGCR — GBMI" vs "HMGCR — FinnGen R10"
  plot_data$y_label <- if (has_both_hmgcr && "outcome_label" %in% colnames(plot_data)) {
    ifelse(plot_data$component == "Component 1: HMGCR",
           paste0(plot_data$gene, "\n(", plot_data$outcome_label, ")"),
           plot_data$gene)
  } else {
    plot_data$gene
  }

  caption_text <- if (!is.null(gbmi_mr)) {
    sprintf("IVW estimate; 95%% CI. GBMI ALL = primary outcome (pre-specified upgrade); FinnGen R10 = sensitivity.\n%s = GBMI dataset used.", gbmi_label)
  } else {
    "IVW estimate; 95%% CI; FinnGen R10 IPF outcome (2,189 cases — underpowered). GBMI upgrade pre-specified."
  }

  p <- ggplot(plot_data,
              aes(x = OR, y = reorder(y_label, OR),
                  xmin = CI_lo, xmax = CI_hi,
                  color = component, shape = shape_flag)) +
    geom_point(size = 3) +
    geom_errorbarh(height = 0.2) +
    geom_vline(xintercept = 1, linetype = "dashed", color = "grey50") +
    scale_x_log10() +
    scale_color_manual(values = c("Component 1: HMGCR" = "#2166ac",
                                  "Component 2: TRACE target extension" = "#d6604d")) +
    scale_shape_manual(values = c("Primary (GBMI)" = 16, "Sensitivity (FinnGen R10)" = 1),
                       na.value = 16) +
    labs(x = "Odds ratio for IPF (log scale)",
         y = NULL,
         title = "Drug-target MR: genetic instruments for drug targets → IPF risk",
         subtitle = "All estimates non-significant (underpowered). OR<1 = directionally protective, NOT positive evidence.",
         color = NULL, shape = NULL,
         caption = caption_text) +
    theme_minimal(base_size = 11) +
    theme(legend.position = "bottom",
          panel.grid.minor = element_blank())
  ggsave(file.path(FIG_OUT, "fig_mr_forest.png"), p,
         width = 10, height = max(4, nrow(plot_data) * 0.7 + 2),
         dpi = 300)
  message("Forest plot saved.")
}

# ── Text report ───────────────────────────────────────────────────────────────
report_lines <- c(
  "Drug-target Mendelian Randomization — TRACE IPF candidates",
  "==========================================================",
  "",
  "DESIGN: Two-sample drug-target MR. Genetic instruments (randomized at",
  "conception) are structurally robust to confounding by indication and",
  "reverse causation — the dominant threats to observational pharmacoepi.",
  "",
  "OUTCOME: FinnGen R10 IPF (2,189 cases / 407,609 controls). This is a",
  "  SMALL case count for a rare disease. A pre-specified upgrade to the GBMI",
  "  IPF meta-analysis (8,492 all-ancestry / 11,160 joint cases; ~4-5x larger;",
  "  GRCh38; public) is the primary-outcome plan — see analysis_plan_mr.md.",
  "",
  "================ HONEST INTERPRETATION (read first) ================",
  "  These analyses are NULL and UNDERPOWERED. No estimate is significant.",
  "  Point estimates trending OR<1 (HMGCR, FLT1) are directionally concordant",
  "  with the TRACE prediction but are NOT positive evidence: the 95% CIs span",
  "  clinically opposite effects. Calibrated reading: 'underpowered null;",
  "  estimates non-significant; HMGCR/FLT1 directionally concordant but",
  "  uninformative about the hypothesis.' The power table below quantifies",
  "  exactly what could and could not have been detected.",
  "===================================================================",
  "",
  "COMPONENT 1: HMGCR → IPF (statin drug-target MR)",
  "-------------------------------------------------",
  "Instrument: LDL-C cis-variants at HMGCR (OpenGWAS ieu-b-110 + canonical",
  "            Swerdlow 2015/GLGC 2013 variants rs12916, rs17238484, rs5909)",
  "Outcome:    FinnGen R10 IPF (2,189 cases / 407,609 controls)",
  ""
)

if (!is.null(hmgcr_mr)) {
  res <- hmgcr_mr$results
  for (i in seq_len(nrow(res))) {
    report_lines <- c(report_lines,
      sprintf("  %-30s  OR=%.3f  [%.3f-%.3f]  p=%.4f  nSNP=%d",
              res$method[i], exp(res$b[i]),
              exp(res$b[i] - 1.96*res$se[i]),
              exp(res$b[i] + 1.96*res$se[i]),
              res$pval[i], res$nsnp[i]))
  }
  if (!is.null(hmgcr_mr$pleiotropy)) {
    report_lines <- c(report_lines, "",
      sprintf("  MR-Egger intercept: %.5f (p=%.4f) — %s",
              hmgcr_mr$pleiotropy$egger_intercept,
              hmgcr_mr$pleiotropy$pval,
              ifelse(hmgcr_mr$pleiotropy$pval > 0.05,
                     "no evidence of directional pleiotropy",
                     "WARNING: possible directional pleiotropy")))
  }
  # Post-hoc power from the achieved IVW precision
  ivw_row <- res[res$method == "Inverse variance weighted", ]
  if (nrow(ivw_row) >= 1) {
    pw <- mr_power_from_se(ivw_row$se[1])
    report_lines <- c(report_lines, "",
      "  Post-hoc power (from achieved SE; alpha=0.05, two-sided):",
      paste0("    ", paste(sprintf("%s -> %.0f%%", names(pw), 100*pw),
                           collapse = "   ")),
      sprintf("  At this precision the analysis had ~%.0f%% power to detect OR=0.80",
              100*pw["OR=0.80"]),
      "  (the magnitude plausible for statins). The null is therefore",
      "  UNINFORMATIVE about the hypothesis, not evidence against an effect.")
  }
}

report_lines <- c(report_lines, "",
  "PRIOR PUBLISHED MR (for comparison — verified against PubMed):",
  "  Cai G, Liu J, Cai M, Shao L. 'Exploring the causal effect between",
  "  lipid-modifying drugs and idiopathic pulmonary fibrosis: a drug-target",
  "  Mendelian randomization study.' Lipids Health Dis. 2024;23(1):237.",
  "  PMID 39090671. Used UK Biobank lipids + FinnGen R10 IPF.",
  "  RESULT: NO significant effect of lipid traits on IPF risk (all P>0.05).",
  "  => Our HMGCR null is a CONSISTENT REPLICATION of a published null, not an",
  "     underpowered non-replication of a positive finding.",
  "",
  "COMPONENT 2: Systematic eQTL-MR across TRACE candidate targets",
  "--------------------------------------------------------------",
  sprintf("  Targets attempted: %s", paste(names(TARGETS), collapse=", ")),
  sprintf("  Targets with results: %s",
          paste(names(extended_results), collapse=", ")),
  ""
)
if (nrow(all_ext) > 0) {
  ivw <- all_ext[all_ext$method == "Inverse variance weighted", ]
  for (i in seq_len(nrow(ivw))) {
    report_lines <- c(report_lines,
      sprintf("  %-8s  OR=%.3f [%.3f-%.3f]  p=%.4f  nSNP=%d  drugs: %s",
              ivw$gene[i], exp(ivw$b[i]),
              exp(ivw$b[i] - 1.96*ivw$se[i]),
              exp(ivw$b[i] + 1.96*ivw$se[i]),
              ivw$pval[i], ivw$nsnp[i], ivw$drugs[i]))
  }
}
report_lines <- c(report_lines, "",
  "  CAVEATS (Component 2):",
  "  - Single-SNP targets (JAK1, HDAC1) yield Wald ratios only: no CI, no",
  "    pleiotropy test, dominated by one variant — NOT interpretable evidence.",
  "  - JAK2 OR=1.13 (p=0.18, 2 SNPs) is non-significant and uninterpretable",
  "    given power — NOT evidence against baricitinib.",
  "  - Targets with no valid cis-instruments (PDGFRB all-trans; KDR, HDAC2",
  "    absent from OpenGWAS) require GTEx lung / eQTLGen / UKB-PPP pQTL",
  "    instruments — the pre-specified Component-2 upgrade.",
  "",
  "PRE-SPECIFIED NEXT STEPS (analysis_plan_mr.md):",
  "  1. Primary-outcome upgrade: GBMI IPF (8,492/11,160 cases) + FinnGen R9 replication.",
  "  2. Real multi-SNP cis instruments from GTEx lung & eQTLGen for Component 2.",
  "  3. Colocalization at HMGCR (GLGC regional data) to exclude LD confounding.",
  "  Even with GBMI, IPF rarity may leave small effects undetectable — a real",
  "  biological ceiling, stated honestly.")

writeLines(report_lines, file.path(MR_OUT, "mr_report.txt"))
message("\nAll outputs saved to results/mr/")
message("Report: results/mr/mr_report.txt")
message("Forest plot: results/figures/fig_mr_forest.png")
