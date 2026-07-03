
import gzip
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

DATA_PROC = Path("data/processed")
DATA_RAW  = Path("data/raw")
DE_DIR    = Path("results/de")
DE_DIR.mkdir(parents=True, exist_ok=True)

RNASEQ_DATASETS = ["GSE213001", "GSE150910"]
ARRAY_DATASETS  = [("GSE38958", "GPL5175"), ("GSE53845", "GPL6480")]

QC_OUTLIERS = {"chp_23", "ipf_779", "chp_142"}

GEO_FTP = "https://ftp.ncbi.nlm.nih.gov/geo/platforms"


def gpl_prefix(gpl: str) -> str:
    return gpl[:-3] + "nnn"


def download_gpl_soft(gpl: str) -> Path:
    dest = DATA_RAW / f"{gpl}_family.soft.gz"
    if dest.exists():
        print(f"  [skip] {dest.name} already downloaded")
        return dest
    prefix = gpl_prefix(gpl)
    url = f"{GEO_FTP}/{prefix}/{gpl}/soft/{gpl}_family.soft.gz"
    print(f"  Downloading {gpl} annotation...", end=" ", flush=True)
    r = requests.get(url, stream=True, timeout=120)
    if r.status_code == 404:
        url = f"{GEO_FTP}/{prefix}/{gpl}/annot/{gpl}.annot.gz"
        r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in r.iter_content(1 << 20):
            f.write(chunk)
    print(f"done ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def parse_gpl_probe_map(gpl_path: Path) -> pd.Series:
    print(f"  Parsing {gpl_path.name} for probe→Entrez mapping...")
    id_col = gene_id_col = None
    rows = []
    in_table = False

    opener = gzip.open if str(gpl_path).endswith(".gz") else open
    with opener(gpl_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#"):
                continue
            if line.startswith("!platform_table_begin") or line.startswith("!Platform_table_begin"):
                in_table = True
                continue
            if line.startswith("!platform_table_end") or line.startswith("!Platform_table_end"):
                break
            if not in_table:
                continue
            if id_col is None:
                cols = line.split("\t")
                id_col = 0
                for i, c in enumerate(cols):
                    if c.strip().lower() in ("gene_id", "entrez_gene_id", "entrez gene id",
                                              "gene id", "ncbi gene", "gene", "geneids"):
                        gene_id_col = i
                        break
                if gene_id_col is None:
                    for i, c in enumerate(cols):
                        if "gene" in c.lower() and "id" in c.lower():
                            gene_id_col = i
                            break
                if gene_id_col is None:
                    for i, c in enumerate(cols):
                        if "symbol" in c.lower() or "gene_symbol" in c.lower():
                            gene_id_col = i
                            break
                print(f"    Header cols: {cols[:8]}  → using col {gene_id_col} ({cols[gene_id_col] if gene_id_col else 'NOT FOUND'})")
                continue
            parts = line.split("\t")
            if gene_id_col is None or gene_id_col >= len(parts):
                continue
            probe_id = parts[id_col].strip()
            gene_val = parts[gene_id_col].strip()
            if gene_val and gene_val not in ("---", "NA", ""):
                first_id = gene_val.split("///")[0].split(",")[0].strip()
                rows.append((probe_id, first_id))

    if not rows:
        raise ValueError(f"Could not parse probe→gene mapping from {gpl_path}")

    probe_map = pd.Series(dict(rows))
    probe_map = probe_map[probe_map != ""].rename("gene_id")
    print(f"    {len(probe_map):,} probes with Entrez/gene ID")
    return probe_map


def run_deseq2(acc: str) -> None:
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats

    out = DE_DIR / f"{acc}_de_results.csv"
    if out.exists():
        print(f"  [skip] {out.name} already exists")
        return

    counts = pd.read_csv(DATA_PROC / acc / "counts_raw.csv.gz", index_col=0)
    counts = counts.select_dtypes(include=[np.number])

    meta = pd.read_csv(DATA_PROC / acc / "metadata.csv", index_col=0, low_memory=False)

    title_col = next((c for c in meta.columns if "title" in c.lower()), None)
    if title_col and not (set(counts.columns) & set(meta.index)):
        title_to_gsm = dict(zip(meta[title_col], meta.index))
        counts.columns = [title_to_gsm.get(c, c) for c in counts.columns]

    sample_meta = meta.reindex(counts.columns)[["condition"]].copy()

    keep = (
        ~sample_meta.index.isin(QC_OUTLIERS) &
        sample_meta["condition"].isin(["IPF", "control"])
    )
    sample_meta = sample_meta[keep]
    counts = counts[sample_meta.index]

    print(f"  Samples: {sample_meta['condition'].value_counts().to_dict()}")

    counts_int = counts.T.round().astype(int)
    sample_meta = sample_meta.loc[counts_int.index]

    min_samples = max(3, int(0.1 * len(counts_int)))
    keep_genes = (counts_int >= 10).sum(axis=0) >= min_samples
    counts_int = counts_int.loc[:, keep_genes]
    print(f"  Genes after low-count filter: {keep_genes.sum():,}")

    print("  Running DESeq2 (pydeseq2)...")
    dds = DeseqDataSet(
        counts=counts_int,
        metadata=sample_meta,
        design_factors="condition",
        quiet=True,
    )
    dds.deseq2()

    stats = DeseqStats(dds, contrast=("condition", "IPF", "control"), quiet=True)
    stats.summary()
    res = stats.results_df.copy()
    res.index.name = "gene_id"
    res = res.rename(columns={
        "log2FoldChange": "log2FoldChange",
        "pvalue": "pvalue",
        "padj": "padj",
    })
    res["dataset"] = acc
    res.to_csv(out)
    print(f"  Results: {len(res):,} genes → {out.name}")
    sig = res[(res["padj"] < 0.05) & (res["log2FoldChange"].abs() > 1)]
    print(f"  Significant (padj<0.05, |LFC|>1): {len(sig):,}  "
          f"(up={( sig['log2FoldChange']>0).sum()}, down={(sig['log2FoldChange']<0).sum()})")


R_SCRIPT = Path("src/de_array.R")


def write_r_script() -> None:
    R_SCRIPT.write_text(
        r"""

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
""",
        encoding="utf-8",
    )


def run_limma(acc: str, gpl: str) -> None:
    out = DE_DIR / f"{acc}_de_results.csv"
    if out.exists():
        print(f"  [skip] {out.name} already exists")
        return

    if gpl == "GPL6480":
        download_gpl_soft(gpl)

    expr_csv = DATA_PROC / acc / "expression_array.csv.gz"
    meta_csv = DATA_PROC / acc / "metadata.csv"

    print(f"  Running limma (R)...")
    result = subprocess.run(
        ["Rscript", str(R_SCRIPT), acc, gpl,
         str(expr_csv), str(meta_csv), str(out)],
        capture_output=True, text=True,
    )
    if result.stdout:
        for line in result.stdout.strip().splitlines():
            print(f"  {line}")
    if result.returncode != 0:
        print(f"  [ERROR] R exited {result.returncode}")
        print(result.stderr[-2000:])
        raise RuntimeError(f"limma failed for {acc}")


def main() -> None:
    write_r_script()

    print("=== RNA-seq DE (pydeseq2) ===\n")
    for acc in RNASEQ_DATASETS:
        print(f"{'─'*50}\n{acc}")
        run_deseq2(acc)
        print()

    print("=== Microarray DE (limma) ===\n")
    for acc, gpl in ARRAY_DATASETS:
        print(f"{'─'*50}\n{acc}  [{gpl}]")
        run_limma(acc, gpl)
        print()

    print("Done. Results in results/de/")
    print("Next: meta-analysis across datasets (RESEARCH.md §1c).")


if __name__ == "__main__":
    main()
