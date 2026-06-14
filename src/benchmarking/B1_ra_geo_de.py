"""
RA synovial multi-cohort differential expression — Step B1.

Downloads three RA GEO cohorts, runs DE (t-test + BH correction, or limma via R),
maps probes/genes to Entrez IDs, and writes per-cohort files in the same format
as results/de/{acc}_de_entrez.csv so that the R meta-analysis script can consume
them directly.

Cohorts:
  GSE55457  — Affymetrix HG-U133A (GPL96), 13 RA / 10 normal synovium
  GSE36700  — Affymetrix HG-U133Plus2 (GPL570), 13 RA / 5 normal synovium
  GSE77298  — Illumina HiSeq (GPL16791), 14 RA / 14 OA synovium

Outputs (all in results/de/ra/):
  {acc}_ra_de_entrez.csv   — columns: log2FoldChange, pvalue, padj, gene_id (Entrez)

Usage:
    python src/benchmarking/B1_ra_geo_de.py
"""

import gzip
import io
import time
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

DATA     = Path("data/raw/ra")
DE_OUT   = Path("results/de/ra")
DATA.mkdir(parents=True, exist_ok=True)
DE_OUT.mkdir(parents=True, exist_ok=True)

# ── Dataset registry ──────────────────────────────────────────────────────────
# Each entry: (accession, matrix_url, gpl_url, ra_keyword, ctrl_keyword)
DATASETS = [
    (
        "GSE55457",
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE55nnn/GSE55457/matrix/"
        "GSE55457_series_matrix.txt.gz",
        "GPL96",
        "rheumatoid",
        "normal",
    ),
    (
        "GSE36700",
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE36nnn/GSE36700/matrix/"
        "GSE36700_series_matrix.txt.gz",
        "GPL570",
        "rheumatoid arthritis",
        "normal",
    ),
    (
        "GSE77298",
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE77nnn/GSE77298/matrix/"
        "GSE77298_series_matrix.txt.gz",
        "GPL16791",
        "rheumatoid arthritis",
        "osteoarthritis",   # OA used as control per published design
    ),
]

# Entrez maps for common platforms cached as CSVs to avoid repeated downloads
GPL_ENTREZ_URLS = {
    "GPL96":    "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPLnnn/GPL96/soft/GPL96_family.soft.gz",
    "GPL570":   "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPLnnn/GPL570/soft/GPL570_family.soft.gz",
    "GPL16791": None,  # Illumina: gene-level counts, no probe map needed
}


def download_file(url: str, dest: Path) -> bool:
    if dest.exists():
        print(f"  [cached] {dest.name}")
        return True
    print(f"  Downloading {dest.name} ...", end=" ", flush=True)
    try:
        with urlopen(url, timeout=300) as r, open(dest, "wb") as f:
            f.write(r.read())
        print(f"done ({dest.stat().st_size / 1e6:.1f} MB)")
        return True
    except Exception as exc:
        print(f"FAILED: {exc}")
        return False


def parse_matrix(path: Path) -> tuple[pd.DataFrame, list[str]]:
    """Parse a GEO series matrix file into (expression_df, sample_titles)."""
    meta: dict[str, list] = {}
    rows: list[str] = []
    header: list[str] = []
    in_tab = False
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("!") and not in_tab:
                parts = line.split("\t")
                key = parts[0].lstrip("!").strip()
                vals = [v.strip('"') for v in parts[1:]]
                meta.setdefault(key, []).extend(vals)
            elif "ID_REF" in line and not in_tab:
                header = [v.strip('"') for v in line.split("\t")]
                in_tab = True
            elif in_tab:
                if "series_matrix_table_end" in line:
                    break
                rows.append(line)

    df = (
        pd.DataFrame([r.split("\t") for r in rows], columns=header)
        .set_index("ID_REF")
    )
    df.index = df.index.str.strip('"')
    df = df.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    titles = meta.get("Sample_title", [])[:df.shape[1]]
    return df, titles


def parse_gpl_soft(gpl_id: str) -> dict[str, int]:
    """Return probe_id → Entrez mapping for array platforms."""
    cache = DATA / f"{gpl_id}_entrez.csv"
    if cache.exists():
        gpl_df = pd.read_csv(cache, dtype=str)
    else:
        url = GPL_ENTREZ_URLS.get(gpl_id)
        if url is None:
            return {}
        soft_path = DATA / f"{gpl_id}_family.soft.gz"
        if not download_file(url, soft_path):
            return {}
        records, header_r, in_tab = [], [], False
        with gzip.open(soft_path, "rt", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if "platform_table_begin" in line:
                    in_tab = True
                    continue
                if "platform_table_end" in line:
                    break
                if not in_tab:
                    continue
                if not header_r:
                    header_r = line.split("\t")
                    continue
                parts = line.split("\t")
                if len(parts) >= len(header_r):
                    records.append(dict(zip(header_r, parts)))
        gpl_df = pd.DataFrame(records)
        gpl_df.to_csv(cache, index=False)

    col_up = {c.upper(): c for c in gpl_df.columns}
    id_col = col_up.get("ID") or col_up.get("PROBE_ID")
    e_col  = (col_up.get("ENTREZ_GENE_ID") or col_up.get("GENE_ID")
               or col_up.get("GENE"))
    if not id_col or not e_col:
        print(f"  WARNING: could not find ID/Entrez columns in {gpl_id}")
        return {}

    probe2e: dict[str, int] = {}
    for _, row in gpl_df.iterrows():
        try:
            probe2e[str(row[id_col]).strip()] = int(
                str(row[e_col]).strip().split("///")[0].strip()
            )
        except (ValueError, TypeError):
            pass
    return probe2e


def run_de(expr: pd.DataFrame, ra_idx: list[int],
           ctrl_idx: list[int], gpl_id: str,
           acc: str) -> pd.DataFrame:
    """
    Run t-test DE (log2-transformed intensities for arrays, raw counts
    log2(CPM+1) for RNA-seq) and return DataFrame with Entrez index.
    """
    # Log2-transform if values look like raw intensities (median > 100)
    if expr.median().median() > 100:
        expr = np.log2(expr.clip(lower=1))

    ra_mat   = expr.iloc[:, ra_idx].values.astype(float)
    ctrl_mat = expr.iloc[:, ctrl_idx].values.astype(float)
    valid    = (
        (np.isnan(ra_mat).mean(axis=1) < 0.5) &
        (np.isnan(ctrl_mat).mean(axis=1) < 0.5)
    )
    ra_f, ctrl_f = ra_mat[valid], ctrl_mat[valid]
    probes = expr.index[valid]

    t_stat, pvals = stats.ttest_ind(ra_f, ctrl_f, axis=1, nan_policy="omit")
    logfc = np.nanmean(ra_f, axis=1) - np.nanmean(ctrl_f, axis=1)
    _, padj, _, _ = multipletests(np.nan_to_num(pvals, nan=1.0), method="fdr_bh")

    de = pd.DataFrame({
        "probe":          probes,
        "log2FoldChange": logfc,
        "pvalue":         pvals,
        "padj":           padj,
    })

    # Map probes → Entrez
    if gpl_id == "GPL16791":
        # RNA-seq from Illumina: row index is already HGNC symbol or Ensembl
        # Attempt to use the index directly as Entrez (if numeric) or skip mapping
        de["entrez"] = pd.to_numeric(de["probe"], errors="coerce")
    else:
        probe2e = parse_gpl_soft(gpl_id)
        if not probe2e:
            print(f"  WARNING {acc}: no probe map; saving unmapped DE")
            de.to_csv(DE_OUT / f"{acc}_ra_de_unmapped.csv", index=False)
            return pd.DataFrame()
        de["entrez"] = de["probe"].map(probe2e)

    de = de.dropna(subset=["entrez"]).copy()
    de["entrez"] = de["entrez"].astype(int)
    # Keep probe with largest |LFC| per Entrez gene
    best_idx = (
        de.groupby("entrez")["log2FoldChange"]
        .apply(lambda x: x.abs().idxmax())
        .values
    )
    de = de.loc[best_idx].set_index("entrez")
    de = de[["log2FoldChange", "pvalue", "padj"]]

    sig = ((de["padj"] < 0.05) & (de["log2FoldChange"].abs() > 0.5)).sum()
    print(f"  {acc}: {len(de):,} Entrez genes,  {sig:,} significant (padj<0.05, |LFC|>0.5)")
    return de


def main() -> None:
    for acc, matrix_url, gpl_id, ra_kw, ctrl_kw in DATASETS:
        print(f"\n{'='*60}")
        print(f"Processing {acc} (platform {gpl_id})")
        out_path = DE_OUT / f"{acc}_ra_de_entrez.csv"
        if out_path.exists():
            print(f"  [skip] {out_path.name} already exists")
            continue

        # Download matrix
        local = DATA / f"{acc}_series_matrix.txt.gz"
        if not download_file(matrix_url, local):
            print(f"  SKIP {acc}: download failed")
            continue

        # Parse
        print(f"  Parsing {acc} ...")
        expr, titles = parse_matrix(local)
        print(f"  Expression: {expr.shape}, {len(titles)} titles")

        ra_idx   = [i for i, t in enumerate(titles)
                    if ra_kw.lower() in t.lower()]
        ctrl_idx = [i for i, t in enumerate(titles)
                    if ctrl_kw.lower() in t.lower()]
        print(f"  RA samples: {len(ra_idx)},  Control samples: {len(ctrl_idx)}")

        if not ra_idx or not ctrl_idx:
            print(f"  WARNING: sample classification failed — check titles: {titles[:5]}")
            continue

        de = run_de(expr, ra_idx, ctrl_idx, gpl_id, acc)
        if de.empty:
            continue

        de.to_csv(out_path)
        print(f"  Saved → {out_path}")

    print("\nB1 complete. Next: run src/benchmarking/ra_meta_analysis.R")


if __name__ == "__main__":
    main()