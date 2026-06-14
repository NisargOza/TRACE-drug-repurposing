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
# Each entry: (accession, matrix_url, gpl_id, ra_keyword, ctrl_keyword)
# Keywords are matched as SUBSTRINGS (case-insensitive) anywhere in the title.
# RA keyword is applied first; control keyword is applied to remaining samples
# so there is no risk of overlap.
#   GSE55457: titles are "Rheumatoid Arthritis synovium N" / "Normal synovium N"
#   GSE36700: titles are "Synovial tissue from patient with Rheumatoid Arthritis N" / "OA N" / "Normal N"
#   GSE77298: titles are "Rheumatoid Arthritis-N" / "Healthy Control-N"
# Each entry: (accession, matrix_url, gpl_id, ra_keyword, ctrl_keyword, use_startswith)
# use_startswith=True  -> keyword must be at the START of the title (short-code datasets)
# use_startswith=False -> keyword matched anywhere as substring (descriptive-title datasets)
#
# Confirmed title formats (from diagnostic run):
#   GSE55457: "Rheumatoid Arthritis synovium N" / "Normal synovium N"  -> substring
#   GSE36700: "RA1".."RA7" / "OA1".."OA5"                             -> startswith
#   GSE77298: "RA-1".."RA-16" / "HC-1".."HC-7"                        -> startswith
DATASETS = [
    (
        "GSE55457",
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE55nnn/GSE55457/matrix/"
        "GSE55457_series_matrix.txt.gz",
        "GPL96",
        "rheumatoid",   # substring match — titles are descriptive
        "normal",
        False,
    ),
    (
        "GSE36700",
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE36nnn/GSE36700/matrix/"
        "GSE36700_series_matrix.txt.gz",
        "GPL570",
        "RA",           # startswith — confirmed titles: RA1..RA7
        "OA",           # startswith — confirmed titles: OA1..OA5
        True,
    ),
    (
        "GSE77298",
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE77nnn/GSE77298/matrix/"
        "GSE77298_series_matrix.txt.gz",
        "GPL570",       # Affymetrix HG-U133Plus2 — same platform as GSE36700
        "RA-",          # startswith — confirmed titles: RA-1..RA-16
        "HC-",          # startswith — confirmed titles: HC-1..HC-7
        True,
    ),
]

# Entrez maps for common platforms.
# Using .annot.gz files (clean probe->Entrez TSVs, ~5 MB each) instead of
# _family.soft.gz files (multi-section family files, ~200 MB, require special parsing).
GPL_ENTREZ_URLS = {
    "GPL96":  "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPLnnn/GPL96/annot/GPL96.annot.gz",
    "GPL570": "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPLnnn/GPL570/annot/GPL570.annot.gz",
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


def parse_gpl_annot(gpl_id: str) -> dict[str, int]:
    """
    Return probe_id -> Entrez mapping by parsing a GEO .annot.gz file.

    GEO .annot.gz files use SOFT format: metadata lines prefixed with ^ or !,
    then the probe table delimited by:
        !platform_table_begin  ...header row...  data rows  !platform_table_end
    This is the same marker-based structure as _family.soft.gz, but .annot.gz
    files contain only a SINGLE platform section so the parser always finds
    exactly one table.

    The parsed table is cached as a CSV so subsequent runs skip re-parsing.
    If the cache exists but is empty (from a previously failed parse), it is
    deleted and re-parsed automatically.
    """
    cache = DATA / f"{gpl_id}_entrez.csv"

    # Guard: delete empty/corrupt cache files from previous failed runs
    if cache.exists() and cache.stat().st_size < 100:
        print(f"  Removing empty cache {cache.name} and re-parsing ...")
        cache.unlink()

    if cache.exists():
        gpl_df = pd.read_csv(cache, dtype=str)
        print(f"  [cached] {cache.name} ({len(gpl_df):,} probes)")
    else:
        url = GPL_ENTREZ_URLS.get(gpl_id)
        if url is None:
            return {}
        annot_path = DATA / f"{gpl_id}.annot.gz"
        if not download_file(url, annot_path):
            return {}

        # Parse SOFT-format table between !platform_table_begin/end markers
        rows, header_r, in_tab = [], [], False
        with gzip.open(annot_path, "rt", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if "platform_table_begin" in line.lower():
                    in_tab = True
                    continue
                if "platform_table_end" in line.lower():
                    break
                if not in_tab:
                    continue
                parts = line.split("\t")
                if not header_r:
                    header_r = parts
                    continue
                if len(parts) >= len(header_r):
                    rows.append(dict(zip(header_r, parts)))

        if not rows:
            print(f"  ERROR: {gpl_id}.annot.gz parsed 0 rows — table markers not found")
            print(f"  Delete {annot_path.name} and re-run to re-download")
            return {}

        gpl_df = pd.DataFrame(rows)
        gpl_df.to_csv(cache, index=False)
        print(f"  Parsed {len(gpl_df):,} probes from {gpl_id}.annot.gz -> cached")

    # Normalise column lookup (case-insensitive, strip whitespace)
    col_up = {c.upper().strip(): c for c in gpl_df.columns}

    # ID column — GPL96/GPL570 annot files use "ID" (same as SOFT)
    id_col = (col_up.get("ID") or col_up.get("PROBE SET ID")
              or col_up.get("PROBE_SET_ID") or col_up.get("PROBE ID"))
    # Entrez column — GPL96/GPL570 annot files use "ENTREZ_GENE_ID"
    e_col  = (col_up.get("ENTREZ_GENE_ID") or col_up.get("ENTREZ GENE")
              or col_up.get("ENTREZ_GENE") or col_up.get("GENE_ID")
              or col_up.get("GENE ID") or col_up.get("GENE"))

    if not id_col or not e_col:
        print(f"  WARNING: could not find Probe/Entrez columns in {gpl_id}")
        print(f"  Available columns: {list(gpl_df.columns)[:10]}")
        return {}

    probe2e: dict[str, int] = {}
    for _, row in gpl_df.iterrows():
        try:
            probe2e[str(row[id_col]).strip()] = int(
                str(row[e_col]).strip().split("///")[0].strip()
            )
        except (ValueError, TypeError):
            pass
    print(f"  {gpl_id}: {len(probe2e):,} probe->Entrez mappings loaded")
    return probe2e


def load_ensembl_to_entrez() -> dict[str, int]:
    """
    Build Ensembl gene ID -> Entrez ID map for human genes.

    Downloads NCBI gene2ensembl.gz (all species, ~150 MB compressed),
    filters to Homo sapiens (tax_id=9606), and caches the human-only
    mapping as data/raw/ra/ensembl_to_entrez_human.csv (~30k rows, ~1 MB).

    gene2ensembl columns (tab-separated, no header line — first line IS data):
      0: tax_id
      1: GeneID  (Entrez)
      2: Ensembl_gene_identifier  (ENSG...)
      3: RNA_nucleotide_accession
      4: Ensembl_rna_identifier
      5: protein_accession
      6: Ensembl_protein_identifier
    """
    cache = DATA / "ensembl_to_entrez_human.csv"

    if cache.exists() and cache.stat().st_size > 1000:
        df = pd.read_csv(cache, dtype=str)
        mapping = dict(zip(df["ensembl_id"], df["entrez_id"].astype(int)))
        print(f"  [cached] ensembl_to_entrez_human.csv ({len(mapping):,} genes)")
        return mapping

    url = "https://ftp.ncbi.nlm.nih.gov/gene/DATA/gene2ensembl.gz"
    raw = DATA / "gene2ensembl.gz"
    if not download_file(url, raw):
        print("  ERROR: could not download gene2ensembl.gz")
        return {}

    print("  Filtering gene2ensembl.gz to human (tax_id=9606) ...")
    rows = []
    with gzip.open(raw, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            if parts[0] != "9606":          # human only
                continue
            ensembl = parts[2].strip()
            entrez  = parts[1].strip()
            if ensembl.startswith("ENSG") and entrez.isdigit():
                rows.append({"ensembl_id": ensembl, "entrez_id": entrez})

    df = pd.DataFrame(rows).drop_duplicates(subset="ensembl_id")
    df.to_csv(cache, index=False)
    mapping = dict(zip(df["ensembl_id"], df["entrez_id"].astype(int)))
    print(f"  Parsed {len(mapping):,} human Ensembl->Entrez mappings -> cached")
    return mapping


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
    probe2e = parse_gpl_annot(gpl_id)
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
    for acc, matrix_url, gpl_id, ra_kw, ctrl_kw, use_startswith in DATASETS:
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

        # Classify RA samples first, then controls from remaining samples.
        # Short-code datasets (use_startswith=True): match at title start only,
        # so "RA" doesn't accidentally match inside longer words.
        # Descriptive-title datasets (use_startswith=False): substring match.
        def matches(title: str, keyword: str) -> bool:
            if use_startswith:
                return title.lower().startswith(keyword.lower())
            return keyword.lower() in title.lower()

        ra_idx   = [i for i, t in enumerate(titles) if matches(t, ra_kw)]
        non_ra   = [i for i in range(len(titles)) if i not in set(ra_idx)]
        ctrl_idx = [i for i in non_ra if matches(titles[i], ctrl_kw)]

        print(f"  RA samples: {len(ra_idx)},  Control samples: {len(ctrl_idx)}")

        if not ra_idx or not ctrl_idx:
            print(f"  WARNING: sample classification failed")
            print(f"  ALL {len(titles)} titles:")
            for _i, _t in enumerate(titles):
                print(f"    [{_i}] {_t!r}")
            continue

        de = run_de(expr, ra_idx, ctrl_idx, gpl_id, acc)
        if de.empty:
            continue

        de.to_csv(out_path)
        print(f"  Saved → {out_path}")

    print("\nB1 complete. Next: run src/benchmarking/ra_meta_analysis.R")


if __name__ == "__main__":
    main()