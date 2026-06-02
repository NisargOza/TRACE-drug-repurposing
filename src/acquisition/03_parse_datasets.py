"""
Parse downloaded GEO datasets — RESEARCH.md §8.

For each selected dataset:
  - Extracts sample metadata from the series matrix header
  - For RNA-seq: reads supplementary raw count matrices
  - For microarray: reads expression values from the series matrix data block
  - Writes to data/processed/{acc}/metadata.csv and expression.parquet

Usage:
    python src/03_parse_datasets.py
"""

import gzip
import re
from io import StringIO
from pathlib import Path

import pandas as pd

DATA_RAW = Path("data/raw")
DATA_PROC = Path("data/processed")

SELECTED = {
    "GSE213001": "rnaseq",
    "GSE150910": "rnaseq",
    "GSE38958":  "array",
    "GSE134692": "rnaseq",
    "GSE53845":  "array",
}

# Supplementary count files to use (prefer raw counts for RNA-seq)
SUPPL_COUNTS = {
    "GSE213001": "GSE213001_Entrez-IDs-Lung-IPF-GRCh38-p12-raw_counts.csv.gz",
    "GSE150910": "GSE150910_gene-level_count_file.csv.gz",
    "GSE134692": "GSE134692_raw_counts.txt.gz",
}


# ---------------------------------------------------------------------------
# Series matrix parsing
# ---------------------------------------------------------------------------

def parse_series_matrix(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse a GEO series matrix file.

    GEO series matrices can have the same key repeated multiple times
    (e.g. !Sample_characteristics_ch1 once per characteristic per sample).
    We handle this by concatenating all occurrences for the same key into
    a single ' | '-joined string per sample, ensuring one row per sample.

    Returns:
        meta  — DataFrame (samples × metadata fields), index = GSM ID
        expr  — DataFrame (genes/probes × samples), or empty DataFrame
    """
    opener = gzip.open if str(path).endswith(".gz") else open

    # occurrences[key] = list of per-line value lists (one inner list per line)
    occurrences: dict[str, list[list[str]]] = {}
    expr_lines: list[str] = []
    in_table = False

    with opener(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("!Sample_"):
                key, _, rest = line.partition("\t")
                key = key.lstrip("!")
                values = [v.strip('"') for v in rest.split("\t")]
                occurrences.setdefault(key, []).append(values)
            elif line == "!series_matrix_table_begin":
                in_table = True
            elif line == "!series_matrix_table_end":
                in_table = False
            elif in_table:
                expr_lines.append(line)
            elif line.startswith('"ID_REF"') or line.startswith("ID_REF"):
                in_table = True
                expr_lines.append(line)

    if not occurrences:
        return pd.DataFrame(), pd.DataFrame()

    # Determine n_samples from geo_accession (appears exactly once)
    n_samples = len(occurrences.get("Sample_geo_accession", [[]])[0])
    if n_samples == 0:
        n_samples = max(len(lines[0]) for lines in occurrences.values())

    meta_dict: dict[str, list[str]] = {}
    for key, lines in occurrences.items():
        if len(lines) == 1:
            vals = lines[0]
        else:
            # Join multiple occurrences per sample with ' | '
            vals = [
                " | ".join(
                    ln[i].strip() if i < len(ln) else ""
                    for ln in lines
                )
                for i in range(n_samples)
            ]
        meta_dict[key] = (vals + [""] * n_samples)[:n_samples]

    meta = pd.DataFrame(meta_dict)
    if "Sample_geo_accession" in meta.columns:
        meta = meta.set_index("Sample_geo_accession")
        meta = meta[~meta.index.duplicated(keep="first")]

    # Build expression DataFrame (microarray or fallback)
    if expr_lines:
        try:
            expr = pd.read_csv(
                StringIO("\n".join(expr_lines)),
                sep="\t",
                index_col=0,
                low_memory=False,
            )
            expr.index.name = "probe_id"
        except Exception:
            expr = pd.DataFrame()
    else:
        expr = pd.DataFrame()

    return meta, expr


# ---------------------------------------------------------------------------
# Supplementary count file readers
# ---------------------------------------------------------------------------

def read_count_file(path: Path) -> pd.DataFrame:
    """Read a supplementary count matrix; return genes × samples DataFrame."""
    sep = "\t" if path.name.endswith((".txt.gz", ".txt")) else ","
    df = pd.read_csv(path, sep=sep, index_col=0, low_memory=False)
    # Drop non-numeric columns (gene symbol annotations etc.)
    numeric_cols = df.columns[df.dtypes.apply(lambda t: pd.api.types.is_numeric_dtype(t))]
    df = df[numeric_cols]
    df.index.name = "gene_id"
    return df


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def extract_condition(meta: pd.DataFrame) -> pd.Series:
    """
    Best-effort extraction of IPF/control label from GEO sample metadata.
    Returns a Series with values like 'IPF', 'control', 'CHP', 'other'.
    """
    # Candidate columns that describe disease/group
    candidate_cols = [
        c for c in meta.columns
        if any(kw in c.lower() for kw in
               ["characteristics", "source", "title", "description", "disease",
                "condition", "status", "group", "diagnosis", "subject_type"])
    ]
    if not candidate_cols:
        candidate_cols = list(meta.columns)
    combined = meta[candidate_cols].astype(str).apply(
        lambda row: " ".join(row.values).lower(), axis=1
    )

    def label(s: str) -> str:
        if any(kw in s for kw in ["ipf", "idiopathic pulmonary fibrosis", "usual interstitial"]):
            return "IPF"
        # NDC = non-diseased control (used in GSE213001)
        if any(kw in s for kw in ["control", "normal", "donor", "healthy", "unaffected", "ndc"]):
            return "control"
        if "chronic hypersensitivity" in s or " chp" in s:
            return "CHP"
        return "other"

    return combined.apply(label).rename("condition")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_dataset(acc: str, dtype: str) -> None:
    raw_dir = DATA_RAW / acc
    proc_dir = DATA_PROC / acc
    proc_dir.mkdir(parents=True, exist_ok=True)

    matrix_files = sorted(raw_dir.glob("*series_matrix*.gz"))
    if not matrix_files:
        print(f"  [WARN] no series matrix found for {acc}")
        return

    print(f"  Parsing series matrix...")
    meta, expr_matrix = parse_series_matrix(matrix_files[0])

    if not meta.empty:
        meta["condition"] = extract_condition(meta)
        meta.to_csv(proc_dir / "metadata.csv")
        counts = meta["condition"].value_counts().to_dict()
        print(f"  Metadata: {len(meta)} samples  |  condition counts: {counts}")
    else:
        print(f"  [WARN] could not parse metadata")

    if dtype == "rnaseq" and acc in SUPPL_COUNTS:
        count_path = raw_dir / "suppl" / SUPPL_COUNTS[acc]
        if count_path.exists():
            print(f"  Reading count matrix: {count_path.name}")
            counts_df = read_count_file(count_path)
            print(f"  Count matrix: {counts_df.shape[0]} genes × {counts_df.shape[1]} samples")
            counts_df.to_csv(proc_dir / "counts_raw.csv.gz", compression="gzip")
        else:
            print(f"  [WARN] count file not found: {count_path}")
    elif dtype == "array" and not expr_matrix.empty:
        print(f"  Expression matrix: {expr_matrix.shape[0]} probes × {expr_matrix.shape[1]} samples")
        expr_matrix.to_csv(proc_dir / "expression_array.csv.gz", compression="gzip")
    elif dtype == "rnaseq" and not expr_matrix.empty:
        # Series matrix has expression values (no separate count file)
        print(f"  Using series matrix expression: {expr_matrix.shape}")
        expr_matrix.to_csv(proc_dir / "expression_rnaseq.csv.gz", compression="gzip")


def main() -> None:
    print(f"Parsing {len(SELECTED)} datasets → {DATA_PROC}/\n")
    for acc, dtype in SELECTED.items():
        print(f"{'─'*50}")
        print(f"{acc}  [{dtype}]")
        process_dataset(acc, dtype)
        print()

    print("Done. Next: QC notebook (PCA, library-size checks, outlier detection).")


if __name__ == "__main__":
    main()
