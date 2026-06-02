"""
Download selected IPF GEO datasets — RESEARCH.md §8 (data acquisition).

Downloads series matrix files (normalized expression + sample metadata) for
the 5 selected datasets. For RNA-seq series, also fetches supplementary
count matrices so DESeq2 can be run on raw counts.

Selected datasets:
  GSE213001 — RNA-seq, bulk lung tissue, IPF vs. healthy donors
  GSE150910 — RNA-seq, lung tissue, IPF + CHP + controls
  GSE38958  — microarray, IPF lung tissue
  GSE134692 — RNA-seq, transplant-stage IPF lung
  GSE53845  — microarray, 40 IPF + 8 healthy controls

Usage:
    python src/02_download_datasets.py
"""

import os
import re
import sys
import time
from pathlib import Path

import requests

SELECTED = {
    "GSE213001": "rnaseq",
    "GSE150910": "rnaseq",
    "GSE38958":  "array",
    "GSE134692": "rnaseq",
    "GSE53845":  "array",
}

GEO_HTTPS = "https://ftp.ncbi.nlm.nih.gov/geo/series"
DATA_DIR = Path("data/raw")


def geo_prefix(acc: str) -> str:
    """GSE213001 → GSE213nnn (GEO FTP directory convention)."""
    return acc[:-3] + "nnn"


def list_ftp_dir(url: str) -> list[str]:
    """Return filenames listed in an NCBI HTTPS FTP directory page."""
    r = requests.get(url, timeout=30)
    if r.status_code == 404:
        return []
    r.raise_for_status()
    # NCBI FTP-over-HTTPS returns an Apache directory listing
    return re.findall(r'href="([^"/?][^"]*)"', r.text)


def download_file(url: str, dest: Path, label: str = "") -> bool:
    """Stream-download url to dest; skip if already exists. Returns True on success."""
    if dest.exists():
        print(f"  [skip] {dest.name} already downloaded")
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  → {label or dest.name} ...", end=" ", flush=True)
    try:
        r = requests.get(url, stream=True, timeout=120)
        if r.status_code == 404:
            print("404 not found")
            return False
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        written = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                written += len(chunk)
        mb = written / 1e6
        print(f"done ({mb:.1f} MB)")
        return True
    except Exception as exc:
        print(f"FAILED: {exc}")
        if dest.exists():
            dest.unlink()
        return False


def download_series_matrix(acc: str, out_dir: Path) -> list[Path]:
    """Download all series matrix files for an accession."""
    prefix = geo_prefix(acc)
    matrix_url = f"{GEO_HTTPS}/{prefix}/{acc}/matrix/"
    files = list_ftp_dir(matrix_url)
    matrix_files = [f for f in files if "series_matrix" in f and f.endswith(".gz")]

    if not matrix_files:
        # Fallback: try the standard single-file name
        matrix_files = [f"{acc}_series_matrix.txt.gz"]

    downloaded = []
    for fname in matrix_files:
        dest = out_dir / fname
        url = f"{matrix_url}{fname}"
        if download_file(url, dest, fname):
            downloaded.append(dest)
    return downloaded


def download_supplementary(acc: str, out_dir: Path) -> list[Path]:
    """Download supplementary files (count matrices etc.) for RNA-seq series."""
    prefix = geo_prefix(acc)
    suppl_url = f"{GEO_HTTPS}/{prefix}/{acc}/suppl/"
    files = list_ftp_dir(suppl_url)

    # Prioritise count/expression matrix files; skip raw FASTQ or huge archives
    wanted_exts = (".txt.gz", ".csv.gz", ".tsv.gz", ".xlsx", ".txt", ".csv", ".tsv")
    skip_keywords = ("RAW.tar", "fastq", "bam", "bigwig", "bw", "raw.tar")

    targets = [
        f for f in files
        if any(f.lower().endswith(e) for e in wanted_exts)
        and not any(kw in f.lower() for kw in skip_keywords)
    ]

    if not targets:
        print(f"  [suppl] no count/expression files found at {suppl_url}")
        return []

    downloaded = []
    suppl_dir = out_dir / "suppl"
    for fname in targets:
        dest = suppl_dir / fname
        url = f"{suppl_url}{fname}"
        if download_file(url, dest, f"suppl/{fname}"):
            downloaded.append(dest)
        time.sleep(0.1)
    return downloaded


def main() -> None:
    print(f"Downloading {len(SELECTED)} IPF datasets to {DATA_DIR}/\n")

    summary: dict[str, dict] = {}

    for acc, dtype in SELECTED.items():
        print(f"{'='*60}")
        print(f"{acc}  [{dtype}]")
        out_dir = DATA_DIR / acc
        out_dir.mkdir(parents=True, exist_ok=True)

        matrix_files = download_series_matrix(acc, out_dir)

        suppl_files: list[Path] = []
        if dtype == "rnaseq":
            suppl_files = download_supplementary(acc, out_dir)

        summary[acc] = {
            "matrix_files": len(matrix_files),
            "suppl_files": len(suppl_files),
        }
        print()

    print("=" * 60)
    print("Download summary:")
    for acc, info in summary.items():
        print(f"  {acc}: {info['matrix_files']} matrix, {info['suppl_files']} suppl files")

    print("\nNext: parse series matrix files to extract expression matrices")
    print("and sample metadata, then run QC (PCA, outlier checks).")


if __name__ == "__main__":
    main()
