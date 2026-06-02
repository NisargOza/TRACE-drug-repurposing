"""
L1000 drug signature acquisition — RESEARCH.md §1a / §2a.

Steps:
  1. Download GSE70138 (Phase II) metadata files — small, identifies all
     available small-molecule perturbagens and their cell lines
  2. Confirm pirfenidone and nintedanib are present (positive controls)
  3. Download the Level 5 signature matrix (.gctx) — subsetted to landmark
     genes and small-molecule perturbagens to keep memory manageable
  4. Extract per-perturbagen consensus signatures (median across replicates)
     and save as a genes x drugs matrix for reversal scoring

GSE70138 is preferred over GSE92742 (Phase I) as it is smaller and includes
the Phase I data in reprocessed form.

Usage:
    python src/aim2_reversal/08_l1000_setup.py
"""

import gzip
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests

DATA_RAW  = Path("data/raw/l1000")
DATA_RAW.mkdir(parents=True, exist_ok=True)
L1000_DIR = Path("results/l1000")
L1000_DIR.mkdir(parents=True, exist_ok=True)

GEO_FTP = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE70nnn/GSE70138/suppl"

# Metadata files (small — download always)
META_FILES = {
    "sig_info":  "GSE70138_Broad_LINCS_sig_info_2017-03-06.txt.gz",
    "gene_info": "GSE70138_Broad_LINCS_gene_info_2017-03-06.txt.gz",
    "inst_info": "GSE70138_Broad_LINCS_inst_info_2017-03-06.txt.gz",
}

# Level 5 consensus z-score matrix (~3 GB uncompressed — downloaded on demand)
GCTX_FILE = "GSE70138_Broad_LINCS_Level5_COMPZ_n118050x12328_2017-03-06.gctx.gz"

# Positive controls per RESEARCH.md — must be present
POSITIVE_CONTROLS = ["pirfenidone", "nintedanib"]


def download(fname: str, dest: Path) -> Path:
    out = dest / fname
    if out.exists():
        print(f"  [skip] {fname}")
        return out
    url = f"{GEO_FTP}/{fname}"
    print(f"  Downloading {fname}...", end=" ", flush=True)
    r = requests.get(url, stream=True, timeout=300)
    r.raise_for_status()
    with open(out, "wb") as f:
        for chunk in r.iter_content(1 << 20):
            f.write(chunk)
    print(f"done ({out.stat().st_size / 1e6:.1f} MB)")
    return out


def load_meta(key: str) -> pd.DataFrame:
    path = DATA_RAW / META_FILES[key]
    return pd.read_csv(path, sep="\t", compression="gzip", low_memory=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # 1. Download metadata
    print("=== Step 1: Download L1000 metadata ===")
    for key, fname in META_FILES.items():
        download(fname, DATA_RAW)

    # 2. Load and inspect signature metadata
    print("\n=== Step 2: Inspect signature metadata ===")
    sig_info  = load_meta("sig_info")
    gene_info = load_meta("gene_info")
    inst_info = load_meta("inst_info")

    print(f"  Signatures total:   {len(sig_info):,}")
    print(f"  Genes measured:     {len(gene_info):,}")
    print(f"  Landmark genes:     {gene_info['pr_is_lm'].sum():,}")

    # Small-molecule perturbagens only
    sm_sigs = sig_info[sig_info["pert_type"] == "trt_cp"].copy()
    print(f"  Small-molecule sigs:{len(sm_sigs):,}")
    print(f"  Unique drugs:       {sm_sigs['pert_iname'].nunique():,}")
    print(f"  Cell lines:         {sm_sigs['cell_id'].nunique():,}")
    print(f"  Cell lines present: {sorted(sm_sigs['cell_id'].unique())[:15]} ...")

    # 3. Confirm positive controls
    print("\n=== Step 3: Positive control check ===")
    sm_lower = sm_sigs.copy()
    sm_lower["pert_lower"] = sm_lower["pert_iname"].str.lower()
    for drug in POSITIVE_CONTROLS:
        hits = sm_lower[sm_lower["pert_lower"].str.contains(drug.lower(), na=False)]
        n_sigs  = len(hits)
        n_cells = hits["cell_id"].nunique() if n_sigs else 0
        status = "FOUND" if n_sigs else "NOT FOUND"
        print(f"  {drug:15} [{status}]  {n_sigs:4} signatures  {n_cells} cell lines")
        if n_sigs:
            print(f"    sig_ids: {hits['sig_id'].head(3).tolist()}")

    # 4. Cell-line summary (for tissue-aware weighting in §2a)
    print("\n=== Step 4: Cell-line coverage ===")
    cl_counts = sm_sigs.groupby("cell_id").size().sort_values(ascending=False)
    print("  Top 15 cell lines by signature count:")
    for cl, n in cl_counts.head(15).items():
        print(f"    {cl:10} {n:6,} signatures")

    # Save filtered sig_info for use by downstream scripts
    sm_sigs.to_csv(L1000_DIR / "sm_sig_info.csv", index=False)
    gene_info[gene_info["pr_is_lm"] == 1].to_csv(L1000_DIR / "landmark_genes.csv", index=False)
    print(f"\n  Saved sm_sig_info.csv ({len(sm_sigs):,} rows)")
    print(f"  Saved landmark_genes.csv ({gene_info['pr_is_lm'].sum():,} genes)")

    # 5. Report on .gctx download
    gctx_path = DATA_RAW / GCTX_FILE
    if gctx_path.exists():
        print(f"\n  Level 5 .gctx already present ({gctx_path.stat().st_size/1e9:.1f} GB)")
    else:
        print(f"\n=== Step 5: Level 5 .gctx not yet downloaded ===")
        print(f"  File: {GCTX_FILE}")
        print(f"  Run:  python src/aim2_reversal/08_l1000_setup.py --download-gctx")
        print(f"  Or download manually from:")
        print(f"  {GEO_FTP}/{GCTX_FILE}")

    print("\nNext: run 09_l1000_signatures.py to extract per-drug consensus")
    print("signatures from the .gctx and apply network propagation.")


if __name__ == "__main__":
    import sys
    if "--download-gctx" in sys.argv:
        print("Downloading Level 5 .gctx (~3 GB)...")
        download(GCTX_FILE, DATA_RAW)
    main()
