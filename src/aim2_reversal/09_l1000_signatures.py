"""
Extract per-drug consensus signatures from L1000 .gctx — RESEARCH.md §2a.

For each drug:
  - Collect all Level 5 signatures (replicates across cell lines / doses)
  - Compute median z-score across replicates = consensus drug signature
  - Also compute cell-line-stratified medians for tissue-aware weighting

Outputs:
  results/l1000/drug_signatures_landmark.csv.gz  — drugs x 978 landmark genes
  results/l1000/drug_cell_signatures.pkl          — drug x cell_line x gene (for weighting)
  results/l1000/drug_metadata.csv                — drug name, n_sigs, cell_lines

Requires: data/raw/l1000/GSE70138_Broad_LINCS_Level5_COMPZ_*.gctx.gz

Usage:
    python src/aim2_reversal/09_l1000_signatures.py
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import cmapPy.pandasGEXpress.parse_gctx as parse_gctx
    HAS_CMAPPY = True
except ImportError:
    HAS_CMAPPY = False

DATA_RAW  = Path("data/raw/l1000")
L1000_DIR = Path("results/l1000")

GCTX_FILE = next(DATA_RAW.glob("*Level5_COMPZ*.gctx*"), None)


def extract_signatures() -> None:
    if not HAS_CMAPPY:
        raise ImportError("cmapPy not installed. Run: pip install cmapPy")
    if GCTX_FILE is None:
        raise FileNotFoundError(
            "Level 5 .gctx not found. Run:\n"
            "  python src/aim2_reversal/08_l1000_setup.py --download-gctx"
        )

    print(f"  Loading .gctx: {GCTX_FILE.name}")
    sig_info = pd.read_csv(L1000_DIR / "sm_sig_info.csv", low_memory=False)
    lm_genes = pd.read_csv(L1000_DIR / "landmark_genes.csv")

    # pr_gene_id = Entrez ID; row IDs in .gctx are also pr_gene_id (integer strings)
    lm_entrez = lm_genes["pr_gene_id"].astype(str).tolist()
    lm_ids    = lm_entrez  # .gctx row IDs match pr_gene_id

    sm_sig_ids = sig_info["sig_id"].tolist()

    print(f"  Parsing {len(sm_sig_ids):,} small-molecule signatures × {len(lm_ids):,} landmark genes...")
    gctx = parse_gctx.parse(str(GCTX_FILE),
                             cid=sm_sig_ids,
                             rid=lm_ids)
    mat = gctx.data_df  # landmark genes × signatures

    print(f"  Matrix shape: {mat.shape}")

    # Build drug → signature IDs map
    drug_to_sigs = sig_info.groupby("pert_iname")["sig_id"].apply(list).to_dict()
    drug_to_cell = sig_info.set_index("sig_id")["cell_id"].to_dict()

    print(f"  Computing per-drug consensus signatures ({len(drug_to_sigs):,} drugs)...")
    drug_medians   = {}
    drug_meta_rows = []

    for drug, sig_ids in drug_to_sigs.items():
        present = [s for s in sig_ids if s in mat.columns]
        if not present:
            continue
        sub = mat[present]
        drug_medians[drug] = sub.median(axis=1)  # median across all replicates
        cells = [drug_to_cell.get(s, "") for s in present]
        drug_meta_rows.append({
            "pert_iname": drug,
            "n_signatures": len(present),
            "n_cell_lines": len(set(cells)),
            "cell_lines": "|".join(sorted(set(cells))),
        })

    # Genes × drugs matrix (landmark, Entrez IDs as index)
    sig_matrix = pd.DataFrame(drug_medians)
    sig_matrix.index = lm_entrez  # replace probe IDs with Entrez IDs
    sig_matrix.index.name = "entrez_id"

    out = L1000_DIR / "drug_signatures_landmark.csv.gz"
    sig_matrix.to_csv(out, compression="gzip")
    print(f"  Saved drug_signatures_landmark.csv.gz  {sig_matrix.shape}")

    meta_df = pd.DataFrame(drug_meta_rows)
    meta_df.to_csv(L1000_DIR / "drug_metadata.csv", index=False)
    print(f"  Saved drug_metadata.csv  ({len(meta_df):,} drugs)")

    # Per-cell-line consensus (for tissue-aware weighting)
    print("  Computing per-drug × per-cell-line signatures...")
    drug_cell_sigs: dict[str, dict[str, pd.Series]] = {}
    for drug, sig_ids in drug_to_sigs.items():
        present = [s for s in sig_ids if s in mat.columns]
        if not present:
            continue
        sub = mat[present].copy()
        sub.columns = [drug_to_cell.get(s, "unknown") for s in present]
        by_cell = sub.T.groupby(level=0).median().T  # genes × cell_lines
        drug_cell_sigs[drug] = {cl: by_cell[cl] for cl in by_cell.columns}

    with open(L1000_DIR / "drug_cell_signatures.pkl", "wb") as f:
        pickle.dump(drug_cell_sigs, f)
    print(f"  Saved drug_cell_signatures.pkl")


def main() -> None:
    out = L1000_DIR / "drug_signatures_landmark.csv.gz"
    if out.exists():
        print(f"[skip] {out.name} already exists")
        sig_matrix = pd.read_csv(out, index_col=0)
        print(f"  {sig_matrix.shape[1]:,} drugs × {sig_matrix.shape[0]:,} landmark genes")
    else:
        extract_signatures()

    print("\nNext: run 10_reversal_scoring.py to compute baseline connectivity")
    print("score and TRACE tissue-aware reversal score.")


if __name__ == "__main__":
    main()
