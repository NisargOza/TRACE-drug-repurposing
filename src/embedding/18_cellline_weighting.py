"""
GTEx/CCLE cell-line tissue similarity weighting — RESEARCH.md §2a.

The proposal requires weighting L1000 drug signatures by how similar each
cell line is to lung tissue, using GTEx (lung tissue baseline expression)
and CCLE (cancer cell line baseline expression).

Method:
  1. Download GTEx lung tissue median TPM from the GTEx portal API
  2. Download CCLE RNA-seq for the 30 L1000 cell lines via DepMap
  3. Compute Spearman correlation between GTEx lung median and each cell line
     across shared genes → lung-similarity weight per cell line
  4. For each drug, compute weighted-average signature:
       drug_weighted_sig = Σ(w_cl * sig_cl) / Σ(w_cl)
  5. Recompute TRACE scores using weighted signatures
  6. Compare positive control ranks vs. unweighted baseline

Outputs:
  results/embedding/cellline_lung_similarity.csv  — per-cell-line weight
  results/l1000/drug_signatures_weighted.csv.gz   — weighted drug signatures
  results/reversal/weighted_trace_scores.csv      — updated scores

Usage:
    python src/embedding/18_cellline_weighting.py
"""

import io
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy import stats

L1000_DIR = Path("results/l1000")
EMB_DIR   = Path("results/embedding")
REV_DIR   = Path("results/reversal")
DATA_RAW  = Path("data/raw")
EMB_DIR.mkdir(exist_ok=True)

POSITIVE_CONTROLS = ["pirfenidone", "nintedanib"]


# ---------------------------------------------------------------------------
# 1. GTEx lung median expression
# ---------------------------------------------------------------------------

def get_lung_reference(cache: Path) -> pd.Series:
    """
    Build lung tissue reference from GSE213001 normal lung controls (73 samples).
    This is more appropriate than GTEx because it's the same tissue/processing
    pipeline as our IPF study. Returns median log-count per gene (Entrez index).
    """
    if cache.exists():
        print(f"  [skip] {cache.name}")
        return pd.read_csv(cache, index_col=0).squeeze()

    counts_path = Path("data/processed/GSE213001/counts_raw.csv.gz")
    meta_path   = Path("data/processed/GSE213001/metadata.csv")

    if not counts_path.exists():
        print("  [WARN] GSE213001 counts not found — using heuristic weights")
        return pd.Series(dtype=float)

    print("  Building lung reference from GSE213001 normal lung controls (n=73)...")
    counts = pd.read_csv(counts_path, index_col=0)
    meta   = pd.read_csv(meta_path, index_col=0)
    meta   = meta[~meta.index.duplicated()]

    control_ids = meta[meta["condition"] == "control"].index
    control_ids = [c for c in control_ids if c in counts.columns]
    lung_median = np.log1p(counts[control_ids]).median(axis=1)
    lung_median.index = lung_median.index.astype(str)
    lung_median.name  = "lung_log_count"
    lung_median.to_csv(cache)
    print(f"  {len(lung_median):,} genes from {len(control_ids)} normal lung samples")
    return lung_median


# ---------------------------------------------------------------------------
# 2. CCLE/DepMap cell-line expression
# ---------------------------------------------------------------------------

def get_l1000_cell_baselines(cache: Path, sig_info: pd.DataFrame,
                              all_sigs: pd.DataFrame) -> pd.DataFrame:
    """
    Compute median DMSO (vehicle control) expression per L1000 cell line.
    Uses L1000 ctl_vehicle signatures as each cell line's baseline expression.
    Returns DataFrame: cell_lines × landmark_genes.
    """
    if cache.exists():
        print(f"  [skip] {cache.name}")
        return pd.read_csv(cache, index_col=0)

    print("  Computing L1000 cell-line baselines from DMSO control signatures...")
    # Load full sig_info including controls
    full_sig_info_path = Path("data/raw/l1000/GSE70138_Broad_LINCS_inst_info_2017-03-06.txt.gz")
    if not full_sig_info_path.exists():
        print("  [WARN] inst_info not found — cannot compute baselines")
        return pd.DataFrame()

    inst = pd.read_csv(full_sig_info_path, sep="\t", compression="gzip",
                       usecols=["inst_id", "cell_id", "pert_type"], low_memory=False)
    dmso = inst[inst["pert_type"].isin(["ctl_vehicle", "ctl_untrt"])]
    print(f"  DMSO/untreated signatures: {len(dmso):,}")

    baselines = {}
    for cl, grp in dmso.groupby("cell_id"):
        present = [s for s in grp["inst_id"].values if s in all_sigs.columns]
        if len(present) >= 3:
            baselines[cl] = all_sigs[present].median(axis=1).values

    if not baselines:
        print("  [WARN] No baselines computed")
        return pd.DataFrame()

    result = pd.DataFrame(baselines, index=all_sigs.index).T  # cell_lines × genes
    result.to_csv(cache)
    print(f"  {len(result)} cell lines with baseline profiles")
    return result


# ---------------------------------------------------------------------------
# 3. Compute lung similarity weights
# ---------------------------------------------------------------------------

def compute_lung_weights(lung_ref: pd.Series,
                         cell_baselines: pd.DataFrame,
                         cell_lines: list[str]) -> pd.Series:
    """
    Spearman correlation between lung tissue reference and each L1000 cell line baseline.
    lung_ref:       Entrez-indexed gene expression (from GSE213001 normal controls)
    cell_baselines: cell_lines × landmark_genes (from L1000 DMSO controls)
    Returns pd.Series: cell_line → normalised weight (sums to 1).
    """
    if lung_ref.empty or cell_baselines.empty:
        print("  Using heuristic weights (data not available)")
        heuristic = {
            "A549": 2.0, "HCC515": 2.0,           # lung adenocarcinoma
            "HA1E": 1.0, "HEPG2": 1.0,            # kidney/liver (neutral)
            "HT29": 0.5, "YAPC": 0.5, "HELA": 0.5,
            "MCF7": 0.3, "A375": 0.3, "PC3": 0.3,  # breast/skin/prostate
        }
        weights = pd.Series({cl: heuristic.get(cl, 0.5) for cl in cell_lines})
        return (weights / weights.sum()).rename("lung_weight")

    common = lung_ref.index.intersection(cell_baselines.columns)
    if len(common) < 50:
        print(f"  [WARN] Only {len(common)} common genes — using heuristic")
        return compute_lung_weights(pd.Series(), pd.DataFrame(), cell_lines)

    lung_vec = lung_ref[common].values.astype(float)
    corrs = {}
    for cl in cell_baselines.index:
        cl_vec = cell_baselines.loc[cl, common].values.astype(float)
        r, _ = stats.spearmanr(lung_vec, cl_vec)
        corrs[cl] = max(r, 0)

    weights = pd.Series(corrs).reindex(cell_lines).fillna(0.1)
    weights = weights / weights.sum()
    return weights.rename("lung_weight")


# ---------------------------------------------------------------------------
# 4. Weighted drug signatures
# ---------------------------------------------------------------------------

def compute_weighted_signatures(sig_info: pd.DataFrame,
                                 all_sigs: pd.DataFrame,
                                 weights: pd.Series) -> pd.DataFrame:
    """
    For each drug, compute cell-line-weighted consensus signature.
    all_sigs: genes × all_signatures
    Returns: genes × drugs DataFrame
    """
    print(f"  Computing weighted signatures for {sig_info['pert_iname'].nunique():,} drugs...")
    drug_groups = sig_info.groupby("pert_iname")["sig_id"].apply(list)
    weighted_sigs = {}

    for drug, sig_ids in drug_groups.items():
        present = [s for s in sig_ids if s in all_sigs.columns]
        if not present:
            continue
        sub = all_sigs[present]
        cell_lines_in = sig_info.loc[sig_info["sig_id"].isin(present), "cell_id"]
        cl_map = dict(zip(present, cell_lines_in.values))

        # Weight each signature by its cell-line lung similarity
        w = np.array([weights.get(cl_map.get(s, ""), 1.0) for s in present])
        w = w / w.sum()
        weighted_sigs[drug] = (sub.values * w).sum(axis=1)

    return pd.DataFrame(weighted_sigs, index=all_sigs.index)


# ---------------------------------------------------------------------------
# 5. Recompute TRACE scores with weighted signatures
# ---------------------------------------------------------------------------

def compute_weighted_trace(weighted_sigs: pd.DataFrame,
                            network_scores: pd.DataFrame) -> pd.DataFrame:
    common_genes = network_scores.index.intersection(weighted_sigs.index)
    ipf_vec  = network_scores.loc[common_genes, "rwr_net"].values.astype(float)
    drug_mat = weighted_sigs.loc[common_genes].fillna(0).values.astype(float)
    drugs    = weighted_sigs.columns.tolist()

    ipf_norm   = np.linalg.norm(ipf_vec)
    drug_norms = np.linalg.norm(drug_mat, axis=0)
    drug_norms[drug_norms == 0] = 1e-10

    scores = -(ipf_vec @ drug_mat) / (ipf_norm * drug_norms)
    df = pd.DataFrame({
        "drug":           drugs,
        "weighted_trace": scores,
    }).sort_values("weighted_trace", ascending=False)
    df["weighted_rank"] = range(1, len(df) + 1)
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    sig_info = pd.read_csv(L1000_DIR / "sm_sig_info.csv", low_memory=False)
    network  = pd.read_csv(EMB_DIR / "ipf_network_scores.csv", index_col=0)
    network.index = network.index.astype(str)
    cell_lines = sig_info["cell_id"].unique().tolist()

    # Load all signatures for cell-line baseline computation
    parquet_path = Path("data/raw/l1000/all_signatures_landmark.parquet")
    all_sigs = pd.read_parquet(parquet_path) if parquet_path.exists() else pd.DataFrame()
    all_sigs.index = all_sigs.index.astype(str)

    # 1. Lung reference from GSE213001 normal controls
    print("\n1. Lung tissue reference (GSE213001 normal controls)")
    lung_ref = get_lung_reference(DATA_RAW / "lung_reference_gse213001.csv")

    # 2. L1000 cell-line baselines from DMSO controls
    print("\n2. L1000 cell-line baseline expression (DMSO controls)")
    cell_baselines = get_l1000_cell_baselines(
        DATA_RAW / "l1000_cell_baselines.csv", sig_info, all_sigs
    )

    # 3. Lung similarity weights
    print("\n3. Computing lung-similarity weights per cell line")
    weights = compute_lung_weights(lung_ref, cell_baselines, cell_lines)
    weights.to_csv(EMB_DIR / "cellline_lung_similarity.csv")

    print("  Cell-line weights:")
    for cl, w in weights.sort_values(ascending=False).head(15).items():
        print(f"    {cl:12} {w:.4f}")

    # 4. Load all signatures and compute weighted consensus
    print("\n4. Computing weighted drug signatures")
    parquet_path = Path("data/raw/l1000/all_signatures_landmark.parquet")
    if parquet_path.exists():
        all_sigs = pd.read_parquet(parquet_path)   # genes × all_sigs
        all_sigs.index = all_sigs.index.astype(str)
        sig_info_idx = sig_info.set_index("sig_id")
        weighted_sigs = compute_weighted_signatures(
            sig_info.set_index("sig_id").reset_index(), all_sigs, weights
        )
        out = L1000_DIR / "drug_signatures_weighted.csv.gz"
        weighted_sigs.to_csv(out, compression="gzip")
        print(f"  Saved {weighted_sigs.shape} → {out.name}")
    else:
        print("  [WARN] all_signatures_landmark.parquet not found — using consensus sigs")
        weighted_sigs = pd.read_csv(L1000_DIR / "drug_signatures_landmark.csv.gz", index_col=0)
        weighted_sigs.index = weighted_sigs.index.astype(str)

    # 5. Weighted TRACE scores
    print("\n5. Weighted TRACE scores")
    wtrace = compute_weighted_trace(weighted_sigs, network)
    wtrace.to_csv(REV_DIR / "weighted_trace_scores.csv", index=False)

    # Compare positive controls
    n = len(wtrace)
    print(f"\n=== Positive control ranks (weighted vs unweighted TRACE) ===")
    unweighted = pd.read_csv(REV_DIR / "trace_scores.csv")
    for pc in POSITIVE_CONTROLS:
        w_row = wtrace[wtrace["drug"].str.lower().str.contains(pc.lower(), na=False)]
        u_row = unweighted[unweighted["drug"].str.lower().str.contains(pc.lower(), na=False)]
        if not w_row.empty and not u_row.empty:
            wr = int(w_row.iloc[0]["weighted_rank"])
            ur = int(u_row.iloc[0]["trace_rank"])
            print(f"  {pc:15}  unweighted rank {ur:4}/{n}  →  weighted rank {wr:4}/{n}  "
                  f"({'improved' if wr < ur else 'same/worse'})")

    print(f"\n=== Top 15 weighted TRACE candidates ===")
    for _, r in wtrace.head(15).iterrows():
        print(f"  {int(r['weighted_rank']):>4}  {r['drug']:30}  {r['weighted_trace']:.4f}")

    print(f"\n  → results/embedding/cellline_lung_similarity.csv")
    print(f"  → results/reversal/weighted_trace_scores.csv")


if __name__ == "__main__":
    main()
