"""
Dual-disease drug scoring (Pearson + CMap) — Step B3.

Scores all L1000 drugs against IPF and RA consensus signatures using:
  1. Pearson correlation between disease network vector and drug signature
     (negative r = reversal candidate)
  2. CMap signed KS enrichment score (Lamb 2006 / Subramanian 2017)
     (negative score = reversal candidate)

Confirmed input formats:
  results/l1000/drug_signatures_landmark.csv.gz
    index col name: entrez_id  (int)
    other columns:  drug names (1,768 drugs)
  results/embedding/ipf_network_scores.csv
    index col name: entrez_id  (int or str)
    score column:   rwr_net
  results/embedding/ra_network_scores.csv  (same format)
  results/meta/consensus_signature.csv
    index: Entrez int, column: meta_log2FC
  results/meta/ra_consensus_signature.csv  (same format)

Outputs (all in results/benchmarking/):
  ipf_drug_scores.csv   — drug, pearson, cmap
  ra_drug_scores.csv    — drug, pearson, cmap
  dual_disease_scores.csv — merged

Usage:
    python src/benchmarking/B3_dual_disease_scoring.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as spstats

ROOT  = Path(__file__).resolve().parents[2]
L1K   = ROOT / "results" / "l1000"
EMB   = ROOT / "results" / "embedding"
META  = ROOT / "results" / "meta"
BENCH = ROOT / "results" / "benchmarking"
BENCH.mkdir(parents=True, exist_ok=True)

N_TOP = 150   # top/bottom genes for CMap query (Lamb 2006)


# ── CMap signed KS enrichment score ─────────────────────────────────────────

def cmap_score_single(disease_lfc: np.ndarray,
                      drug_sig: np.ndarray,
                      n_top: int = N_TOP) -> float:
    """
    Signed KS score for one drug.
    Returns value in [-1, 1]; negative = reversal (disease up, drug down).

    n_top is capped at n // 4 so the function always produces a score even
    when the query gene set is smaller than the default 150 (e.g. RA with
    76 landmark-overlapping consensus genes). Using the top/bottom 25% of
    available genes matches the adaptive query approach in Subramanian 2017.
    """
    n = len(disease_lfc)
    # Adaptive cap: never use more than 25% of available genes per tail
    n_top = min(n_top, n // 4)
    if n_top < 5:
        return np.nan

    sorted_idx = np.argsort(disease_lfc)
    dn_query = set(sorted_idx[:n_top].tolist())    # disease down-regulated
    up_query = set(sorted_idx[-n_top:].tolist())   # disease up-regulated

    drug_rank = spstats.rankdata(-drug_sig)         # rank 1 = most up by drug
    miss_step = 1.0 / max(n - n_top, 1)

    def ks_one_tail(query_set: set) -> float:
        hw     = np.array([abs(disease_lfc[g]) for g in query_set])
        hw_sum = hw.sum()
        if hw_sum == 0:
            return 0.0
        query_list = list(query_set)
        order      = np.argsort([drug_rank[g] for g in query_list])
        genes_ord  = [query_list[o] for o in order]

        cum, max_dev = 0.0, 0.0
        hit_seen     = 0
        rank_sorted  = np.argsort(drug_rank)

        for pos, gene_idx in enumerate(rank_sorted):
            if gene_idx in query_set:
                cum      += abs(disease_lfc[gene_idx]) / hw_sum
                hit_seen += 1
            else:
                cum -= miss_step
            if abs(cum) > abs(max_dev):
                max_dev = cum
        return max_dev

    ks_up = ks_one_tail(up_query)
    ks_dn = ks_one_tail(dn_query)

    # Connectivity: high when up-query enriched at top of drug list AND
    # dn-query enriched at bottom (opposite signs → reversal)
    if np.sign(ks_up) != np.sign(ks_dn):
        return float(ks_up - ks_dn) / 2.0
    return 0.0


def pearson_reversal(disease_vec: np.ndarray,
                     drug_matrix: np.ndarray) -> np.ndarray:
    """
    Vectorised Pearson r between disease_vec and each drug column.
    Returns NEGATIVE r so that higher score = stronger reversal.
    drug_matrix shape: (n_genes, n_drugs)
    """
    d     = disease_vec - disease_vec.mean()
    d_std = d.std()
    if d_std == 0:
        return np.zeros(drug_matrix.shape[1])
    dm      = drug_matrix - drug_matrix.mean(axis=0)
    dm_std  = dm.std(axis=0)
    dm_std[dm_std == 0] = 1.0
    r = (d @ dm) / (d_std * dm_std * len(d))
    return -r   # negate: high value = reversal


def score_disease(label: str,
                  net_scores_path: Path,
                  consensus_path: Path,
                  drug_mat: pd.DataFrame) -> pd.DataFrame:
    """Score all drugs for one disease. Returns DataFrame(drug, pearson, cmap)."""
    print(f"\n--- {label} ---")

    # Load network scores
    net = pd.read_csv(net_scores_path, index_col=0)
    net.index = net.index.astype(str)
    dis_net = net["rwr_net"]

    # Align network vector to drug matrix index
    drug_mat.index = drug_mat.index.astype(str)
    common_net = dis_net.index.intersection(drug_mat.index)
    print(f"  Network genes on drug matrix: {len(common_net):,}")
    dis_net_vec = dis_net.loc[common_net].values.astype(float)
    dm_net      = drug_mat.loc[common_net].values.astype(float)   # genes × drugs

    print(f"  Computing Pearson scores ({drug_mat.shape[1]:,} drugs) ...")
    pearson = pearson_reversal(dis_net_vec, dm_net)

    # Load consensus LFC for CMap
    cons = pd.read_csv(consensus_path, index_col=0)
    cons.index = cons.index.astype(str)
    common_lfc = cons.index.intersection(drug_mat.index)
    print(f"  Consensus genes on drug matrix: {len(common_lfc):,}")

    dis_lfc = cons.loc[common_lfc, "meta_log2FC"].values.astype(float)
    dm_lfc  = drug_mat.loc[common_lfc].values.astype(float)

    drugs = drug_mat.columns.tolist()
    print(f"  Computing CMap scores ...")
    cmap = np.zeros(len(drugs))
    for j in range(len(drugs)):
        if j % 500 == 0:
            print(f"    {j:,}/{len(drugs):,}", end="\r", flush=True)
        cmap[j] = cmap_score_single(dis_lfc, dm_lfc[:, j])
    print(f"    {len(drugs):,}/{len(drugs):,} done    ")

    return pd.DataFrame({"drug": drugs, "pearson": pearson, "cmap": cmap})


def main() -> None:
    print("Loading L1000 drug matrix ...")
    drug_mat = pd.read_csv(
        L1K / "drug_signatures_landmark.csv.gz", index_col=0
    )
    # index_col=0 gives index named "entrez_id"
    drug_mat.index = drug_mat.index.astype(str)
    print(f"  Drug matrix: {drug_mat.shape[0]:,} genes × {drug_mat.shape[1]:,} drugs")

    diseases = [
        (
            "IPF",
            EMB  / "ipf_network_scores.csv",
            META / "consensus_signature.csv",
        ),
        (
            "RA",
            EMB  / "ra_network_scores.csv",
            META / "ra_consensus_signature.csv",
        ),
    ]

    all_scores = {}
    for label, net_path, cons_path in diseases:
        if not net_path.exists():
            print(f"WARNING: {net_path.name} missing — skipping {label}")
            continue
        if not cons_path.exists():
            print(f"WARNING: {cons_path.name} missing — skipping {label}")
            continue

        scores = score_disease(label, net_path, cons_path, drug_mat)
        scores.to_csv(BENCH / f"{label.lower()}_drug_scores.csv", index=False)
        print(f"  Saved → results/benchmarking/{label.lower()}_drug_scores.csv")
        all_scores[label] = scores

    # Merged dual-disease file
    if len(all_scores) == 2:
        ipf = all_scores["IPF"].rename(
            columns={"pearson": "ipf_pearson", "cmap": "ipf_cmap"})
        ra  = all_scores["RA"].rename(
            columns={"pearson": "ra_pearson",  "cmap": "ra_cmap"})
        merged = ipf.merge(ra, on="drug", how="outer")
        merged.to_csv(BENCH / "dual_disease_scores.csv", index=False)
        print(f"\nMerged → results/benchmarking/dual_disease_scores.csv "
              f"({len(merged):,} drugs)")

    print("\nNext: python src/benchmarking/B4_ablation_dual.py")


if __name__ == "__main__":
    main()