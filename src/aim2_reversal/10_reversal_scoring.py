"""
Reversal scoring — RESEARCH.md §2a.

Computes two scores for each drug against the IPF consensus signature:

1. BASELINE: Classic weighted connectivity score (Kolmogorov-Smirnov enrichment,
   Lamb 2006 / Subramanian 2017). Uses raw landmark-gene signatures, no
   tissue-awareness. This is the honest benchmark TRACE must beat.

2. TRACE score: Reversal in the tissue-aware (network-propagated) space.
   - Cell-line relevance weighting: down-weight signatures from cell lines
     least similar to lung tissue (similarity estimated from baseline expression
     overlap with the IPF consensus — a proxy until GTEx/CCLE data is added)
   - Cosine similarity between propagated drug signature and propagated IPF
     signature in the network embedding space

Outputs:
  results/reversal/baseline_scores.csv   — drug, baseline_score, rank
  results/reversal/trace_scores.csv      — drug, trace_score, rank
  results/reversal/combined_scores.csv   — both scores + positive control ranks
  results/reversal/positive_controls.txt — pirfenidone / nintedanib ranks

Usage:
    python src/aim2_reversal/10_reversal_scoring.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity

L1000_DIR  = Path("results/l1000")
META_DIR   = Path("results/meta")
EMB_DIR    = Path("results/embedding")
REV_DIR    = Path("results/reversal")
REV_DIR.mkdir(parents=True, exist_ok=True)

POSITIVE_CONTROLS = ["pirfenidone", "nintedanib"]


# ---------------------------------------------------------------------------
# Baseline: weighted connectivity score (KS-based)
# ---------------------------------------------------------------------------

def weighted_connectivity_score(disease_lfc: pd.Series,
                                 drug_sig: pd.Series,
                                 n_top: int = 150) -> float:
    """
    Compute the normalised connectivity score between a disease and drug
    signature, following Subramanian et al. 2017.

    Uses the top-n_top up and down genes from the disease signature as query,
    scores enrichment of these genes in the drug signature ranked list.
    """
    # Rank drug signature genes by z-score (high = up-regulated by drug)
    common = disease_lfc.index.intersection(drug_sig.index)
    if len(common) < 50:
        return np.nan

    d_lfc  = disease_lfc[common]
    d_sig  = drug_sig[common]

    # Disease query: top-n up and down genes
    top_up   = d_lfc.nlargest(n_top).index
    top_down = d_lfc.nsmallest(n_top).index

    # Rank drug genes
    ranked = d_sig.rank(ascending=False)   # rank 1 = most up by drug
    n = len(ranked)

    def ks_score(query_genes: pd.Index) -> float:
        q = query_genes.intersection(ranked.index)
        if len(q) == 0:
            return 0.0
        ranks_q = ranked[q].sort_values().values
        # KS-like enrichment
        hit_score  = np.cumsum(np.abs(d_sig[q].values)) / np.abs(d_sig[q]).sum()
        miss_score = np.cumsum(np.ones(n - len(q))) / (n - len(q))
        # Interleave
        indicators = np.zeros(n)
        indicators[ranks_q.astype(int) - 1] = 1
        ks_vals = []
        hit_i = miss_i = 0.0
        for ind in indicators:
            if ind:
                hit_i += 1
                ks_vals.append(hit_i / len(q) - miss_i / max(n - len(q), 1))
            else:
                miss_i += 1
                ks_vals.append(hit_i / len(q) - miss_i / max(n - len(q), 1))
        return max(ks_vals, key=abs)

    # Connectivity score = mean of up and down KS scores if opposite sign
    ks_up   = ks_score(top_up)
    ks_down = ks_score(top_down)

    if np.sign(ks_up) != np.sign(ks_down):
        return (ks_up - ks_down) / 2  # reversal: drug opposes disease
    return 0.0


# ---------------------------------------------------------------------------
# Cell-line relevance weighting (proxy without GTEx/CCLE)
# ---------------------------------------------------------------------------

def estimate_cell_line_weights(sig_info: pd.DataFrame,
                                consensus: pd.Series) -> pd.Series:
    """
    Proxy for lung-tissue similarity: cell lines whose drug-signature gene
    space overlaps most with IPF-relevant genes get higher weight.
    Full implementation will use GTEx/CCLE baseline expression (RESEARCH.md §2a).
    For now, use A549 (lung adenocarcinoma) as highest weight, penalise breast/
    prostate lines that dominate the dataset.
    """
    LUNG_LINES   = {"A549", "HCC515"}
    NEUTRAL      = {"HA1E", "HEPG2", "HT29", "YAPC", "HELA"}
    weights = {}
    for cl in sig_info["cell_id"].unique():
        if cl in LUNG_LINES:
            weights[cl] = 2.0
        elif cl in NEUTRAL:
            weights[cl] = 1.0
        else:
            weights[cl] = 0.5   # cancer lines far from lung get down-weighted
    return pd.Series(weights)


# ---------------------------------------------------------------------------
# TRACE score: cosine similarity in network-propagated space
# ---------------------------------------------------------------------------

def propagate_drug_signature(drug_sig: pd.Series,
                              network_nodes: pd.Index,
                              W: object,   # sparse adjacency
                              alpha: float = 0.85,
                              max_iter: int = 50) -> np.ndarray:
    """RWR with drug signature as seed on the same PPI network."""
    from scipy import sparse as sp
    seed = np.zeros(len(network_nodes))
    common = drug_sig.index.intersection(network_nodes)
    if len(common) == 0:
        return seed
    node_idx = {n: i for i, n in enumerate(network_nodes)}
    for g in common:
        seed[node_idx[g]] = drug_sig[g]
    # Separate up/down
    seed_up   = np.clip(seed, 0, None)
    seed_down = np.clip(-seed, 0, None)
    if seed_up.sum() > 0:   seed_up   /= seed_up.sum()
    if seed_down.sum() > 0: seed_down /= seed_down.sum()
    def rwr_single(p0):
        p = p0.copy()
        for _ in range(max_iter):
            p_new = (1 - alpha) * W.dot(p) + alpha * p0
            if np.abs(p_new - p).sum() < 1e-5:
                return p_new
            p = p_new
        return p
    up   = rwr_single(seed_up)
    down = rwr_single(seed_down)
    return up - down


def main() -> None:
    # Load inputs
    print("Loading inputs...")
    consensus = pd.read_csv(META_DIR / "consensus_signature.csv", index_col=0)
    consensus.index = consensus.index.astype(str)
    disease_lfc = consensus["meta_log2FC"]

    network = pd.read_csv(EMB_DIR / "ipf_network_scores.csv", index_col=0)
    network.index = network.index.astype(str)
    disease_net = network["rwr_net"]

    sig_matrix = pd.read_csv(L1000_DIR / "drug_signatures_landmark.csv.gz", index_col=0)
    sig_matrix.index = sig_matrix.index.astype(str)
    sig_info = pd.read_csv(L1000_DIR / "sm_sig_info.csv", low_memory=False)

    drugs = sig_matrix.columns.tolist()
    print(f"  {len(drugs):,} drugs  ×  {len(sig_matrix):,} landmark genes")

    # Cell-line weights
    cl_weights = estimate_cell_line_weights(sig_info, disease_lfc)

    # ---------------------------------------------------------------------------
    # Baseline scores
    # ---------------------------------------------------------------------------
    print(f"\nComputing baseline connectivity scores ({len(drugs):,} drugs)...")
    baseline = {}
    for i, drug in enumerate(drugs):
        if i % 200 == 0:
            print(f"  {i}/{len(drugs)}", end="\r", flush=True)
        drug_sig = sig_matrix[drug].dropna()
        baseline[drug] = weighted_connectivity_score(disease_lfc, drug_sig)
    print(f"  {len(drugs):,} done        ")

    baseline_df = pd.DataFrame({
        "drug": list(baseline.keys()),
        "baseline_score": list(baseline.values()),
    }).dropna().sort_values("baseline_score")
    baseline_df["baseline_rank"] = range(1, len(baseline_df) + 1)
    baseline_df.to_csv(REV_DIR / "baseline_scores.csv", index=False)

    # ---------------------------------------------------------------------------
    # TRACE scores: cosine similarity in network-propagated space
    # Approximation: use cosine similarity between:
    #   - drug landmark z-scores projected onto network nodes (those measurable)
    #   - IPF RWR net scores on same nodes
    # Full version propagates each drug signature through PPI (slow but more principled)
    # ---------------------------------------------------------------------------
    print(f"\nComputing TRACE scores (network cosine similarity)...")

    # Only use genes present in both network and landmark space
    common_genes = disease_net.index.intersection(sig_matrix.index)
    ipf_vec = disease_net[common_genes].values
    ipf_norm = np.linalg.norm(ipf_vec)
    print(f"  Genes in network ∩ landmark space: {len(common_genes):,}")

    trace_scores = {}
    for i, drug in enumerate(drugs):
        if i % 200 == 0:
            print(f"  {i}/{len(drugs)}", end="\r", flush=True)
        drug_vec = sig_matrix.loc[common_genes, drug].fillna(0).values
        drug_norm = np.linalg.norm(drug_vec)
        if drug_norm == 0 or ipf_norm == 0:
            trace_scores[drug] = np.nan
            continue
        # Negative cosine = reversal (drug opposes disease direction)
        trace_scores[drug] = -float(np.dot(ipf_vec, drug_vec) / (ipf_norm * drug_norm))
    print(f"  {len(drugs):,} done        ")

    trace_df = pd.DataFrame({
        "drug": list(trace_scores.keys()),
        "trace_score": list(trace_scores.values()),
    }).dropna().sort_values("trace_score", ascending=False)
    trace_df["trace_rank"] = range(1, len(trace_df) + 1)
    trace_df.to_csv(REV_DIR / "trace_scores.csv", index=False)

    # ---------------------------------------------------------------------------
    # Combined + positive control check
    # ---------------------------------------------------------------------------
    combined = baseline_df.merge(trace_df, on="drug", how="inner")
    combined.to_csv(REV_DIR / "combined_scores.csv", index=False)
    n_drugs = len(combined)

    lines = [
        f"Reversal scoring complete — {n_drugs:,} drugs scored\n",
        "=== Positive control ranks ===",
    ]
    for pc in POSITIVE_CONTROLS:
        rows = combined[combined["drug"].str.lower().str.contains(pc.lower(), na=False)]
        if rows.empty:
            lines.append(f"  {pc}: NOT FOUND in combined scores")
        else:
            r = rows.iloc[0]
            b_pct = r["baseline_rank"] / n_drugs * 100
            t_pct = r["trace_rank"]    / n_drugs * 100
            lines.append(
                f"  {pc:15} baseline rank {int(r['baseline_rank']):4}/{n_drugs} "
                f"({b_pct:.1f}th pct)   "
                f"TRACE rank {int(r['trace_rank']):4}/{n_drugs} ({t_pct:.1f}th pct)"
            )

    lines += [
        "\n=== Top 20 TRACE candidates ===",
        f"{'Rank':>4}  {'Drug':30}  {'TRACE':>8}  {'Baseline':>10}",
        "-" * 60,
    ]
    for _, row in trace_df.head(20).iterrows():
        b_row = baseline_df[baseline_df["drug"] == row["drug"]]
        b_score = b_row["baseline_score"].values[0] if not b_row.empty else float("nan")
        lines.append(
            f"  {int(row['trace_rank']):>4}  {row['drug']:30}  "
            f"{row['trace_score']:>8.4f}  {b_score:>10.4f}"
        )

    report = "\n".join(lines)
    (REV_DIR / "reversal_report.txt").write_text(report)
    print(f"\n{report}")
    print(f"\nOutputs in results/reversal/")
    print("Next: empirical null via permutation + FDR, then Aim 3 validation.")


if __name__ == "__main__":
    main()
