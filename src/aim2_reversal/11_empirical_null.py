"""
Empirical null, p-values, FDR, and bootstrap CI — RESEARCH.md §2c.

Steps:
  1. Build null distribution by scrambling the IPF signature gene labels
     (>= 1,000 permutations) and re-scoring all drugs under each permutation.
     Reversal scores must collapse toward null under scrambling.
  2. Compute empirical p-value per drug: fraction of null scores >= observed score.
  3. Apply Benjamini-Hochberg FDR correction.
  4. Bootstrap confidence intervals on TRACE rank (resample IPF datasets).
  5. Write final ranked candidate list with all statistical annotations.

Outputs:
  results/reversal/null_distribution.npz       — permutation null scores
  results/reversal/fdr_candidates.csv          — FDR-controlled candidate list
  results/reversal/null_diagnostics.png        — null vs observed score distribution
  results/reversal/positive_control_summary.txt

Usage:
    python src/aim2_reversal/11_empirical_null.py [--n-perm 1000]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

L1000_DIR = Path("results/l1000")
META_DIR  = Path("results/meta")
EMB_DIR   = Path("results/embedding")
REV_DIR   = Path("results/reversal")

N_PERM    = int(sys.argv[sys.argv.index("--n-perm") + 1]) if "--n-perm" in sys.argv else 1000
POSITIVE_CONTROLS = ["pirfenidone", "nintedanib"]
FDR_THRESH = 0.05
SCORE_COL  = "trace_score"   # which score to build null for


# ---------------------------------------------------------------------------
# Fast TRACE score: cosine similarity in network space (vectorised)
# ---------------------------------------------------------------------------

def score_all_drugs_vectorised(disease_vec: np.ndarray,
                                drug_matrix: np.ndarray) -> np.ndarray:
    """
    Compute -cosine(disease_vec, drug_col) for each drug (column) in drug_matrix.
    Negative cosine = reversal.
    drug_matrix: genes × drugs
    """
    d_norm = np.linalg.norm(disease_vec)
    col_norms = np.linalg.norm(drug_matrix, axis=0)
    col_norms[col_norms == 0] = 1e-10
    dots = disease_vec @ drug_matrix          # shape: (n_drugs,)
    return -(dots / (d_norm * col_norms))     # negative cosine = reversal score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Load inputs
    print("Loading inputs...")
    consensus = pd.read_csv(META_DIR / "consensus_signature.csv", index_col=0)
    consensus.index = consensus.index.astype(str)

    network = pd.read_csv(EMB_DIR / "ipf_network_scores.csv", index_col=0)
    network.index = network.index.astype(str)

    sig_matrix = pd.read_csv(L1000_DIR / "drug_signatures_landmark.csv.gz", index_col=0)
    sig_matrix.index = sig_matrix.index.astype(str)

    observed = pd.read_csv(REV_DIR / "trace_scores.csv")
    drugs = observed["drug"].tolist()
    obs_scores = observed.set_index("drug")[SCORE_COL]

    # Align genes: network ∩ landmark
    common_genes = network.index.intersection(sig_matrix.index)
    ipf_vec  = network.loc[common_genes, "rwr_net"].values.astype(float)
    drug_mat = sig_matrix.loc[common_genes, drugs].fillna(0).values.astype(float)
    n_genes, n_drugs = drug_mat.shape
    print(f"  {n_drugs} drugs × {n_genes} genes (network ∩ landmark)")

    # Verify vectorised scores match saved scores
    check = score_all_drugs_vectorised(ipf_vec, drug_mat)
    check_series = pd.Series(check, index=drugs)
    max_diff = (check_series - obs_scores.reindex(drugs)).abs().max()
    print(f"  Score recomputation check — max diff: {max_diff:.2e} {'OK' if max_diff < 1e-4 else 'WARN'}")

    # -----------------------------------------------------------------------
    # Permutation null
    # -----------------------------------------------------------------------
    null_path = REV_DIR / "null_distribution.npz"
    if null_path.exists():
        print(f"\nLoading cached null ({null_path.name})...")
        null_scores = np.load(null_path)["null_scores"]
        print(f"  {null_scores.shape[0]} permutations loaded")
    else:
        print(f"\nBuilding null distribution ({N_PERM} permutations)...")
        rng = np.random.default_rng(42)
        null_scores = np.empty((N_PERM, n_drugs), dtype=np.float32)
        for i in range(N_PERM):
            if i % 100 == 0:
                print(f"  {i}/{N_PERM}", end="\r", flush=True)
            perm_vec = rng.permutation(ipf_vec)
            null_scores[i] = score_all_drugs_vectorised(perm_vec, drug_mat)
        print(f"  {N_PERM}/{N_PERM} done        ")
        np.savez_compressed(null_path, null_scores=null_scores)
        print(f"  Saved null_distribution.npz")

    # -----------------------------------------------------------------------
    # Empirical p-values
    # -----------------------------------------------------------------------
    print("\nComputing empirical p-values...")
    obs_arr = obs_scores.reindex(drugs).values
    # p = fraction of null scores >= observed (one-tailed: high score = reversal)
    emp_pvals = (null_scores >= obs_arr[None, :]).mean(axis=0)
    emp_pvals = np.clip(emp_pvals, 1 / N_PERM, 1.0)   # minimum = 1/N_PERM

    reject, padj, _, _ = multipletests(emp_pvals, method="fdr_bh")

    results = pd.DataFrame({
        "drug":          drugs,
        "trace_score":   obs_arr,
        "emp_pvalue":    emp_pvals,
        "emp_padj":      padj,
        "significant":   reject,
    }).sort_values("trace_score", ascending=False)
    results["trace_rank"] = range(1, len(results) + 1)

    # Add baseline score
    baseline = pd.read_csv(REV_DIR / "baseline_scores.csv")[["drug", "baseline_score", "baseline_rank"]]
    results = results.merge(baseline, on="drug", how="left")

    # Save full results
    results.to_csv(REV_DIR / "fdr_candidates.csv", index=False)

    n_sig = reject.sum()
    print(f"  FDR < {FDR_THRESH}: {n_sig} drugs")

    # -----------------------------------------------------------------------
    # Diagnostics plot
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # 1. Null vs observed distribution
    ax = axes[0]
    null_flat = null_scores.flatten()
    ax.hist(null_flat, bins=80, color="#aaaaaa", alpha=0.6,
            density=True, label="Null (permuted)")
    ax.hist(obs_arr, bins=60, color="#d62728", alpha=0.7,
            density=True, label="Observed")
    ax.set_xlabel("TRACE score")
    ax.set_ylabel("Density")
    ax.set_title("Observed vs null distribution")
    ax.legend(fontsize=8)

    # 2. Empirical p-value histogram
    ax = axes[1]
    ax.hist(emp_pvals, bins=40, color="#1f77b4", edgecolor="white")
    ax.axvline(FDR_THRESH, color="red", lw=1.5, ls="--", label=f"α={FDR_THRESH}")
    ax.set_xlabel("Empirical p-value")
    ax.set_ylabel("Count")
    ax.set_title("P-value distribution")
    ax.legend(fontsize=8)

    # 3. Score vs -log10(padj)
    ax = axes[2]
    non_sig = results[~results["significant"]]
    sig_df  = results[results["significant"]]
    ax.scatter(non_sig["trace_score"], -np.log10(non_sig["emp_padj"] + 1e-10),
               s=4, alpha=0.3, c="#aaaaaa")
    ax.scatter(sig_df["trace_score"],  -np.log10(sig_df["emp_padj"]  + 1e-10),
               s=8, alpha=0.7, c="#d62728", label="FDR sig")
    # Highlight positive controls
    for pc in POSITIVE_CONTROLS:
        row = results[results["drug"].str.lower().str.contains(pc.lower(), na=False)]
        if not row.empty:
            r = row.iloc[0]
            ax.scatter(r["trace_score"], -np.log10(r["emp_padj"] + 1e-10),
                       s=80, marker="*", c="#ff7f0e", zorder=5)
            ax.annotate(pc, (r["trace_score"], -np.log10(r["emp_padj"] + 1e-10)),
                        fontsize=7, ha="right")
    ax.set_xlabel("TRACE score")
    ax.set_ylabel("-log10(FDR adj. p)")
    ax.set_title("Drug ranking — FDR significance")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(REV_DIR / "null_diagnostics.png", dpi=150)
    plt.close(fig)
    print(f"  Saved null_diagnostics.png")

    # -----------------------------------------------------------------------
    # Positive control summary
    # -----------------------------------------------------------------------
    n_total = len(results)
    lines = [
        f"Empirical null: {N_PERM} permutations",
        f"Total drugs scored: {n_total}",
        f"FDR < {FDR_THRESH}: {n_sig} drugs",
        "",
        "=== Positive control summary ===",
    ]
    for pc in POSITIVE_CONTROLS:
        row = results[results["drug"].str.lower().str.contains(pc.lower(), na=False)]
        if row.empty:
            lines.append(f"  {pc}: NOT FOUND")
            continue
        r = row.iloc[0]
        lines.append(
            f"  {pc:15}  TRACE rank {int(r['trace_rank']):4}/{n_total} "
            f"({r['trace_rank']/n_total*100:.1f}th pct)  "
            f"emp_p={r['emp_pvalue']:.4f}  FDR={r['emp_padj']:.4f}  "
            f"sig={r['significant']}"
        )

    lines += ["", "=== Top 20 FDR-significant TRACE candidates ===",
              f"{'Rank':>4}  {'Drug':30}  {'TRACE':>7}  {'FDR':>8}  {'Baseline rank':>14}"]
    top = results[results["significant"]].head(20)
    for _, r in top.iterrows():
        lines.append(
            f"  {int(r['trace_rank']):>4}  {r['drug']:30}  "
            f"{r['trace_score']:>7.4f}  {r['emp_padj']:>8.4f}  "
            f"{int(r['baseline_rank']) if pd.notna(r['baseline_rank']) else 'N/A':>14}"
        )

    report = "\n".join(lines)
    (REV_DIR / "positive_control_summary.txt").write_text(report)
    print(f"\n{report}")
    print(f"\nNext: bootstrap CI on candidate ranks, then ablation study.")


if __name__ == "__main__":
    main()
