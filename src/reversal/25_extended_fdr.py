
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REV  = ROOT / "results" / "reversal"
META = ROOT / "results" / "meta"
EMB  = ROOT / "results" / "embedding"

N_PERMS = 10_000
TOP_N   = 100

def main():
    ipf_net = pd.read_csv(EMB / "ipf_network_scores.csv", index_col=0)
    score_col = [c for c in ipf_net.columns if "net" in c.lower() or "score" in c.lower()]
    if score_col:
        ipf_vec = ipf_net[score_col[0]].values
    else:
        ipf_vec = ipf_net.iloc[:, 0].values
    gene_ids = ipf_net.index.values
    print(f"IPF network vector: {len(ipf_vec)} genes")

    trace = pd.read_csv(REV / "trace_scores.csv")
    top100 = trace.nlargest(TOP_N, "trace_score").copy()
    print(f"Top-{TOP_N} drugs by Net-TRACE score: {list(top100['drug'][:5])}...")

    drug_sig_path = REV / "final_candidates_full.csv"
    cand = pd.read_csv(drug_sig_path)

    obs_scores = {}
    for _, row in top100.iterrows():
        obs_scores[row["drug"]] = float(row["trace_score"])


    drug_net_path = REV / "drug_network_scores.csv"
    if not drug_net_path.exists():
        print("Per-drug network profiles not saved; using landmark-gene permutation proxy")
        use_proxy = True
    else:
        use_proxy = False
        drug_net = pd.read_csv(drug_net_path, index_col=0)

    import gzip
    drug_sigs_path = ROOT / "results" / "l1000" / "drug_signatures_landmark.csv.gz"
    print("Loading drug signatures...")
    drug_sigs_raw = pd.read_csv(drug_sigs_path, index_col=0)
    if drug_sigs_raw.shape[0] < drug_sigs_raw.shape[1]:
        drug_sigs = drug_sigs_raw.T
    else:
        drug_sigs = drug_sigs_raw

    top100_drugs_in_sigs = [d for d in top100["drug"].values if d in drug_sigs.index]
    print(f"Top-100 drugs with landmark signatures: {len(top100_drugs_in_sigs)}")

    if len(top100_drugs_in_sigs) == 0:
        print("Drug signatures missing — cannot run extended FDR")
        return

    consensus = pd.read_csv(META / "consensus_signature.csv", index_col=0)
    common_genes = list(set(drug_sigs.columns.tolist()) & set(consensus.index.tolist()))
    print(f"Common genes (landmark ∩ consensus): {len(common_genes)}")
    consensus_landmark = consensus.loc[common_genes, "meta_log2FC"].values.astype(float)
    drug_mat_aligned   = drug_sigs[common_genes].values.astype(float)
    ipf_lm = consensus_landmark
    ipf_lm_norm = ipf_lm / (np.linalg.norm(ipf_lm) + 1e-12)

    drug_idx = [i for i, d in enumerate(drug_sigs.index) if d in top100_drugs_in_sigs]
    drug_mat = drug_mat_aligned[drug_idx]
    top100_drugs_in_sigs = [drug_sigs.index[i] for i in drug_idx]
    norms = np.linalg.norm(drug_mat, axis=1, keepdims=True)
    drug_mat_norm = drug_mat / (norms + 1e-12)

    obs_cos = -drug_mat_norm @ ipf_lm_norm

    rng = np.random.default_rng(42)
    null_mat = np.zeros((N_PERMS, len(top100_drugs_in_sigs)), dtype=np.float32)

    print(f"Running {N_PERMS:,} permutations (top-{len(top100_drugs_in_sigs)} drugs)...")
    for i in range(N_PERMS):
        ipf_perm = rng.permutation(ipf_lm_norm)
        null_mat[i] = -drug_mat_norm @ ipf_perm
        if (i + 1) % 2000 == 0:
            print(f"  {i+1:,}/{N_PERMS:,}")

    np.savez_compressed(REV / "extended_fdr_null.npz",
                        null_scores=null_mat,
                        drugs=np.array(top100_drugs_in_sigs))
    print("Saved extended_fdr_null.npz")

    emp_pvals = np.array([
        (null_mat[:, j] >= obs_cos[j]).mean()
        for j in range(len(obs_cos))
    ])
    from statsmodels.stats.multitest import multipletests
    _, fdr_vals, _, _ = multipletests(emp_pvals, method="fdr_bh")

    results = pd.DataFrame({
        "drug": top100_drugs_in_sigs,
        "net_trace_cos": obs_cos,
        "emp_pval": emp_pvals,
        "bh_fdr": fdr_vals,
        "fdr_sig": fdr_vals < 0.05,
    }).sort_values("emp_pval")

    results.to_csv(REV / "extended_fdr_results.csv", index=False)

    n_sig = results["fdr_sig"].sum()
    lines = [
        f"Extended FDR — {N_PERMS:,} permutations on top-{len(top100_drugs_in_sigs)} drugs",
        "=" * 60,
        f"Min achievable p-value: 1/{N_PERMS} = {1/N_PERMS:.1e}",
        f"BH FDR threshold for {len(top100_drugs_in_sigs)} tests: 0.05/{len(top100_drugs_in_sigs)} = {0.05/len(top100_drugs_in_sigs):.2e}",
        f"",
        f"Drugs with FDR < 0.05: {n_sig}/{len(top100_drugs_in_sigs)}",
        "",
        f"{'Drug':<20} {'Net-TRACE':>10} {'Emp p':>10} {'BH FDR':>10} {'Sig?':>6}",
        "-" * 60,
    ]
    for _, row in results.head(20).iterrows():
        sig = "✓" if row["fdr_sig"] else ""
        lines.append(
            f"  {row['drug']:<18} {row['net_trace_cos']:>10.4f} "
            f"{row['emp_pval']:>10.4f} {row['bh_fdr']:>10.4f} {sig:>6}"
        )

    lines += ["", "Positive controls:"]
    for ctrl in ["nintedanib", "pirfenidone"]:
        r = results[results["drug"].str.lower() == ctrl]
        if len(r):
            row = r.iloc[0]
            lines.append(
                f"  {ctrl:<18} emp_p={row['emp_pval']:.4f}  FDR={row['bh_fdr']:.4f}  "
                f"{'FDR-significant' if row['fdr_sig'] else 'not FDR-significant'}"
            )

    report_path = REV / "extended_fdr_report.txt"
    report_path.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nSaved extended_fdr_results.csv, extended_fdr_report.txt")

if __name__ == "__main__":
    main()
