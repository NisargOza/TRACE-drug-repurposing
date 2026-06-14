"""
Permutation significance test — Step B6.

For each disease, shuffles gene labels of the network score vector 1,000 times,
recomputes Pearson-based AUROC each time, and reports the empirical p-value.

Usage:
    python src/benchmarking/B6_permutation_test.py
    python src/benchmarking/B6_permutation_test.py --n-perm 1000
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

ROOT        = Path(__file__).resolve().parents[2]
L1K         = ROOT / "results" / "l1000"
EMB         = ROOT / "results" / "embedding"
BENCH       = ROOT / "results" / "benchmarking"
ACTIVES_DIR = ROOT / "data" / "known_actives"
BENCH.mkdir(parents=True, exist_ok=True)

N_PERM = 1000
if "--n-perm" in sys.argv:
    N_PERM = int(sys.argv[sys.argv.index("--n-perm") + 1])


def load_actives(disease: str) -> set[str]:
    path = ACTIVES_DIR / f"{disease.lower()}_actives.txt"
    if not path.exists():
        return set()
    return {l.strip().lower()
            for l in path.read_text().splitlines() if l.strip()}


def pearson_scores(disease_vec: np.ndarray,
                   drug_matrix: np.ndarray) -> np.ndarray:
    """Negative Pearson r per drug column. Higher = stronger reversal."""
    d     = disease_vec - disease_vec.mean()
    d_std = d.std()
    if d_std == 0:
        return np.zeros(drug_matrix.shape[1])
    dm     = drug_matrix - drug_matrix.mean(axis=0)
    dm_std = dm.std(axis=0)
    dm_std[dm_std == 0] = 1.0
    return -(d @ dm) / (d_std * dm_std * len(d))


def run_permutation(disease: str,
                    dis_vec: np.ndarray,
                    drug_mat: np.ndarray,
                    drugs: list[str],
                    actives: set[str]) -> tuple[float, np.ndarray, float]:
    """
    Returns (observed_auroc, null_auroc_array, empirical_pvalue).
    """
    labels = np.array([1 if d.lower() in actives else 0 for d in drugs])
    if labels.sum() < 2:
        print(f"  {disease}: fewer than 2 actives found — skipping")
        return np.nan, np.array([]), np.nan

    obs_scores = pearson_scores(dis_vec, drug_mat)
    obs_auc    = float(roc_auc_score(labels, obs_scores))
    print(f"  {disease}: observed AUROC = {obs_auc:.4f},  "
          f"running {N_PERM:,} permutations ...")

    rng       = np.random.default_rng(seed=42)
    null_aucs = np.zeros(N_PERM)
    for i in range(N_PERM):
        perm        = rng.permutation(len(dis_vec))
        perm_scores = pearson_scores(dis_vec[perm], drug_mat)
        try:
            null_aucs[i] = roc_auc_score(labels, perm_scores)
        except Exception:
            null_aucs[i] = 0.5
        if (i + 1) % 250 == 0:
            print(f"    {i+1:,}/{N_PERM:,}", end="\r", flush=True)
    print(f"    {N_PERM:,}/{N_PERM:,} done    ")

    emp_pval = float((null_aucs >= obs_auc).sum()) / N_PERM
    return obs_auc, null_aucs, emp_pval


def main() -> None:
    print("Loading L1000 drug matrix ...")
    drug_mat_df = pd.read_csv(
        L1K / "drug_signatures_landmark.csv.gz", index_col=0
    )
    drug_mat_df.index = drug_mat_df.index.astype(str)
    drugs = drug_mat_df.columns.tolist()

    diseases = {
        "IPF": EMB / "ipf_network_scores.csv",
        "RA":  EMB / "ra_network_scores.csv",
    }

    records    = []
    null_store = {}

    for disease, net_path in diseases.items():
        if not net_path.exists():
            print(f"WARNING: {net_path.name} missing — skipping {disease}")
            continue

        actives = load_actives(disease)
        if not actives:
            print(f"WARNING: no actives file for {disease}")
            continue

        net = pd.read_csv(net_path, index_col=0)
        net.index = net.index.astype(str)

        common      = net.index.intersection(drug_mat_df.index)
        dis_vec     = net.loc[common, "rwr_net"].values.astype(float)
        dm_aligned  = drug_mat_df.loc[common].values.astype(float)

        obs_auc, null_aucs, emp_pval = run_permutation(
            disease, dis_vec, dm_aligned, drugs, actives
        )

        if np.isnan(obs_auc):
            continue

        null_store[disease] = null_aucs
        records.append({
            "disease":        disease,
            "arm":            "Pearson (network)",
            "observed_auroc": round(obs_auc,  4),
            "emp_pvalue":     round(emp_pval, 4),
            "n_perm":         N_PERM,
            "n_actives":      sum(1 for d in drugs if d.lower() in actives),
            "n_drugs":        len(drugs),
        })

    # Save null distributions
    if null_store:
        np.savez_compressed(
            BENCH / "permutation_null_distributions.npz",
            **{f"{k}_null": v for k, v in null_store.items()}
        )

    pval_df = pd.DataFrame(records)
    pval_df.to_csv(BENCH / "permutation_pvalues.csv", index=False)

    # Plot
    n_plots = len(null_store)
    if n_plots > 0:
        fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
        axes = [axes] if n_plots == 1 else list(axes)
        for ax, (disease, null_aucs) in zip(axes, null_store.items()):
            row = pval_df[pval_df["disease"] == disease]
            obs = float(row["observed_auroc"].iloc[0])
            ax.hist(null_aucs, bins=40, color="#aaaaaa", alpha=0.75,
                    edgecolor="white", label=f"Null ({N_PERM:,} perms)")
            ax.axvline(obs, color="#d62728", lw=2,
                       label=f"Observed={obs:.3f}  p={float(row['emp_pvalue'].iloc[0]):.4f}")
            ax.set_xlabel("AUROC")
            ax.set_ylabel("Count")
            ax.set_title(f"{disease} — Permutation null")
            ax.legend(fontsize=9)
        plt.tight_layout()
        fig.savefig(BENCH / "permutation_null_plot.png", dpi=200)
        plt.close(fig)

    print("\n=== Permutation Test Results ===")
    for _, row in pval_df.iterrows():
        print(f"  {row['disease']:4}  {row['arm']:22} "
              f"AUROC={row['observed_auroc']:.4f}  "
              f"p={row['emp_pvalue']:.4f}  "
              f"({N_PERM:,} permutations)")

    print("\nOutputs → results/benchmarking/")
    print("  permutation_pvalues.csv")
    print("  permutation_null_distributions.npz")
    print("  permutation_null_plot.png")
    print("\nBenchmarking pipeline complete.")


if __name__ == "__main__":
    main()