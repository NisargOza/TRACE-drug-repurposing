"""
Consensus IPF signature via meta-analysis — RESEARCH.md §1c.

Combines per-dataset DE results with inverse-variance-weighted meta-analysis
(fixed + random effects). Keeps only genes that:
  1. Are measured in >= 3 of 4 datasets
  2. Replicate direction in a majority of datasets (sign concordance)
  3. Pass meta-analysis FDR threshold (BH q < 0.05)

Outputs:
  results/meta/consensus_signature.csv  — ranked consensus gene list
  results/meta/replication_stats.csv    — per-gene replication metrics (for report figure)
  results/meta/meta_analysis_summary.png

Usage:
    python src/06_meta_analysis.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DE_DIR   = Path("results/de")
META_DIR = Path("results/meta")
META_DIR.mkdir(exist_ok=True)

DATASETS = ["GSE213001", "GSE150910", "GSE38958", "GSE53845"]
MIN_DATASETS = 3   # gene must appear in >= this many datasets


# ---------------------------------------------------------------------------
# Load and harmonise DE results
# ---------------------------------------------------------------------------

def load_de_results() -> dict[str, pd.DataFrame]:
    results = {}
    for acc in DATASETS:
        f = DE_DIR / f"{acc}_de_entrez.csv"
        df = pd.read_csv(f, index_col=0)
        df.index = df.index.astype(str).str.strip()
        # Ensure required columns exist
        for col in ("log2FoldChange", "pvalue", "padj"):
            if col not in df.columns:
                raise ValueError(f"{acc}: missing column {col}")
        # Convert pvalue = 0 to minimum float to avoid log(0)
        df["pvalue"] = df["pvalue"].replace(0, np.finfo(float).tiny)
        df["pvalue"] = df["pvalue"].clip(lower=np.finfo(float).tiny)
        results[acc] = df[["log2FoldChange", "pvalue", "padj"]].dropna(subset=["log2FoldChange", "pvalue"])
        print(f"  {acc}: {len(df):,} genes, {(df['padj'] < 0.05).sum():,} sig")
    return results


# ---------------------------------------------------------------------------
# Inverse-variance weighted meta-analysis
# ---------------------------------------------------------------------------

def meta_analyse(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    For each gene present in >= MIN_DATASETS datasets, compute:
      - pooled log2FC (inverse-variance weighted mean)
      - meta p-value (Stouffer's z-score method, variance-weighted)
      - direction concordance across datasets
    """
    # Build gene × dataset matrices for LFC and SE
    all_genes = sorted(set().union(*[set(df.index) for df in results.values()]))
    lfc_mat   = pd.DataFrame(index=all_genes, columns=DATASETS, dtype=float)
    se_mat    = pd.DataFrame(index=all_genes, columns=DATASETS, dtype=float)
    pval_mat  = pd.DataFrame(index=all_genes, columns=DATASETS, dtype=float)

    for acc, df in results.items():
        lfc_mat.loc[df.index, acc]  = df["log2FoldChange"].values
        pval_mat.loc[df.index, acc] = df["pvalue"].values
        # SE approximated from p-value and LFC: SE = |LFC| / |z|
        z = np.abs(stats.norm.ppf(df["pvalue"].clip(1e-300, 1 - 1e-10) / 2))
        z = np.where(z == 0, 1e-10, z)
        se_mat.loc[df.index, acc] = (df["log2FoldChange"].abs() / z).values

    # Filter to genes in >= MIN_DATASETS datasets
    n_measured = lfc_mat.notna().sum(axis=1)
    keep = n_measured >= MIN_DATASETS
    lfc_mat  = lfc_mat[keep].astype(float)
    se_mat   = se_mat[keep].astype(float)
    pval_mat = pval_mat[keep].astype(float)
    print(f"\n  Genes in >= {MIN_DATASETS} datasets: {keep.sum():,}")

    # Inverse-variance weights: w = 1 / SE^2
    var_mat = se_mat ** 2
    w_mat   = 1.0 / var_mat  # large SE → small weight

    # Pooled LFC (weighted mean)
    pooled_lfc = (lfc_mat * w_mat).sum(axis=1) / w_mat.sum(axis=1)
    pooled_se  = np.sqrt(1.0 / w_mat.sum(axis=1))
    pooled_z   = pooled_lfc / pooled_se
    meta_pval  = 2 * stats.norm.sf(np.abs(pooled_z))

    # Direction concordance: fraction of datasets where sign matches pooled sign
    expected_sign = np.sign(pooled_lfc)
    sign_mat = np.sign(lfc_mat)
    concordant = (sign_mat == expected_sign.values[:, None]) & lfc_mat.notna()
    n_concordant = concordant.sum(axis=1)
    frac_concordant = n_concordant / n_measured[keep]

    # FDR correction
    reject, padj, _, _ = multipletests(meta_pval, method="fdr_bh")

    result = pd.DataFrame({
        "meta_log2FC":       pooled_lfc,
        "meta_SE":           pooled_se,
        "meta_z":            pooled_z,
        "meta_pvalue":       meta_pval,
        "meta_padj":         padj,
        "n_datasets":        n_measured[keep],
        "n_concordant":      n_concordant,
        "frac_concordant":   frac_concordant,
    })

    # Replication filter: majority concordance (> 50% of measured datasets)
    result["replicated"] = frac_concordant > 0.5

    return result.sort_values("meta_padj")


# ---------------------------------------------------------------------------
# Summary plot
# ---------------------------------------------------------------------------

def plot_summary(meta: pd.DataFrame) -> None:
    sig = meta[meta["meta_padj"] < 0.05]
    rep = meta[(meta["meta_padj"] < 0.05) & meta["replicated"]]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # 1. Volcano plot
    ax = axes[0]
    non_sig = meta[meta["meta_padj"] >= 0.05]
    ax.scatter(non_sig["meta_log2FC"], -np.log10(non_sig["meta_pvalue"]),
               s=4, alpha=0.3, c="#aaaaaa", rasterized=True)
    ax.scatter(sig["meta_log2FC"], -np.log10(sig["meta_pvalue"]),
               s=6, alpha=0.6,
               c=sig["meta_log2FC"].apply(lambda x: "#d62728" if x > 0 else "#1f77b4"),
               rasterized=True)
    ax.axhline(-np.log10(0.05), color="black", lw=0.8, ls="--")
    ax.axvline(0, color="black", lw=0.5)
    ax.set_xlabel("Meta log2 Fold Change (IPF vs control)")
    ax.set_ylabel("-log10(meta p-value)")
    ax.set_title("Meta-analysis volcano")

    # 2. Replication rate by n_datasets
    ax = axes[1]
    for n in sorted(meta["n_datasets"].unique()):
        sub = meta[meta["n_datasets"] == n]
        rep_rate = sub["replicated"].mean() * 100
        ax.bar(n, rep_rate, color="#2ca02c", alpha=0.7)
    ax.set_xlabel("Number of datasets gene is measured in")
    ax.set_ylabel("% direction-concordant genes")
    ax.set_title("Replication rate by dataset coverage")
    ax.set_xticks(sorted(meta["n_datasets"].unique()))

    # 3. LFC distribution: sig replicated vs all
    ax = axes[2]
    rep_sig = meta[(meta["meta_padj"] < 0.05) & meta["replicated"]]
    ax.hist(meta["meta_log2FC"], bins=80, color="#aaaaaa", alpha=0.5, label="All genes")
    ax.hist(rep_sig["meta_log2FC"], bins=40, color="#d62728", alpha=0.7, label="Sig & replicated")
    ax.set_xlabel("Meta log2 Fold Change")
    ax.set_ylabel("Count")
    ax.set_title("LFC distribution")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(META_DIR / "meta_analysis_summary.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading DE results...")
    results = load_de_results()

    print("\nRunning inverse-variance weighted meta-analysis...")
    meta = meta_analyse(results)

    # Save full results
    meta.to_csv(META_DIR / "replication_stats.csv")

    # Consensus signature: FDR < 0.05 AND direction replicated in majority
    consensus = meta[(meta["meta_padj"] < 0.05) & meta["replicated"]].copy()
    consensus = consensus.sort_values("meta_padj")
    consensus.to_csv(META_DIR / "consensus_signature.csv")

    up   = (consensus["meta_log2FC"] > 0).sum()
    down = (consensus["meta_log2FC"] < 0).sum()

    print(f"\n  Total genes tested:          {len(meta):,}")
    print(f"  FDR < 0.05:                  {(meta['meta_padj'] < 0.05).sum():,}")
    print(f"  FDR < 0.05 + replicated:     {len(consensus):,}")
    print(f"    Up in IPF:   {up:,}")
    print(f"    Down in IPF: {down:,}")

    # Replication rate: single-dataset hits vs consensus
    single_hits = sum((df["padj"] < 0.05).sum() for df in results.values())
    print(f"\n  Single-dataset sig genes (sum): {single_hits:,}")
    print(f"  Consensus (replicated):          {len(consensus):,}")
    print(f"  Replication rate:                {len(consensus)/single_hits*100:.1f}%")

    plot_summary(meta)
    print(f"\n  Plots -> results/meta/meta_analysis_summary.png")
    print(f"  Consensus signature -> results/meta/consensus_signature.csv")
    print("\nNext: tissue-aware embedding / network propagation (RESEARCH.md §1d).")


if __name__ == "__main__":
    main()
