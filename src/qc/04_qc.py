"""
QC for downloaded IPF datasets — RESEARCH.md §8.

Per-dataset checks:
  1. Library-size distribution (RNA-seq) / expression distribution (array)
  2. PCA coloured by condition — spot batch effects and outliers
  3. Sample-level outlier flagging (> 3 SD from mean PC1/PC2)

Outputs per dataset:
  results/qc/{acc}_libsize.png
  results/qc/{acc}_pca.png
  results/qc/qc_summary.csv   — outlier flags for all samples

Usage:
    python src/04_qc.py
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore", category=RuntimeWarning)

DATA_PROC = Path("data/processed")
QC_DIR = Path("results/qc")
QC_DIR.mkdir(parents=True, exist_ok=True)

CONDITION_COLORS = {
    "IPF":     "#d62728",
    "control": "#1f77b4",
    "CHP":     "#ff7f0e",
    "other":   "#aaaaaa",
}

DATASETS = {
    "GSE213001": "rnaseq",
    "GSE150910": "rnaseq",
    "GSE38958":  "array",
    "GSE134692": "rnaseq",
    "GSE53845":  "array",
}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_expression(acc: str, dtype: str) -> pd.DataFrame:
    """Return expression DataFrame (genes/probes × samples), log1p-transformed."""
    proc = DATA_PROC / acc
    if dtype == "rnaseq":
        f = proc / "counts_raw.csv.gz"
        if not f.exists():
            f = proc / "expression_rnaseq.csv.gz"
    else:
        f = proc / "expression_array.csv.gz"

    if not f.exists():
        raise FileNotFoundError(f"No expression file for {acc}: {f}")

    df = pd.read_csv(f, index_col=0, low_memory=False)
    # Keep only numeric columns (some datasets include gene-symbol annotation cols)
    df = df.select_dtypes(include=[np.number])
    # log1p-transform counts; arrays are already log-scale but log1p is idempotent
    # for values >> 0
    if dtype == "rnaseq":
        df = np.log1p(df)
    return df


def load_metadata(acc: str, expr_cols: pd.Index | None = None) -> pd.DataFrame:
    """
    Load sample metadata. If expr_cols are provided and don't match the
    metadata GSM index, attempt to re-index via Sample_title (short IDs
    used in some supplementary count files).
    """
    f = DATA_PROC / acc / "metadata.csv"
    meta = pd.read_csv(f, index_col=0, low_memory=False)

    if expr_cols is None:
        return meta

    # Check if expr columns already align with metadata index
    if set(expr_cols) & set(meta.index):
        return meta

    # Try aligning via Sample_title (short IDs like 'chp_1', 'ipf_23')
    title_col = next((c for c in meta.columns if "title" in c.lower()), None)
    if title_col is not None:
        title_to_gsm = dict(zip(meta[title_col], meta.index))
        new_index = [title_to_gsm.get(c, c) for c in expr_cols]
        meta_reindexed = meta.reindex(new_index)
        meta_reindexed.index = expr_cols
        return meta_reindexed

    return meta


# ---------------------------------------------------------------------------
# QC plots
# ---------------------------------------------------------------------------

def plot_libsize(expr: pd.DataFrame, meta: pd.DataFrame,
                 acc: str, dtype: str) -> None:
    """Barplot of per-sample total counts (RNA-seq) or median expression (array)."""
    if dtype == "rnaseq":
        sizes = np.expm1(expr).sum(axis=0)  # back to raw counts for interpretability
        ylabel = "Total counts"
    else:
        sizes = expr.median(axis=0)
        ylabel = "Median expression"

    conditions = meta.reindex(sizes.index).get("condition", pd.Series("other", index=sizes.index))
    colors = [CONDITION_COLORS.get(c, "#aaaaaa") for c in conditions]

    order = sizes.sort_values().index
    fig, ax = plt.subplots(figsize=(max(6, len(sizes) * 0.06), 4))
    ax.bar(range(len(order)), sizes[order], color=[CONDITION_COLORS.get(conditions.get(s, "other"), "#aaa") for s in order], width=1.0)
    ax.set_xlabel("Samples (sorted)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{acc} — {ylabel} per sample")
    ax.set_xticks([])
    patches = [mpatches.Patch(color=v, label=k) for k, v in CONDITION_COLORS.items()
               if k in conditions.values]
    ax.legend(handles=patches, fontsize=8)
    fig.tight_layout()
    fig.savefig(QC_DIR / f"{acc}_libsize.png", dpi=150)
    plt.close(fig)


def plot_pca(expr: pd.DataFrame, meta: pd.DataFrame, acc: str) -> pd.DataFrame:
    """
    PCA on top-5000 variable genes/probes.
    Returns DataFrame with PC1, PC2, outlier flag per sample.
    """
    # Top variable features
    var = expr.var(axis=1)
    top = var.nlargest(min(5000, len(var))).index
    X = expr.loc[top].T  # samples × features

    # Standardise and PCA
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=min(10, X.shape[0] - 1), random_state=42)
    coords = pca.fit_transform(Xs)
    var_exp = pca.explained_variance_ratio_ * 100

    pc_df = pd.DataFrame(
        coords[:, :2], index=X.index, columns=["PC1", "PC2"]
    )

    conditions = meta.reindex(pc_df.index).get(
        "condition", pd.Series("other", index=pc_df.index)
    )

    # Outlier flag: > 3 SD on PC1 or PC2
    for pc in ["PC1", "PC2"]:
        mu, sd = pc_df[pc].mean(), pc_df[pc].std()
        pc_df[f"{pc}_zscore"] = (pc_df[pc] - mu) / sd
    pc_df["outlier"] = (pc_df["PC1_zscore"].abs() > 3) | (pc_df["PC2_zscore"].abs() > 3)
    pc_df["condition"] = conditions.values

    # Plot
    fig, ax = plt.subplots(figsize=(7, 5))
    for cond, grp in pc_df.groupby("condition"):
        color = CONDITION_COLORS.get(cond, "#aaaaaa")
        ax.scatter(grp["PC1"], grp["PC2"], c=color, label=cond,
                   alpha=0.75, s=25, edgecolors="none")
    # Mark outliers
    out = pc_df[pc_df["outlier"]]
    if not out.empty:
        ax.scatter(out["PC1"], out["PC2"], s=80, facecolors="none",
                   edgecolors="black", linewidths=1.5, zorder=5, label="outlier")
        for idx, row in out.iterrows():
            ax.annotate(str(idx)[:10], (row["PC1"], row["PC2"]),
                        fontsize=6, ha="left", va="bottom")

    ax.set_xlabel(f"PC1 ({var_exp[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({var_exp[1]:.1f}%)")
    ax.set_title(f"{acc} — PCA (top-5000 variable features)")
    ax.legend(fontsize=8, markerscale=1.5)
    fig.tight_layout()
    fig.savefig(QC_DIR / f"{acc}_pca.png", dpi=150)
    plt.close(fig)

    return pc_df[["PC1", "PC2", "PC1_zscore", "PC2_zscore", "outlier", "condition"]]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    all_flags = []

    for acc, dtype in DATASETS.items():
        print(f"\n{'─'*50}")
        print(f"{acc}  [{dtype}]")
        try:
            expr = load_expression(acc, dtype)
            meta = load_metadata(acc, expr_cols=expr.columns)
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
            continue

        print(f"  Expression: {expr.shape[0]} features × {expr.shape[1]} samples")

        plot_libsize(expr, meta, acc, dtype)
        print(f"  Library-size plot → results/qc/{acc}_libsize.png")

        pc_df = plot_pca(expr, meta, acc)
        print(f"  PCA plot         → results/qc/{acc}_pca.png")

        n_out = pc_df["outlier"].sum()
        if n_out:
            outlier_ids = pc_df[pc_df["outlier"]].index.tolist()
            print(f"  Outliers flagged (>3 SD on PC1/PC2): {n_out}")
            for s in outlier_ids:
                row = pc_df.loc[s]
                print(f"    {s}  PC1_z={row['PC1_zscore']:.2f}  PC2_z={row['PC2_zscore']:.2f}  cond={row['condition']}")
        else:
            print(f"  No outliers flagged")

        pc_df["dataset"] = acc
        all_flags.append(pc_df.reset_index().rename(columns={"index": "sample_id",
                                                               "Sample_geo_accession": "sample_id"}))

    if all_flags:
        summary = pd.concat(all_flags, ignore_index=True)
        summary.to_csv(QC_DIR / "qc_summary.csv", index=False)
        print(f"\nQC summary → results/qc/qc_summary.csv")

    print("\nNext: review PCA plots for batch effects, then run differential")
    print("expression per dataset (limma for arrays, DESeq2 for RNA-seq).")


if __name__ == "__main__":
    main()
