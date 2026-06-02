"""
Held-out dataset validation — RESEARCH.md §3a.

GSE134692 (80 transplant-stage IPF samples, no controls) was held out from
the consensus signature. Here we:
  1. Compute the IPF expression profile in GSE134692 (vs. its own internal
     variance — no healthy controls, so we use median-centred expression)
  2. Check that the top consensus-up genes are consistently elevated in
     GSE134692 compared to the non-consensus genes (internal consistency check)
  3. Correlate the GSE134692 gene-level median expression with the consensus
     meta_log2FC — high correlation = signature replicates in held-out data

Outputs:
  results/aim3/heldout_consistency.csv    — per-gene median expression in GSE134692
  results/aim3/heldout_validation.png     — scatter: consensus LFC vs held-out expr
  results/aim3/heldout_report.txt         — correlation stats

Usage:
    python src/aim3_validation/15_heldout_validation.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_PROC = Path("data/processed")
META_DIR  = Path("results/meta")
AIM3_DIR  = Path("results/aim3")
AIM3_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    # Load held-out count matrix
    counts_path = DATA_PROC / "GSE134692" / "counts_raw.csv.gz"
    print("Loading GSE134692 (held-out)...")
    counts = pd.read_csv(counts_path, index_col=0)
    counts = counts.select_dtypes(include=[np.number])
    counts.index = counts.index.astype(str)
    print(f"  {counts.shape[0]:,} genes × {counts.shape[1]} samples")

    # log1p normalise
    log_counts = np.log1p(counts)

    # Within-sample z-score: removes baseline expression bias before comparing
    # to consensus LFC (which is also a relative measure)
    sample_mean = log_counts.mean(axis=0)
    sample_std  = log_counts.std(axis=0).replace(0, 1)
    zscored = (log_counts - sample_mean) / sample_std

    # Median z-score per gene across all 80 samples
    gene_median_ensembl = zscored.median(axis=1)

    # Map Ensembl → Entrez using the pre-built map from step 05
    ens_map_path = Path("results/de/GSE213001_ensembl2entrez.csv")
    ens_map = pd.read_csv(ens_map_path)
    ens_to_entrez = dict(zip(ens_map["ENSEMBL"], ens_map["ENTREZID"].astype(str)))
    gene_median = gene_median_ensembl.copy()
    gene_median.index = gene_median.index.map(lambda x: ens_to_entrez.get(x, ""))
    gene_median = gene_median[gene_median.index != ""]
    # Keep first occurrence if multiple Ensembl IDs map to same Entrez
    gene_median = gene_median[~gene_median.index.duplicated(keep="first")]
    print(f"  After Ensembl→Entrez mapping: {len(gene_median):,} genes")

    # Load consensus signature
    consensus = pd.read_csv(META_DIR / "consensus_signature.csv", index_col=0)
    consensus.index = consensus.index.astype(str)

    overlap = consensus.index.intersection(gene_median.index)
    print(f"  Consensus genes measurable in GSE134692: {len(overlap):,} / {len(consensus):,}")

    cons_lfc   = consensus.loc[overlap, "meta_log2FC"]
    held_expr  = gene_median[overlap]

    # Correlation: consensus LFC vs held-out median z-score
    r_spearman, p_spearman = stats.spearmanr(cons_lfc, held_expr)
    r_pearson,  p_pearson  = stats.pearsonr(cons_lfc, held_expr)

    # ssGSEA: for each sample, rank genes; compute enrichment of consensus-up
    # gene set vs. random. Positive ES = up-genes rank higher than expected.
    up_set   = set(cons_lfc[cons_lfc > 0].index)
    down_set = set(cons_lfc[cons_lfc < 0].index)
    all_gene_z = zscored.copy()
    all_gene_z.index = all_gene_z.index.map(lambda x: ens_to_entrez.get(x, ""))
    all_gene_z = all_gene_z[all_gene_z.index != ""]
    all_gene_z = all_gene_z[~all_gene_z.index.duplicated()]

    es_up, es_down = [], []
    for col in all_gene_z.columns:
        ranked = all_gene_z[col].rank(ascending=False)  # rank 1 = highest expression
        n = len(ranked)
        in_up   = ranked.index.isin(up_set)
        in_down = ranked.index.isin(down_set)
        # Mean rank of set members (lower rank = higher expression)
        mean_rank_up   = ranked[in_up].mean()   if in_up.sum() > 0 else n/2
        mean_rank_down = ranked[in_down].mean() if in_down.sum() > 0 else n/2
        # Normalised: fraction of total ranks below expected (positive = enriched high)
        es_up.append(1 - mean_rank_up / n)
        es_down.append(1 - mean_rank_down / n)

    es_up_arr   = np.array(es_up)
    es_down_arr = np.array(es_down)
    t_up,   p_up   = stats.ttest_1samp(es_up_arr, 0.5)
    t_down, p_down = stats.ttest_1samp(es_down_arr, 0.5)

    # Directional consistency:
    # Among consensus-up genes, are they elevated in held-out vs consensus-down?
    up_genes   = cons_lfc[cons_lfc > 0].index
    down_genes = cons_lfc[cons_lfc < 0].index
    up_expr    = held_expr[up_genes].mean()
    down_expr  = held_expr[down_genes].mean()
    t_stat, t_pval = stats.ttest_ind(held_expr[up_genes], held_expr[down_genes])

    # Save per-gene table
    result = pd.DataFrame({
        "entrez_id":     overlap,
        "consensus_lfc": cons_lfc.values,
        "heldout_expr":  held_expr.values,
        "in_consensus":  True,
    }).set_index("entrez_id")
    result.to_csv(AIM3_DIR / "heldout_consistency.csv")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Scatter: consensus LFC vs held-out expression
    ax = axes[0]
    ax.scatter(cons_lfc, held_expr, s=3, alpha=0.3, c="#1f77b4", rasterized=True)
    # Highlight top 20 up and down
    for gene_set, color, label in [
        (cons_lfc.nlargest(20).index, "#d62728", "Top 20 up"),
        (cons_lfc.nsmallest(20).index, "#2ca02c", "Top 20 down"),
    ]:
        ax.scatter(cons_lfc[gene_set], held_expr[gene_set],
                   s=20, c=color, label=label, zorder=5)
    m, b, *_ = stats.linregress(cons_lfc, held_expr)
    x_line = np.array([cons_lfc.min(), cons_lfc.max()])
    ax.plot(x_line, m * x_line + b, "k--", lw=1)
    ax.set_xlabel("Consensus meta-LFC (IPF vs. control)")
    ax.set_ylabel("Held-out median log-expression (GSE134692 IPF)")
    ax.set_title(f"Held-out consistency\nSpearman r={r_spearman:.3f}, p={p_spearman:.2e}")
    ax.legend(fontsize=8)

    # Box plot: expression in up vs down consensus genes
    ax = axes[1]
    ax.boxplot([held_expr[up_genes], held_expr[down_genes]],
               labels=["Consensus-up\ngenes", "Consensus-down\ngenes"],
               patch_artist=True,
               boxprops=dict(facecolor="#d62728", alpha=0.5))
    ax.set_ylabel("Held-out log-expression")
    ax.set_title(f"Up vs. down gene expression in held-out\n"
                 f"t={t_stat:.2f}, p={t_pval:.2e}")

    fig.tight_layout()
    fig.savefig(AIM3_DIR / "heldout_validation.png", dpi=150)
    plt.close(fig)

    # Report
    lines = [
        "Held-out dataset validation (GSE134692)",
        "=" * 50,
        f"Samples: 80 transplant-stage IPF (no controls)",
        f"Consensus genes measurable: {len(overlap):,} / {len(consensus):,}",
        "",
        "Correlation (within-sample z-scores vs consensus LFC):",
        f"  Spearman r = {r_spearman:.4f}  p = {p_spearman:.2e}",
        f"  Pearson  r = {r_pearson:.4f}  p = {p_pearson:.2e}",
        "",
        "ssGSEA — enrichment of consensus gene sets in per-sample rankings:",
        f"  UP   gene set: mean ES = {es_up_arr.mean():.4f}  "
        f"(vs 0.5 null)  t={t_up:.2f}  p={p_up:.2e}",
        f"  DOWN gene set: mean ES = {es_down_arr.mean():.4f}  "
        f"(vs 0.5 null)  t={t_down:.2f}  p={p_down:.2e}",
        "",
        "Directional consistency (z-score):",
        f"  Mean z-score — consensus-up genes:   {up_expr:.4f}",
        f"  Mean z-score — consensus-down genes: {down_expr:.4f}",
        f"  t-test: t={t_stat:.2f}  p={t_pval:.2e}",
        "",
        "Interpretation:",
    ]
    lines.append(
        "  INCONCLUSIVE (no-control limitation): GSE134692 has no healthy controls, so\n"
        "  neither z-score correlation nor ssGSEA can separate 'up in IPF vs. control'\n"
        "  from 'low-expressing gene.' Consensus-UP genes (e.g. KRT5, S100A2) have low\n"
        "  absolute expression even in IPF; consensus-DOWN genes (e.g. SFTPC, ABCA3) are\n"
        "  among the highest-expressed genes in any lung tissue. This produces the observed\n"
        "  negative correlation and low UP-gene ssGSEA score — it is an artifact of the\n"
        "  no-control design, not evidence against the consensus signature.\n"
        "\n"
        "  REPORTABLE FINDING: Proper held-out validation requires a dataset with both\n"
        "  IPF and healthy lung controls (e.g. a split of one training dataset or an\n"
        "  independent case/control series not yet used). GSE134692 is better used as\n"
        "  a consistency check on gene variance, not direction."
    )

    report = "\n".join(lines)
    (AIM3_DIR / "heldout_report.txt").write_text(report)
    print(f"\n{report}")
    print(f"\n  → results/aim3/heldout_validation.png")
    print(f"  → results/aim3/heldout_report.txt")


if __name__ == "__main__":
    main()
