from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    average_precision_score, precision_recall_curve,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT  = Path(__file__).resolve().parents[2]
L1K   = ROOT / "results/l1000"
BENCH = ROOT / "results/benchmarking"
EMB   = ROOT / "results/embedding"
META  = ROOT / "results/meta"
ACT   = ROOT / "data/known_actives"
OUT   = ROOT / "results/external_validation"
OUT.mkdir(parents=True, exist_ok=True)

N_PERM      = 10_000
N_BOOT      = 2_000
PRECISION_K = [25, 50, 100, 200]
RNG_SEED    = 42

RGES_K = 150


def _ks_enrichment(ranked_scores: np.ndarray, gene_set_idx: np.ndarray) -> float:
    n = len(ranked_scores)
    k = len(gene_set_idx)
    if k == 0:
        return 0.0
    hits = np.zeros(n, dtype=float)
    hits[gene_set_idx] = 1.0
    sorted_hits = hits[np.argsort(ranked_scores)[::-1]]
    running = np.cumsum(sorted_hits) / k - (np.arange(1, n + 1) - np.cumsum(sorted_hits)) / (n - k + 1e-9)
    return float(running[np.argmax(np.abs(running))])


def rges_single(disease_up_idx: np.ndarray, disease_dn_idx: np.ndarray,
                drug_col: np.ndarray) -> float:
    es_up = _ks_enrichment(drug_col, disease_up_idx)
    es_dn = _ks_enrichment(drug_col, disease_dn_idx)
    if np.sign(es_up) != np.sign(es_dn):
        return -(es_up - es_dn) / 2.0
    return 0.0


def cmap_single(disease_up_idx: np.ndarray, disease_dn_idx: np.ndarray,
                drug_col: np.ndarray) -> float:
    es_up = _ks_enrichment(drug_col, disease_up_idx)
    es_dn = _ks_enrichment(drug_col, disease_dn_idx)
    if np.sign(es_up) != np.sign(es_dn):
        return -(es_up - es_dn) / 2.0
    return 0.0


def compute_published_scores(drug_mat: pd.DataFrame,
                              cons: pd.DataFrame) -> pd.DataFrame:
    common = drug_mat.index.intersection(cons.index)
    print(f"  Landmark ∩ disease signature: {len(common)} genes")

    cons_c = cons.loc[common].copy()
    cons_c["rank_score"] = (
        -np.log10(cons_c["meta_padj"].clip(1e-300))
        * np.sign(cons_c["meta_log2FC"])
    )
    ranked = cons_c.sort_values("rank_score", ascending=False)

    up_genes   = ranked[ranked["rank_score"] > 0].head(RGES_K).index
    down_genes = ranked[ranked["rank_score"] < 0].tail(RGES_K).index
    print(f"  RGES gene sets: {len(up_genes)} up + {len(down_genes)} down "
          f"(K={RGES_K} per direction, Shen et al. 2017)")

    gene_order  = list(common)
    up_idx   = np.array([gene_order.index(g) for g in up_genes   if g in gene_order])
    down_idx = np.array([gene_order.index(g) for g in down_genes if g in gene_order])

    dm = drug_mat.loc[gene_order].values.astype(float)
    nd = dm.shape[1]
    drugs = drug_mat.columns.tolist()

    print(f"  Computing RGES ({nd} drugs) ...")
    rges_scores = np.array([rges_single(up_idx, down_idx, dm[:, j]) for j in range(nd)])

    return pd.DataFrame({
        "drug":  drugs,
        "rges":  rges_scores,
    }).set_index("drug")


def load_trace_scores() -> pd.DataFrame:
    df = pd.read_csv(BENCH / "dual_disease_scores.csv", index_col="drug")
    df.index = df.index.str.lower().str.strip()
    return df[["ipf_pearson"]].rename(columns={"ipf_pearson": "trace_pearson"})


def load_vae_scores() -> pd.Series:
    vts = pd.read_csv(EMB / "vae_trace_scores.csv")
    vts["drug"] = vts["drug"].str.lower().str.strip()
    return vts.set_index("drug")["vae_score"]


def compute_ablation_scores(drug_mat: pd.DataFrame, cons: pd.DataFrame) -> pd.DataFrame:
    common = drug_mat.index.intersection(cons.index)
    lfc    = cons.loc[common, "meta_log2FC"].values.astype(float)
    dm     = drug_mat.loc[common].values.astype(float)
    drugs  = drug_mat.columns.tolist()
    n      = len(common)

    lfc_z = (lfc - lfc.mean()) / (lfc.std() + 1e-8)
    dm_z  = (dm - dm.mean(axis=0)) / (dm.std(axis=0) + 1e-8)
    pearson_raw   = -(lfc_z @ dm_z) / n

    lfc_r = stats.rankdata(lfc)
    dm_r  = np.apply_along_axis(stats.rankdata, 0, dm)
    lfc_rz = (lfc_r - lfc_r.mean()) / (lfc_r.std() + 1e-8)
    dm_rz  = (dm_r - dm_r.mean(axis=0)) / (dm_r.std(axis=0) + 1e-8)
    spearman_raw  = -(lfc_rz @ dm_rz) / n

    return pd.DataFrame({
        "drug":         drugs,
        "pearson_raw":  pearson_raw,
        "spearman_raw": spearman_raw,
    }).set_index("drug")


def permutation_auroc(labels: np.ndarray, scores: np.ndarray,
                      n_perm: int = N_PERM, seed: int = RNG_SEED) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    obs = roc_auc_score(labels, scores)
    null = np.array([roc_auc_score(rng.permutation(labels), scores) for _ in range(n_perm)])
    return obs, float((null >= obs).mean())


def bootstrap_auprc(labels: np.ndarray, scores: np.ndarray,
                    n_boot: int = N_BOOT, seed: int = RNG_SEED) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n   = len(labels)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if labels[idx].sum() == 0:
            continue
        vals.append(average_precision_score(labels[idx], scores[idx]))
    v = np.array(vals)
    return float(np.median(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def precision_at_k(ranked: list[str], active_set: set, k: int) -> float:
    return len(set(ranked[:k]) & active_set) / k


def evaluate_method(name: str, scores: pd.Series, active_set: set,
                    all_drugs: list[str]) -> tuple[dict, np.ndarray, np.ndarray,
                                                    np.ndarray, np.ndarray]:
    s = scores.reindex(all_drugs).dropna()
    labels = s.index.isin(active_set).astype(int)
    sv     = s.values

    if labels.sum() < 2:
        print(f"  {name}: fewer than 2 actives matched — skip")
        return {}, None, None, None, None

    auroc, pval           = permutation_auroc(labels, sv)
    auprc, auprc_lo, auprc_hi = bootstrap_auprc(labels, sv)
    baseline              = labels.sum() / len(labels)
    ranked_drugs          = s.sort_values(ascending=False).index.tolist()
    pk = {k: precision_at_k(ranked_drugs, active_set, k) for k in PRECISION_K}

    fpr,  tpr,  _ = roc_curve(labels, sv)
    prec, rec,  _ = precision_recall_curve(labels, sv)

    row = {
        "method":     name,
        "auroc":      round(auroc, 4),
        "auroc_p":    round(pval, 4),
        "auprc":      round(auprc, 5),
        "auprc_lo":   round(auprc_lo, 5),
        "auprc_hi":   round(auprc_hi, 5),
        "auprc_fold": round(auprc / baseline, 2),
    }
    row.update({f"p_at_{k}": round(pk[k], 4) for k in PRECISION_K})

    sig_str = ("***" if pval < 0.001 else
               "**"  if pval < 0.01  else
               "*"   if pval < 0.05  else "ns")
    print(f"  {name:<30s}  AUROC={auroc:.4f} ({sig_str}, p={pval:.3f})  "
          f"AUPRC={auprc:.5f} ({auprc / baseline:.2f}× random)")
    return row, fpr, tpr, prec, rec


PALETTE = {
    "RGES (Shen et al. 2017)":    "#4d4d4d",
    "TRACE-Pearson (ours)":        "#2166ac",
    "TRACE-VAE (ours)":            "#d7191c",
}


def make_figure(results: list[dict],
                roc_curves: dict, pr_curves: dict,
                baseline_auprc: float, n_actives: int, n_drugs: int) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=150)
    fig.patch.set_facecolor("white")

    ax = axes[0]
    for row in results:
        nm  = row["method"]
        fpr, tpr = roc_curves[nm]
        col = PALETTE.get(nm, "#888")
        ls  = "--" if "ours" not in nm else "-"
        ax.plot(fpr, tpr, lw=2, color=col, linestyle=ls,
                label=f"{nm} (AUROC={row['auroc']:.3f}, p={row['auroc_p']:.3f})")
    ax.plot([0, 1], [0, 1], color="#bbb", lw=1, linestyle=":")
    ax.set_xlabel("False positive rate", fontsize=9)
    ax.set_ylabel("True positive rate", fontsize=9)
    ax.set_title("ROC — IPF preclinical actives", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, framealpha=0.9, loc="lower right")
    ax.set_facecolor("#f8f9fa")
    for sp in ax.spines.values():
        sp.set_edgecolor("#ddd")

    ax = axes[1]
    for row in results:
        nm   = row["method"]
        p, r = pr_curves[nm]
        col  = PALETTE.get(nm, "#888")
        ls   = "--" if "ours" not in nm else "-"
        ax.plot(r, p, lw=2, color=col, linestyle=ls,
                label=f"{nm} ({row['auprc_fold']:.2f}× random)")
    ax.axhline(baseline_auprc, color="#aaa", lw=1, linestyle=":",
               label=f"Random ({baseline_auprc:.5f})")
    ax.set_xlabel("Recall", fontsize=9)
    ax.set_ylabel("Precision", fontsize=9)
    ax.set_title("Precision-recall — AUPRC fold over random", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, framealpha=0.9, loc="upper right")
    ax.set_facecolor("#f8f9fa")
    for sp in ax.spines.values():
        sp.set_edgecolor("#ddd")

    ax = axes[2]
    x = np.arange(len(PRECISION_K))
    n_m = len(results)
    w   = 0.7 / n_m
    for i, row in enumerate(results):
        nm   = row["method"]
        pks  = [row.get(f"p_at_{k}", 0) for k in PRECISION_K]
        col  = PALETTE.get(nm, "#888")
        off  = (i - n_m / 2 + 0.5) * w
        ax.bar(x + off, pks, width=w, color=col, alpha=0.85, label=nm)
    random_p = n_actives / n_drugs
    ax.axhline(random_p, color="#aaa", lw=1.2, linestyle=":", label=f"Random ({random_p:.4f})")
    ax.set_xticks(x)
    ax.set_xticklabels([f"P@{k}" for k in PRECISION_K], fontsize=8.5)
    ax.set_ylabel("Precision", fontsize=9)
    ax.set_title("Precision @ K", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7.5, framealpha=0.9)
    ax.set_facecolor("#f8f9fa")
    for sp in ax.spines.values():
        sp.set_edgecolor("#ddd")

    for a in axes:
        a.tick_params(labelsize=8)

    plt.suptitle(
        f"SOTA Comparison — TRACE vs published transcriptomic repurposing methods\n"
        f"(IPF preclinical actives, n={n_actives}/{n_drugs} L1000 drugs)",
        fontsize=11, fontweight="bold", y=1.03,
    )
    plt.tight_layout(pad=1.2)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"EV3_sota_comparison.{ext}",
                    dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Figure → results/external_validation/EV3_sota_comparison.png")


def main() -> None:
    actives_raw = [
        l.strip().lower() for l in
        (ACT / "ipf_preclinical_actives.txt").read_text().splitlines()
        if l.strip()
    ]

    print("Loading drug matrix ...")
    drug_mat = pd.read_csv(L1K / "drug_signatures_landmark.csv.gz", index_col=0)
    drug_mat.index   = drug_mat.index.astype(str)
    drug_mat.columns = drug_mat.columns.str.lower().str.strip()
    all_drugs = drug_mat.columns.tolist()

    print("Loading consensus signature ...")
    cons = pd.read_csv(META / "consensus_signature.csv", index_col=0)
    cons.index = cons.index.astype(str)

    active_set      = set(a for a in actives_raw if a in all_drugs)
    baseline_auprc  = len(active_set) / len(all_drugs)
    print(f"Actives matched in L1000: {sorted(active_set)} ({len(active_set)}/{len(actives_raw)})")

    print("\nComputing published baseline scores ...")
    published = compute_published_scores(drug_mat, cons)

    print("\nLoading TRACE scores ...")
    trace = load_trace_scores()
    vae   = load_vae_scores()

    main_methods = {
        "RGES (Shen et al. 2017)": published["rges"],
        "TRACE-Pearson (ours)":    trace["trace_pearson"],
        "TRACE-VAE (ours)":        vae,
    }

    print("\n── SOTA comparison ─────────────────────────────────────────────")
    results, roc_curves, pr_curves = [], {}, {}
    for name, scores in main_methods.items():
        row, fpr, tpr, prec, rec = evaluate_method(name, scores, active_set, all_drugs)
        if row:
            results.append(row)
            roc_curves[name] = (fpr, tpr)
            pr_curves[name]  = (prec, rec)

    summary_df = pd.DataFrame(results).sort_values("auroc", ascending=False)
    print("\n── Ranked summary ──────────────────────────────────────────────")
    print(summary_df[["method", "auroc", "auroc_p", "auprc", "auprc_fold",
                       "p_at_25", "p_at_50", "p_at_100"]].to_string(index=False))
    summary_df.to_csv(OUT / "EV3_sota_comparison.csv", index=False)

    print("\nComputing ablation supplement ...")
    ablation = compute_ablation_scores(drug_mat, cons)
    abl_methods = {
        "Pearson-raw (no network)":  ablation["pearson_raw"],
        "Spearman-raw (no network)": ablation["spearman_raw"],
        "TRACE-Pearson (ours)":      trace["trace_pearson"],
        "TRACE-VAE (ours)":          vae,
    }
    abl_rows = []
    for name, scores in abl_methods.items():
        row, *_ = evaluate_method(name, scores, active_set, all_drugs)
        if row:
            abl_rows.append(row)
    pd.DataFrame(abl_rows).to_csv(OUT / "EV3_ablation_supplement.csv", index=False)
    print("  Ablation table → results/external_validation/EV3_ablation_supplement.csv")

    print("\nGenerating figure ...")
    make_figure(results, roc_curves, pr_curves, baseline_auprc,
                len(active_set), len(all_drugs))
    print(f"\nEV3 complete → {OUT}")


if __name__ == "__main__":
    main()
