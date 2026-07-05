from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT  = Path(__file__).resolve().parents[2]
BENCH = ROOT / "results/benchmarking"
ACT   = ROOT / "data/known_actives"
OUT   = ROOT / "results/external_validation"
OUT.mkdir(parents=True, exist_ok=True)

PRECLINICAL_EVIDENCE = {
    "imatinib":    "Daniels 2004 JCI; Raghu 2010 ERJ (phase II met 28-wk endpoint); bleomycin attenuation",
    "sildenafil":  "Zisman 2010 Chest; Ryu 2011 Chest (phase II 6MWT improvement); PDE5 inhibition",
    "sorafenib":   "Hostettler 2014 Resp Res; bleomycin model antifibrotic",
    "gefitinib":   "Hardie 2008 AJP; EGFR inhibition reverses TGF-β-induced fibrosis",
    "erlotinib":   "Tsakiri 2008; EGFR pathway antifibrotic in lung fibrosis models",
    "dasatinib":   "Rhee 2014 PLOS ONE; Src/ABL inhibition, bleomycin model",
    "sirolimus":   "Shegogue 2003 J Immunol; mTOR inhibition reduces collagen deposition",
    "roflumilast": "PDE4 inhibition; antifibrotic in rodent bleomycin models",
    "losartan":    "Otsuka 2004 ERJ; AT1R blockade attenuates fibrosis",
    "bosentan":    "Shi-Wen 2007; endothelin-1 antagonism; BUILD trials (clinical) negative",
}

NEGATIVE_EVIDENCE = {
    "azathioprine": "PANTHER NEJM 2012 — stopped early, excess mortality in IPF",
    "acetylcysteine": "PANTHER-IPF NEJM 2014 — no benefit vs placebo",
}

N_PERMUTATIONS = 10_000
PRECISION_AT_K = [25, 50, 100, 200]
SCORE_COL      = "ipf_pearson"


def load_scores() -> pd.DataFrame:
    df = pd.read_csv(BENCH / "dual_disease_scores.csv", index_col="drug")
    df.index = df.index.str.lower().str.strip()
    df = df.sort_values(SCORE_COL, ascending=False).reset_index()
    df["rank"] = range(1, len(df) + 1)
    return df


def match_compounds(df: pd.DataFrame, compounds: list[str]) -> list[str]:
    names = set(df["drug"].tolist())
    matched = []
    for c in compounds:
        if c in names:
            matched.append(c)
        else:
            fuzzy = [n for n in names if c in n or n in c]
            if fuzzy:
                matched.append(fuzzy[0])
    return matched


def bootstrap_auprc(labels: np.ndarray, scores: np.ndarray,
                    n_boot: int = 2000, rng_seed: int = 42) -> tuple[float, float, float]:
    rng = np.random.default_rng(rng_seed)
    n = len(labels)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if labels[idx].sum() == 0:
            continue
        vals.append(average_precision_score(labels[idx], scores[idx]))
    vals = np.array(vals)
    return float(np.median(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def permutation_auroc(labels: np.ndarray, scores: np.ndarray,
                      n_perm: int = N_PERMUTATIONS, rng_seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(rng_seed)
    obs = roc_auc_score(labels, scores)
    null = np.array([
        roc_auc_score(rng.permutation(labels), scores)
        for _ in range(n_perm)
    ])
    pval = float((null >= obs).mean())
    return obs, pval


def precision_at_k(df: pd.DataFrame, active_set: set, k: int) -> float:
    top_k = set(df.head(k)["drug"].tolist())
    hits = top_k & active_set
    return len(hits) / k


def make_figure(df: pd.DataFrame, active_set: set, negative_set: set,
                auroc: float, auprc: float, auprc_lo: float, auprc_hi: float,
                pval: float) -> None:
    labels = df["drug"].isin(active_set).astype(int).values
    neg_labels = df["drug"].isin(negative_set).astype(int).values
    scores = df[SCORE_COL].values
    n_total = len(df)
    n_act = int(labels.sum())

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), dpi=150)
    fig.patch.set_facecolor("white")

    ax = axes[0]
    from sklearn.metrics import roc_curve, precision_recall_curve
    fpr, tpr, _ = roc_curve(labels, scores)
    ax.plot(fpr, tpr, color="#2c7bb6", lw=2, label=f"TRACE (AUROC={auroc:.3f}, p={pval:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4)
    ax.set_xlabel("False positive rate", fontsize=9)
    ax.set_ylabel("True positive rate", fontsize=9)
    ax.set_title("ROC — Preclinical IPF Actives", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, framealpha=0.8)
    ax.set_facecolor("#f8f9fa")

    ax = axes[1]
    prec, rec, _ = precision_recall_curve(labels, scores)
    baseline = n_act / n_total
    ax.plot(rec, prec, color="#d7191c", lw=2,
            label=f"TRACE (AUPRC={auprc:.4f})")
    ax.axhline(baseline, color="gray", lw=1, linestyle="--",
               label=f"Random ({baseline:.4f})")
    ax.fill_between(rec, prec, alpha=0.12, color="#d7191c")
    ax.set_xlabel("Recall", fontsize=9)
    ax.set_ylabel("Precision", fontsize=9)
    ax.set_title(f"PR Curve  [95% CI: {auprc_lo:.4f}–{auprc_hi:.4f}]", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, framealpha=0.8)
    ax.set_facecolor("#f8f9fa")

    ax = axes[2]
    ks = PRECISION_AT_K
    p_at_k = [precision_at_k(df, active_set, k) for k in ks]
    baseline_k = [n_act / n_total] * len(ks)
    x = np.arange(len(ks))
    w = 0.35
    ax.bar(x - w/2, p_at_k, width=w, color="#2c7bb6", label="TRACE", alpha=0.85)
    ax.bar(x + w/2, baseline_k, width=w, color="#aaa", label="Random baseline", alpha=0.70)
    ax.set_xticks(x)
    ax.set_xticklabels([f"P@{k}" for k in ks], fontsize=8.5)
    ax.set_ylabel("Precision", fontsize=9)
    ax.set_title("Precision at K", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, framealpha=0.8)
    ax.set_facecolor("#f8f9fa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#ddd")

    for a in axes:
        for spine in a.spines.values():
            spine.set_edgecolor("#ddd")
        a.tick_params(labelsize=8)

    plt.suptitle(
        f"External Validation — {n_act} Preclinical IPF Actives (independent of training set)",
        fontsize=11, fontweight="bold", y=1.02
    )
    plt.tight_layout(pad=1.2)
    fig.savefig(OUT / "EV1_preclinical_validation.png", dpi=150, bbox_inches="tight",
                facecolor="white")
    fig.savefig(OUT / "EV1_preclinical_validation.svg", bbox_inches="tight", facecolor="white")
    plt.close()


def main() -> None:
    print("Loading TRACE scores ...")
    df = load_scores()
    n_total = len(df)
    print(f"  {n_total} drugs ranked by {SCORE_COL}")

    preclinical_raw = [
        l.strip().lower() for l in
        (ACT / "ipf_preclinical_actives.txt").read_text().splitlines()
        if l.strip()
    ]
    negatives_raw = [
        l.strip().lower() for l in
        (ACT / "ipf_negatives.txt").read_text().splitlines()
        if l.strip()
    ]

    preclinical_matched = match_compounds(df, preclinical_raw)
    negatives_matched   = match_compounds(df, negatives_raw)

    print(f"\nPreclinical actives matched in L1000 ({len(preclinical_matched)}/{len(preclinical_raw)}):")
    for c in preclinical_matched:
        rank = int(df[df["drug"] == c]["rank"].values[0])
        score = float(df[df["drug"] == c][SCORE_COL].values[0])
        ev = PRECLINICAL_EVIDENCE.get(c, "")
        print(f"  {c:<20s}  rank={rank:4d}/{n_total}  score={score:+.4f}  | {ev[:60]}")

    print(f"\nNegative controls matched ({len(negatives_matched)}/{len(negatives_raw)}):")
    for c in negatives_matched:
        rank = int(df[df["drug"] == c]["rank"].values[0])
        score = float(df[df["drug"] == c][SCORE_COL].values[0])
        ev = NEGATIVE_EVIDENCE.get(c, "")
        print(f"  {c:<20s}  rank={rank:4d}/{n_total}  score={score:+.4f}  | {ev[:60]}")

    active_set   = set(preclinical_matched)
    negative_set = set(negatives_matched)

    labels = df["drug"].isin(active_set).astype(int).values
    scores = df[SCORE_COL].values

    if labels.sum() < 2:
        print("Fewer than 2 actives matched — cannot compute AUROC.")
        return

    auroc, pval = permutation_auroc(labels, scores)
    auprc_med, auprc_lo, auprc_hi = bootstrap_auprc(labels, scores)
    baseline = labels.sum() / n_total
    fold = auprc_med / baseline if baseline > 0 else float("nan")

    stat, mwu_p = mannwhitneyu(
        scores[labels == 1], scores[labels == 0], alternative="greater"
    )

    print(f"\n── Enrichment results ───────────────────────────────────")
    print(f"  n actives   : {int(labels.sum())} / {n_total}")
    print(f"  AUROC       : {auroc:.4f}  (permutation p={pval:.4f})")
    print(f"  AUPRC       : {auprc_med:.4f}  [95% CI: {auprc_lo:.4f}–{auprc_hi:.4f}]")
    print(f"  Fold/random : {fold:.2f}×  (random baseline={baseline:.4f})")
    print(f"  Mann-Whitney: U={stat:.0f}, p={mwu_p:.4f}")
    print()
    for k in PRECISION_AT_K:
        pk = precision_at_k(df, active_set, k)
        print(f"  Precision@{k:<4d}: {pk:.4f}  ({pk/baseline:.2f}× random)")

    rows = []
    for c in preclinical_matched:
        r = int(df[df["drug"] == c]["rank"].values[0])
        s = float(df[df["drug"] == c][SCORE_COL].values[0])
        rows.append({"compound": c, "l1000_rank": r, "ipf_pearson": round(s, 5),
                     "percentile": round(100 * (1 - r / n_total), 1),
                     "evidence": PRECLINICAL_EVIDENCE.get(c, "")})
    result_df = pd.DataFrame(rows).sort_values("l1000_rank")
    result_df.to_csv(OUT / "EV1_compound_ranks.csv", index=False)

    summary = {
        "n_actives": int(labels.sum()),
        "n_total": n_total,
        "auroc": round(auroc, 4),
        "auroc_permutation_pval": round(pval, 4),
        "auprc": round(auprc_med, 5),
        "auprc_ci_lo": round(auprc_lo, 5),
        "auprc_ci_hi": round(auprc_hi, 5),
        "auprc_fold_over_random": round(fold, 2),
        "mannwhitney_pval": round(mwu_p, 4),
    }
    pd.DataFrame([summary]).to_csv(OUT / "EV1_summary.csv", index=False)

    print(f"\nGenerating figure ...")
    make_figure(df, active_set, negative_set, auroc, auprc_med, auprc_lo, auprc_hi, pval)
    print(f"Outputs → {OUT}")


if __name__ == "__main__":
    main()
