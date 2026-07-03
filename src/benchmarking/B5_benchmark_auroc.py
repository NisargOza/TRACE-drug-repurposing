
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score

ROOT        = Path(__file__).resolve().parents[2]
EMB         = ROOT / "results" / "embedding"
REV         = ROOT / "results" / "reversal"
BENCH       = ROOT / "results" / "benchmarking"
ACTIVES_DIR = ROOT / "data" / "known_actives"
BENCH.mkdir(parents=True, exist_ok=True)


def load_actives(disease: str) -> set[str]:
    path = ACTIVES_DIR / f"{disease.lower()}_actives.txt"
    if not path.exists():
        return set()
    return {l.strip().lower()
            for l in path.read_text().splitlines() if l.strip()}


def hanley_mcneil_ci(y_true: np.ndarray,
                     y_score: np.ndarray,
                     alpha: float = 0.05) -> tuple[float, float, float]:
    n1  = int(y_true.sum())
    n2  = int((1 - y_true).sum())
    if n1 < 2 or n2 < 2:
        auc = float(roc_auc_score(y_true, y_score)) if n1 >= 1 else 0.5
        return auc, np.nan, np.nan
    auc = float(roc_auc_score(y_true, y_score))
    q1  = auc / (2 - auc)
    q2  = 2 * auc ** 2 / (1 + auc)
    var = (auc * (1 - auc)
           + (n1 - 1) * (q1 - auc ** 2)
           + (n2 - 1) * (q2 - auc ** 2)) / (n1 * n2)
    se  = np.sqrt(max(var, 0.0))
    z   = 1.96
    return auc, max(0.0, auc - z * se), min(1.0, auc + z * se)


def load_arms_for_disease(disease: str) -> dict[str, pd.Series]:
    arms = {}

    if disease == "IPF":
        f = REV / "trace_scores.csv"
        if f.exists():
            df = pd.read_csv(f)
            arms["Net-TRACE"] = df.set_index("drug")["trace_score"]

        f = EMB / "vae_trace_scores.csv"
        if f.exists():
            df = pd.read_csv(f)
            arms["VAE-TRACE"] = df.set_index("drug")["vae_score"]

        f = REV / "baseline_scores.csv"
        if f.exists():
            df = pd.read_csv(f)
            arms["Baseline-KS"] = -df.set_index("drug")["baseline_score"]

    label = disease.lower()
    f = BENCH / f"{label}_drug_scores.csv"
    if f.exists():
        df = pd.read_csv(f)
        arms[f"Pearson ({disease})"]  = df.set_index("drug")["pearson"]
        arms[f"CMap-KS ({disease})"]  = -df.set_index("drug")["cmap"]

    return arms


def main() -> None:
    diseases = ["IPF", "RA"]
    records  = []

    fig, all_axes = plt.subplots(len(diseases), 2,
                                  figsize=(14, 6 * len(diseases)))

    for d_idx, disease in enumerate(diseases):
        ax_roc, ax_pr = all_axes[d_idx]
        actives = load_actives(disease)
        print(f"\n{disease}: {len(actives)} known actives (from file)")

        arms = load_arms_for_disease(disease)
        if not arms:
            print(f"  No score files found for {disease} — run B3 first")
            continue

        ax_roc.plot([0, 1], [0, 1], "k--", lw=0.8, label="Random (AUC=0.50)")
        ax_roc.set_xlabel("False Positive Rate"); ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title(f"{disease} — ROC Curves")
        ax_pr.set_xlabel("Recall"); ax_pr.set_ylabel("Precision")
        ax_pr.set_title(f"{disease} — Precision-Recall Curves")

        for arm_name, scores in arms.items():
            drugs  = scores.dropna().index.tolist()
            labels = np.array([1 if d.lower() in actives else 0 for d in drugs])
            n_pos  = int(labels.sum())
            if n_pos < 2:
                print(f"  {arm_name}: only {n_pos} actives in drug list — skipping")
                continue

            score_vals = scores.loc[drugs].fillna(0).values
            try:
                auroc, ci_lo, ci_hi = hanley_mcneil_ci(labels, score_vals)
            except Exception as exc:
                print(f"  {arm_name}: AUROC failed ({exc})")
                continue

            ap         = float(average_precision_score(labels, score_vals))
            rand_base  = n_pos / len(drugs)

            fpr, tpr, _ = roc_curve(labels, score_vals)
            prec, rec, _ = precision_recall_curve(labels, score_vals)
            ci_str = f"[{ci_lo:.3f}–{ci_hi:.3f}]" if not np.isnan(ci_lo) else ""
            ax_roc.plot(fpr, tpr, lw=1.5, label=f"{arm_name}  AUC={auroc:.3f} {ci_str}")
            ax_pr.plot(rec, prec, lw=1.5, label=f"{arm_name}  AP={ap:.4f} ({ap/rand_base:.1f}× random)")

            records.append({
                "disease":                disease,
                "arm":                    arm_name,
                "auroc":                  round(auroc, 4),
                "ci_lo_95":               round(ci_lo, 4) if not np.isnan(ci_lo) else np.nan,
                "ci_hi_95":               round(ci_hi, 4) if not np.isnan(ci_hi) else np.nan,
                "auprc":                  round(ap, 6),
                "auprc_random_baseline":  round(rand_base, 6),
                "auprc_fold_over_random": round(ap / rand_base, 2) if rand_base > 0 else np.nan,
                "n_actives":              n_pos,
                "n_total":                len(drugs),
            })

        n_all = len(next(iter(arms.values())))
        n_act = sum(1 for d in next(iter(arms.values())).index if d.lower() in actives)
        ax_pr.axhline(n_act / n_all, ls="--", lw=0.8, color="k",
                      label=f"random ({n_act/n_all:.5f})")
        ax_roc.legend(fontsize=6.5, loc="lower right")
        ax_pr.legend(fontsize=6.5, loc="upper right")

    plt.suptitle("TRACE Benchmark — ROC and Precision-Recall (B5)", fontsize=12)
    plt.tight_layout()
    fig.savefig(BENCH / "roc_pr_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot → results/benchmarking/roc_pr_curves.png")

    auroc_df = pd.DataFrame(records)
    auroc_df.to_csv(BENCH / "auroc_summary.csv", index=False)

    print("\n=== AUROC + AUPRC Summary ===")
    for _, row in auroc_df.iterrows():
        ci = (f"[{row['ci_lo_95']:.3f}–{row['ci_hi_95']:.3f}]"
              if not np.isnan(row.get("ci_lo_95", np.nan)) else "")
        print(f"  {row['disease']:4} {row['arm']:25} "
              f"AUROC={row['auroc']:.3f} {ci}  "
              f"AUPRC={row['auprc']:.5f} ({row['auprc_fold_over_random']:.1f}× random)  "
              f"({row['n_actives']} actives / {row['n_total']} drugs)")

    print("\nNext: python src/benchmarking/B6_permutation_test.py --n-perm 1000")


if __name__ == "__main__":
    main()