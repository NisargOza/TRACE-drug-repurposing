"""
AUROC benchmark — Step B5.

Computes AUROC (area under ROC curve) with 95% Hanley-McNeil CI for each
scoring arm × disease against the known-actives reference lists.

Hanley-McNeil CI is used (not DeLong) because the number of positives is
small (< 20 known actives).

Outputs:
  results/benchmarking/auroc_summary.csv
  results/benchmarking/roc_curves.png

Usage:
    python src/benchmarking/B5_benchmark_auroc.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

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
    """AUROC with 95% CI via Hanley & McNeil (1982) normal approximation."""
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
    """
    Load all available scoring arms for a disease.
    Returns dict: arm_label -> Series(index=drug_name, values=score)
    Higher score = stronger reversal candidate in all arms.
    """
    arms = {}

    if disease == "IPF":
        # Net-TRACE
        f = REV / "trace_scores.csv"
        if f.exists():
            df = pd.read_csv(f)
            arms["Net-TRACE"] = df.set_index("drug")["trace_score"]

        # VAE-TRACE
        f = EMB / "vae_trace_scores.csv"
        if f.exists():
            df = pd.read_csv(f)
            arms["VAE-TRACE"] = df.set_index("drug")["vae_score"]

        # Baseline KS (negated: baseline_score is already negative for reversals)
        f = REV / "baseline_scores.csv"
        if f.exists():
            df = pd.read_csv(f)
            # baseline_score: more negative = stronger KS reversal
            # negate so higher = better reversal, consistent with other arms
            arms["Baseline-KS"] = -df.set_index("drug")["baseline_score"]

    # Pearson + CMap from B3 (both diseases)
    label = disease.lower()
    f = BENCH / f"{label}_drug_scores.csv"
    if f.exists():
        df = pd.read_csv(f)
        arms[f"Pearson ({disease})"]  = df.set_index("drug")["pearson"]
        # CMap: negative score = reversal → negate for consistent direction
        arms[f"CMap-KS ({disease})"]  = -df.set_index("drug")["cmap"]

    return arms


def main() -> None:
    diseases = ["IPF", "RA"]
    records  = []

    fig, axes = plt.subplots(1, len(diseases),
                              figsize=(7 * len(diseases), 6))
    axes = [axes] if len(diseases) == 1 else list(axes)

    for ax, disease in zip(axes, diseases):
        actives = load_actives(disease)
        print(f"\n{disease}: {len(actives)} known actives")

        arms = load_arms_for_disease(disease)
        if not arms:
            print(f"  No score files found for {disease} — run B3 first")
            continue

        ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Random (AUC=0.50)")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{disease} — ROC Curves")

        for arm_name, scores in arms.items():
            drugs  = scores.dropna().index.tolist()
            labels = np.array([1 if d.lower() in actives else 0
                                for d in drugs])
            n_pos  = labels.sum()
            if n_pos < 2:
                print(f"  {arm_name}: only {n_pos} actives in drug list — skipping")
                continue

            score_vals = scores.loc[drugs].fillna(0).values
            try:
                auc, ci_lo, ci_hi = hanley_mcneil_ci(labels, score_vals)
            except Exception as exc:
                print(f"  {arm_name}: AUROC failed ({exc})")
                continue

            fpr, tpr, _ = roc_curve(labels, score_vals)
            ci_str = (f"[{ci_lo:.3f}–{ci_hi:.3f}]"
                      if not np.isnan(ci_lo) else "")
            ax.plot(fpr, tpr, lw=1.5,
                    label=f"{arm_name}  AUC={auc:.3f} {ci_str}")

            records.append({
                "disease":    disease,
                "arm":        arm_name,
                "auroc":      round(auc,   4),
                "ci_lo_95":   round(ci_lo, 4) if not np.isnan(ci_lo) else np.nan,
                "ci_hi_95":   round(ci_hi, 4) if not np.isnan(ci_hi) else np.nan,
                "n_actives":  int(n_pos),
                "n_total":    len(drugs),
            })

        ax.legend(fontsize=7, loc="lower right")

    plt.tight_layout()
    fig.savefig(BENCH / "roc_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nROC plot → results/benchmarking/roc_curves.png")

    auroc_df = pd.DataFrame(records)
    auroc_df.to_csv(BENCH / "auroc_summary.csv", index=False)

    print("\n=== AUROC Summary ===")
    for _, row in auroc_df.iterrows():
        ci = (f"[{row['ci_lo_95']:.3f}–{row['ci_hi_95']:.3f}]"
              if not np.isnan(row.get("ci_lo_95", np.nan)) else "")
        print(f"  {row['disease']:4} {row['arm']:25} "
              f"AUC={row['auroc']:.3f} {ci}  "
              f"({row['n_actives']} actives / {row['n_total']} drugs)")

    print("\nNext: python src/benchmarking/B6_permutation_test.py --n-perm 1000")


if __name__ == "__main__":
    main()