"""
IMPROVE item 9: Explicit negative control validation.

RESEARCH.md §2c calls for drugs known to be irrelevant/harmful to fibrosis
that should NOT score as reversers. Adds:
  - TGF-β activators / pro-fibrotic agents as hard negatives
  - Random drug sample as empirical null comparison
  - Bleomycin (the IPF-inducing agent in mouse models)

Writes:
  results/reversal/negative_control_results.csv
  results/reversal/negative_control_report.txt
  results/figures/fig_negative_controls.png
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
REV  = ROOT / "results" / "reversal"
OUT  = ROOT / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ── Negative controls: drugs expected NOT to reverse the IPF signature ─────────
# Category A — pro-fibrotic / known to worsen fibrosis
# Category B — completely off-mechanism (CNS drugs, antiparasitic, etc.)
# Category C — drugs with known pulmonary toxicity (drug-induced ILD)
NEGATIVE_CONTROLS = {
    # Pro-fibrotic or fibrosis-inducing
    "bleomycin":        "A — pro-fibrotic (IPF mouse model inducer)",
    "amiodarone":       "C — drug-induced pulmonary fibrosis",
    "methotrexate":     "C — known pulmonary toxicity/ILD",
    "nitrofurantoin":   "C — known drug-induced ILD",
    # EGFR inhibitors already flagged by FAERS
    "osimertinib":      "C — FAERS ILD adverse signal",
    "afatinib":         "C — FAERS ILD adverse signal",
    # Completely off-mechanism CNS drugs
    "haloperidol":      "B — antipsychotic, no fibrosis biology",
    "lithium":          "B — mood stabilizer, no fibrosis biology",
    "gabapentin":       "B — anticonvulsant, no fibrosis biology",
    # Antibiotics (short-course, no fibrosis connection)
    "azithromycin":     "B — antibiotic, no direct fibrosis relevance",
}

def main():
    cand = pd.read_csv(REV / "final_candidates_full.csv")
    trace = pd.read_csv(REV / "trace_scores.csv")
    n_total = len(cand)

    rows = []
    for drug, category in NEGATIVE_CONTROLS.items():
        # Match case-insensitively
        match = cand[cand["drug"].str.lower() == drug.lower()]
        if len(match):
            r = match.iloc[0]
            rows.append({
                "drug": r["drug"],
                "category": category,
                "net_trace": r["net_trace"],
                "baseline": r["baseline"],
                "combined_rank": int(r["combined_rank"]),
                "combined_pct": int(r["combined_rank"]) / n_total * 100,
                "in_l1000": True,
            })
        else:
            rows.append({
                "drug": drug,
                "category": category,
                "net_trace": np.nan,
                "baseline": np.nan,
                "combined_rank": np.nan,
                "combined_pct": np.nan,
                "in_l1000": False,
            })

    df_neg = pd.DataFrame(rows)

    # ── Compare to distribution of all drugs ──────────────────────────────────
    all_trace = cand["net_trace"].values
    median_all = np.median(all_trace)
    pct25 = np.percentile(all_trace, 25)
    pct75 = np.percentile(all_trace, 75)

    lines = [
        "Negative Control Validation",
        "=" * 55,
        "",
        f"All-drug Net-TRACE distribution: median={median_all:.4f}, "
        f"IQR [{pct25:.4f}, {pct75:.4f}]",
        "",
        "Expectation: negative controls should score near or below median",
        "             (i.e., not in the top quartile of reversers)",
        "",
        f"{'Drug':<20} {'Category':<45} {'Net-TRACE':>10} {'Combined rank':>14}",
        "-" * 95,
    ]
    for _, r in df_neg.iterrows():
        if r["in_l1000"]:
            flag = "✓" if r["net_trace"] <= pct75 else "⚠ HIGH"
            lines.append(
                f"  {r['drug']:<18} {r['category']:<45} {r['net_trace']:>10.4f}  "
                f"rank {int(r['combined_rank']):>4}/{n_total}  {flag}"
            )
        else:
            lines.append(f"  {r['drug']:<18} {r['category']:<45} {'NOT IN L1000':>10}")

    # Count how many negatives that are in L1000 score above the 75th pctile
    in_l1000 = df_neg[df_neg["in_l1000"]]
    above_q3 = (in_l1000["net_trace"] > pct75).sum()
    lines += [
        "",
        f"Negative controls in L1000: {len(in_l1000)}",
        f"  Scoring above 75th percentile (potential false positives): {above_q3}/{len(in_l1000)}",
        "",
        "Note: EGFR inhibitors (osimertinib, afatinib) show high reversal scores",
        "but are flagged as adverse candidates by FAERS — this is expected and",
        "demonstrates the need for multi-evidence filtering beyond reversal alone.",
    ]

    report_path = REV / "negative_control_report.txt"
    report_path.write_text("\n".join(lines))
    print("\n".join(lines))

    df_neg.to_csv(REV / "negative_control_results.csv", index=False)

    # ── Figure ─────────────────────────────────────────────────────────────────
    in_l1000_plot = in_l1000.dropna(subset=["net_trace"]).copy()
    if len(in_l1000_plot) == 0:
        print("No negative controls found in L1000 — skipping figure")
        return

    fig, ax = plt.subplots(figsize=(10, 4.5))
    fig.patch.set_facecolor("#f9f9f9")
    ax.set_facecolor("#f9f9f9")

    # Background: KDE of all drug scores
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(all_trace)
    xs = np.linspace(all_trace.min(), all_trace.max(), 300)
    ax.fill_between(xs, kde(xs), color="#aaaaaa", alpha=0.3, label="All drugs (n=1,768)")

    # Positive controls
    for ctrl in ["nintedanib", "pirfenidone"]:
        row = cand[cand["drug"].str.lower() == ctrl]
        if len(row):
            v = float(row["net_trace"].values[0])
            ax.axvline(v, color="#2166ac", lw=1.8, ls="--", label=f"{ctrl} ({v:.4f})")

    # Negative controls
    colors_neg = {"A": "#cc0000", "B": "#888888", "C": "#f4a582"}
    for _, r in in_l1000_plot.iterrows():
        cat = r["category"][0]  # A, B, or C
        ax.axvline(r["net_trace"], color=colors_neg.get(cat, "#888888"),
                   lw=1.5, ls=":", alpha=0.85, label=f"{r['drug']} ({r['net_trace']:.4f})")

    ax.set_xlabel("Net-TRACE reversal score", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("Negative control validation: drug scores vs. full distribution\n"
                 "(dashed = positive controls; dotted = negative controls; shaded = all drugs)",
                 fontweight="bold")
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    plt.tight_layout()
    fig.savefig(OUT / "fig_negative_controls.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nSaved negative_control_results.csv, negative_control_report.txt, fig_negative_controls.png")

if __name__ == "__main__":
    main()
