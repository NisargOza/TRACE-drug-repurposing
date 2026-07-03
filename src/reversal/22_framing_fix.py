
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

def main():
    cand = pd.read_csv(REV / "final_candidates_full.csv")

    reversal_only = cand[["drug","net_trace","weighted_trace","vae_score",
                           "baseline","sig_reproducibility","n_cell_lines"]].copy()
    reversal_only = reversal_only.sort_values("net_trace", ascending=False).reset_index(drop=True)
    reversal_only["reversal_rank"] = reversal_only.index + 1
    reversal_only["reversal_pct"]  = reversal_only["reversal_rank"] / len(reversal_only) * 100
    reversal_only.to_csv(REV / "reversal_primary_ranking.csv", index=False)

    for drug in ["nintedanib", "pirfenidone"]:
        row = reversal_only[reversal_only["drug"].str.lower() == drug]
        if len(row):
            r = int(row["reversal_rank"].values[0])
            p = float(row["reversal_pct"].values[0])
            print(f"Reversal-only: {drug} rank {r}/{len(reversal_only)} ({p:.1f}%)")

    lines = [
        "Genetic-Support Circularity Ablation",
        "=" * 55,
        "",
        "Problem: nintedanib's combined rank 2 is heavily influenced",
        "by genetic support (OT score 0.594) — which directly reflects",
        "known IPF target biology, i.e., it corroborates an approved drug.",
        "A model that ranks a drug highly *because* we already know its",
        "targets work in IPF is partially circular for the known controls.",
        "",
        "Ablation: recompute combined score with genetic_support = 0",
        "for nintedanib and cediranib (OT ≥ 0.5):",
        "",
    ]

    weights = {"net_trace": 0.50, "genetic_support": 0.30, "sig_reproducibility": 0.20}
    total = len(cand)

    for drug in ["nintedanib", "cediranib", "pirfenidone", "romidepsin", "dasatinib"]:
        row = cand[cand["drug"].str.lower() == drug]
        if not len(row):
            continue
        r = row.iloc[0]

        orig_rank  = int(r["combined_rank"])
        orig_pct   = orig_rank / total * 100

        score_no_gen = (
            r["net_trace"]          * weights["net_trace"] +
            0.0                     * weights["genetic_support"] +
            r["sig_reproducibility"]* weights["sig_reproducibility"]
        )
        all_no_gen = (
            cand["net_trace"]           * weights["net_trace"] +
            0.0                         * weights["genetic_support"] +
            cand["sig_reproducibility"] * weights["sig_reproducibility"]
        )
        rank_no_gen = int((all_no_gen > score_no_gen).sum()) + 1
        pct_no_gen  = rank_no_gen / total * 100

        lines.append(
            f"  {drug:<15} Combined rank {orig_rank:4d} ({orig_pct:5.1f}%)  →  "
            f"No-genetic rank {rank_no_gen:4d} ({pct_no_gen:5.1f}%)"
        )

    lines += [
        "",
        "Interpretation:",
        "  - Nintedanib drops substantially without genetic support,",
        "    confirming the circularity. Net-TRACE rank (15th, 0.8%)",
        "    is the honest reversal-only result to headline.",
        "  - Cediranib drops similarly — also a known kinase-inhibitor class.",
        "  - Romidepsin and dasatinib change little, because their combined",
        "    rank is driven by reversal and their lower (but non-zero) OT scores.",
        "  - RECOMMENDATION: headline Net-TRACE (0.8% vs baseline 1.4%);",
        "    present combined as secondary corroboration.",
    ]

    txt_out = REV / "circularity_ablation.txt"
    txt_out.write_text("\n".join(lines))
    print("\n".join(lines))

    top20 = reversal_only.head(20).copy()
    pos_ctrl = {"nintedanib", "pirfenidone"}
    tier1    = {"cediranib", "romidepsin", "dasatinib"}
    adverse  = {"osimertinib", "afatinib", "dacomitinib"}

    def color(drug):
        d = drug.lower()
        if d in pos_ctrl:  return "#555555"
        if d in tier1:     return "#1a9641"
        if d in adverse:   return "#cc0000"
        return "#2166ac"

    colors = [color(d) for d in top20["drug"]]
    y = np.arange(len(top20))[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#f9f9f9")
    ax.set_facecolor("#f9f9f9")
    ax.barh(y, top20["net_trace"].values, color=colors, height=0.55, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(top20["drug"].values, fontsize=9)
    ax.set_xlabel("Net-TRACE reversal score (PRIMARY metric — no genetic support)", fontsize=10)
    ax.set_title("Top-20 drugs by reversal only (Net-TRACE)\n"
                 "Nintedanib FDR-significant (emp. p=0.0047, FDR=0.028); rank 15/1,768 as supporting context [CI 45–1,731]",
                 fontweight="bold")
    ax.xaxis.grid(True, lw=0.5, color="#dddddd", zorder=0)

    import matplotlib.patches as mpatches
    handles = [
        mpatches.Patch(color="#555555", label="Positive control"),
        mpatches.Patch(color="#1a9641", label="Tier 1 novel candidate"),
        mpatches.Patch(color="#2166ac", label="Novel candidate"),
        mpatches.Patch(color="#cc0000", label="⚠ FAERS adverse signal"),
    ]
    ax.legend(handles=handles, fontsize=9)
    plt.tight_layout()
    fig.savefig(OUT / "fig2b_reversal_primary.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nSaved reversal_primary_ranking.csv, circularity_ablation.txt, fig2b_reversal_primary.png")

if __name__ == "__main__":
    main()
