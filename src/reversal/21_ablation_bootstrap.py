"""
Final TRACE model: ablation table + bootstrap confidence intervals — RESEARCH.md §2c, §10.

Combines all four scoring arms:
  1. Baseline   — classic KS-based weighted connectivity (Lamb 2006)
  2. Net-TRACE  — negative cosine in network-propagated space (RWR, no weighting)
  3. W-TRACE    — Net-TRACE with cell-line lung-similarity weighting (§2a)
  4. VAE-TRACE  — negative cosine in VAE latent space (§1d)

Then integrates genetic support (Open Targets IPF scores, §2b) into a final
combined model via normalised rank aggregation.

Ablation table (RESEARCH.md §2c): shows positive control recovery at each stage.

Bootstrap CIs (RESEARCH.md §10): resample consensus IPF signature genes 1000×,
recompute the combined score, report 95% CIs on rank for top candidates.

Outputs:
  results/reversal/ablation_table.csv          — method × positive-control metric
  results/reversal/bootstrap_rank_ci.csv       — top candidate ranks with 95% CIs
  results/reversal/final_candidates_full.csv   — comprehensive ranked list

Usage:
    python src/reversal/21_ablation_bootstrap.py [--bootstrap 1000]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REV_DIR  = Path("results/reversal")
EMB_DIR  = Path("results/embedding")
META_DIR = Path("results/meta")
L1000_DIR = Path("results/l1000")

N_BOOT = int(sys.argv[sys.argv.index("--bootstrap") + 1]) if "--bootstrap" in sys.argv else 1000
POSITIVE_CONTROLS = ["pirfenidone", "nintedanib"]


# ---------------------------------------------------------------------------
# Load all score tables
# ---------------------------------------------------------------------------

def load_scores() -> pd.DataFrame:
    """
    Merge all scoring arms into one DataFrame indexed by drug name.
    Returns columns: drug, baseline_score, trace_score, weighted_trace, vae_score,
                     genetic_support, n_cell_lines, sig_reproducibility
    """
    # 1. Baseline + Net-TRACE (from combined_scores or individual files)
    base = pd.read_csv(REV_DIR / "combined_scores.csv")
    base = base.rename(columns={"baseline_score": "baseline",
                                 "trace_score":    "net_trace"})[
        ["drug", "baseline", "net_trace"]
    ]

    # 2. Weighted TRACE
    wt = pd.read_csv(REV_DIR / "weighted_trace_scores.csv")[["drug", "weighted_trace"]]

    # 3. VAE-TRACE (may not exist yet)
    vae_path = EMB_DIR / "vae_trace_scores.csv"
    if vae_path.exists():
        vae = pd.read_csv(vae_path)[["drug", "vae_score"]]
    else:
        print("  [WARN] vae_trace_scores.csv not found — VAE arm excluded from ablation")
        vae = None

    # 4. Genetic support + reproducibility from model_scores
    gen = pd.read_csv(REV_DIR / "model_scores.csv")[
        ["drug", "genetic_support", "sig_reproducibility", "n_cell_lines"]
    ]

    merged = base.merge(wt, on="drug", how="outer")
    if vae is not None:
        merged = merged.merge(vae, on="drug", how="outer")
    else:
        merged["vae_score"] = np.nan
    merged = merged.merge(gen, on="drug", how="left")
    merged = merged.fillna({"genetic_support": 0.0, "sig_reproducibility": 0.0,
                             "vae_score": np.nan})
    return merged


# ---------------------------------------------------------------------------
# Normalised rank score: converts a raw score column to [0,1] rank percentile
# where higher = better reversal candidate
# ---------------------------------------------------------------------------

def rank_pct(series: pd.Series, ascending: bool = False) -> pd.Series:
    """
    Rank entries; ties get average rank.
    ascending=False → highest score gets rank 1 (best reversal).
    Returns rank fraction [0, 1], 1 = best.
    """
    r = series.rank(ascending=ascending, method="average", na_option="bottom")
    return 1.0 - (r - 1) / (len(series) - 1)


# ---------------------------------------------------------------------------
# Ablation table: positive control recovery at each method step
# ---------------------------------------------------------------------------

def build_ablation_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each scoring arm, report nintedanib and pirfenidone rank/percentile.
    """
    arms = {
        "Baseline (KS)":        ("baseline",       False),
        "Net-TRACE":             ("net_trace",      True),
        "Weighted TRACE":        ("weighted_trace", True),
        "VAE-TRACE":             ("vae_score",      True),
    }

    records = []
    for arm_name, (col, higher_better) in arms.items():
        if col not in df.columns or df[col].isna().all():
            continue
        sub = df[["drug", col]].dropna()
        n   = len(sub)
        ranked = sub.sort_values(col, ascending=(not higher_better)).reset_index(drop=True)
        ranked["rank"] = range(1, n + 1)

        for pc in POSITIVE_CONTROLS:
            rows = ranked[ranked["drug"].str.lower().str.contains(pc, na=False)]
            if rows.empty:
                rank, pct = np.nan, np.nan
            else:
                rank = int(rows.iloc[0]["rank"])
                pct  = round(rank / n * 100, 1)
            records.append({
                "method":    arm_name,
                "drug":      pc,
                "rank":      rank,
                "n_drugs":   n,
                "pct":       pct,
            })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Combined model: weighted rank aggregation
# ---------------------------------------------------------------------------

def combined_score(df: pd.DataFrame) -> pd.Series:
    """
    Final score = weighted average of normalised rank percentiles.
    Weights from RESEARCH.md §2b:
      - Reversal score: 50% (net_trace or weighted if available, or VAE)
      - Genetic support: 30%
      - Reproducibility: 20%
    When multiple reversal arms exist, average them for the reversal component.
    """
    reversal_arms = []
    for col in ["net_trace", "weighted_trace", "vae_score"]:
        if col in df.columns and df[col].notna().sum() > 100:
            # ascending=False: higher reversal score → rank 1 → rank_pct = 1 (best)
            reversal_arms.append(rank_pct(df[col], ascending=False))

    if not reversal_arms:
        raise ValueError("No reversal scores available")

    reversal = pd.concat(reversal_arms, axis=1).mean(axis=1)
    gen      = rank_pct(df["genetic_support"], ascending=False)
    repro    = rank_pct(df["sig_reproducibility"].fillna(0), ascending=False)

    return 0.50 * reversal + 0.30 * gen + 0.20 * repro


# ---------------------------------------------------------------------------
# Bootstrap CIs on candidate ranks
# ---------------------------------------------------------------------------

def bootstrap_rank_ci(df: pd.DataFrame, n_boot: int = 1000) -> pd.DataFrame:
    """
    Resample consensus IPF signature gene weights with replacement.
    Recompute net_trace cosine reversal score from the resampled weights.
    Record rank distribution for the top 25 candidates.
    Returns DataFrame with: drug, median_rank, ci_lo_95, ci_hi_95, rank_stability
    """
    print(f"  Running {n_boot} bootstrap iterations (resampling IPF signature genes)…")

    # Load the network propagation scores (the IPF vector used for net_trace)
    network = pd.read_csv(EMB_DIR / "ipf_network_scores.csv", index_col=0)
    network.index = network.index.astype(str)
    ipf_vec_full = network["rwr_net"].values.astype(float)
    n_genes = len(ipf_vec_full)

    # Load drug signatures aligned to network genes
    drug_sig = pd.read_csv(L1000_DIR / "drug_signatures_landmark.csv.gz", index_col=0)
    drug_sig.index = drug_sig.index.astype(str)
    common = network.index.intersection(drug_sig.index)

    ipf_base = network.loc[common, "rwr_net"].values.astype(float)
    dm       = drug_sig.loc[common].fillna(0).values.astype(float)  # genes × drugs
    drugs    = drug_sig.columns.tolist()

    # Point-estimate ranking for reference
    def score(ipf_v: np.ndarray) -> np.ndarray:
        ipf_n  = np.linalg.norm(ipf_v) + 1e-10
        d_nrms = np.linalg.norm(dm, axis=0)
        d_nrms[d_nrms == 0] = 1e-10
        return -(ipf_v @ dm) / (ipf_n * d_nrms)

    point_scores = score(ipf_base)
    point_ranks  = (-point_scores).argsort().argsort() + 1   # rank 1 = best reversal

    # Top 25 candidates by point estimate
    top_idx = np.argsort(point_scores)[::-1][:25]

    boot_ranks = np.zeros((n_boot, 25), dtype=int)
    np.random.seed(42)
    for b in range(n_boot):
        # Resample genes with replacement (same size)
        idx   = np.random.choice(len(ipf_base), size=len(ipf_base), replace=True)
        boot_s = score(ipf_base[idx])
        boot_r = (-boot_s).argsort().argsort() + 1
        boot_ranks[b, :] = boot_r[top_idx]

        if (b + 1) % 200 == 0:
            print(f"    bootstrap {b+1}/{n_boot}")

    rows = []
    for j, di in enumerate(top_idx):
        ranks_j = boot_ranks[:, j]
        rows.append({
            "drug":          drugs[di],
            "point_rank":    int(point_ranks[di]),
            "median_rank":   int(np.median(ranks_j)),
            "ci_lo_95":      int(np.percentile(ranks_j, 2.5)),
            "ci_hi_95":      int(np.percentile(ranks_j, 97.5)),
            "rank_stability": round(float(np.std(ranks_j)), 1),
        })

    return pd.DataFrame(rows).sort_values("point_rank")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading all score tables…")
    df = load_scores()
    n  = len(df)
    print(f"  {n} drugs with scores")

    # ---- Ablation table ----
    print("\nBuilding ablation table…")
    abl = build_ablation_table(df)
    abl.to_csv(REV_DIR / "ablation_table.csv", index=False)
    print("\n=== Ablation: positive control recovery by method ===")
    pivot = abl.pivot(index="method", columns="drug", values=["rank", "pct"])
    print(pivot.to_string())

    # ---- Combined model ----
    print("\n\nBuilding combined model (reversal 50% + genetic 30% + repro 20%)…")
    df["combined_score"] = combined_score(df)
    df_sorted = df.sort_values("combined_score", ascending=False).reset_index(drop=True)
    df_sorted["combined_rank"] = range(1, len(df_sorted) + 1)

    out_full = REV_DIR / "final_candidates_full.csv"
    df_sorted.to_csv(out_full, index=False)
    print(f"  Saved {len(df_sorted)} drugs → {out_full.name}")

    print(f"\n=== Positive controls in combined model ===")
    for pc in POSITIVE_CONTROLS:
        row = df_sorted[df_sorted["drug"].str.lower().str.contains(pc, na=False)]
        if not row.empty:
            r = row.iloc[0]
            print(f"  {pc:15}  combined rank {int(r.combined_rank):4}/{n} "
                  f"({r.combined_rank/n*100:.1f}th pct)  "
                  f"reversal={r.get('net_trace', float('nan')):.4f}  "
                  f"genetic={r.genetic_support:.3f}")

    # Novel candidates (exclude positive controls)
    novel = df_sorted[~df_sorted["drug"].str.lower().str.contains(
        "|".join(POSITIVE_CONTROLS), na=False
    )].head(20)
    print(f"\n=== Top 20 novel candidates (combined model) ===")
    cols_show = ["combined_rank", "drug", "net_trace", "weighted_trace",
                 "vae_score", "genetic_support"]
    cols_show = [c for c in cols_show if c in novel.columns]
    print(novel[cols_show].to_string(index=False))

    # ---- Bootstrap CIs ----
    print(f"\n\nRunning bootstrap ({N_BOOT} iterations)…")
    ci_df = bootstrap_rank_ci(df, n_boot=N_BOOT)
    ci_df.to_csv(REV_DIR / "bootstrap_rank_ci.csv", index=False)

    print(f"\n=== Top 20 candidates: rank stability (95% CI) ===")
    print(f"{'Drug':30}  {'Point rank':>10}  {'CI 2.5%':>8}  {'CI 97.5%':>9}  {'SD':>6}")
    for _, r in ci_df.head(20).iterrows():
        print(f"  {r.drug:30}  {r.point_rank:>10}  {r.ci_lo_95:>8}  {r.ci_hi_95:>9}  {r.rank_stability:>6.1f}")

    print(f"\nOutputs → {REV_DIR}/")
    print("  ablation_table.csv")
    print("  bootstrap_rank_ci.csv")
    print("  final_candidates_full.csv")


if __name__ == "__main__":
    main()
