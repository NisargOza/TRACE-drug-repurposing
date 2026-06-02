"""
Populate genetic support feature for top TRACE candidates — RESEARCH.md §2b.

Uses Open Targets drug(chemblId) endpoint to get drug targets as Ensembl IDs,
then scores each drug by the maximum OT IPF association score among its targets.

Requires:
  results/reversal/chembl_ids_cache.json  (built by 12_prioritization_model.py)
  results/reversal/ot_ipf_targets_cache.json

Updates:
  results/reversal/features.csv           — adds populated genetic_support column
  results/reversal/drug_targets_ensembl.json  — cached Ensembl target lists
  results/reversal/model_scores.csv       — re-ranked with genetic support
  results/reversal/final_candidates.csv   — updated top 50 novel candidates

Usage:
    python src/aim2_reversal/13_genetic_support.py
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REV_DIR  = Path("results/reversal")
META_DIR = Path("results/meta")

OT_API = "https://api.platform.opentargets.org/api/v4/graphql"
POSITIVE_CONTROLS = ["pirfenidone", "nintedanib"]


# ---------------------------------------------------------------------------
# Fetch Ensembl target IDs per drug via Open Targets drug endpoint
# ---------------------------------------------------------------------------

def fetch_ensembl_targets(chembl_ids: dict[str, str],
                           cache: Path) -> dict[str, list[str]]:
    """Returns {drug_name: [ensembl_id, ...]} via OT drug(chemblId) endpoint."""
    if cache.exists():
        cached = json.loads(cache.read_text())
        missing = {d: c for d, c in chembl_ids.items() if d not in cached}
        if not missing:
            print(f"  [skip] All {len(cached)} drugs already in cache")
            return cached
        print(f"  {len(cached)} cached, fetching {len(missing)} new...")
        chembl_ids = missing
        drug_targets = cached
    else:
        drug_targets = {}

    query_template = """
    query {{
      drug(chemblId: "{chembl_id}") {{
        name
        mechanismsOfAction {{
          rows {{ targets {{ id approvedSymbol }} }}
        }}
      }}
    }}
    """

    for i, (drug, chembl_id) in enumerate(chembl_ids.items()):
        if i % 20 == 0:
            print(f"  {i}/{len(chembl_ids)}", end="\r", flush=True)
        try:
            r = requests.post(
                OT_API,
                json={"query": query_template.format(chembl_id=chembl_id)},
                timeout=15,
            )
            data = r.json().get("data", {}).get("drug") or {}
            targets = []
            for row in (data.get("mechanismsOfAction") or {}).get("rows", []):
                for t in row.get("targets", []):
                    if t.get("id"):
                        targets.append(t["id"])
            drug_targets[drug] = list(set(targets))
            time.sleep(0.12)
        except Exception:
            drug_targets[drug] = []

    print(f"  {len(drug_targets)} drugs processed          ")
    cache.write_text(json.dumps(drug_targets))
    return drug_targets


# ---------------------------------------------------------------------------
# Score genetic support
# ---------------------------------------------------------------------------

def compute_genetic_support(drugs: list[str],
                             drug_targets: dict[str, list[str]],
                             ipf_scores: dict[str, float]) -> pd.Series:
    """
    For each drug, genetic_support = max OT IPF association score among targets.
    0 if no targets or no targets with IPF association.
    """
    scores = {}
    for drug in drugs:
        targets = drug_targets.get(drug, [])
        if not targets:
            scores[drug] = 0.0
        else:
            scores[drug] = max(ipf_scores.get(t, 0.0) for t in targets)
    return pd.Series(scores, name="genetic_support")


# ---------------------------------------------------------------------------
# Weighted rank combination (same as model 12)
# ---------------------------------------------------------------------------

def weighted_rank_score(feat: pd.DataFrame,
                        w_trace: float = 0.5,
                        w_genetic: float = 0.3,
                        w_repro: float = 0.2) -> pd.Series:
    n = len(feat)
    pct = pd.DataFrame(index=feat.index)
    pct["trace"]   = feat["trace_score"].rank(ascending=True) / n
    pct["genetic"] = feat["genetic_support"].rank(ascending=True) / n
    pct["repro"]   = feat["sig_reproducibility"].rank(ascending=True) / n
    return w_trace * pct["trace"] + w_genetic * pct["genetic"] + w_repro * pct["repro"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Load cached ChEMBL IDs and IPF target scores
    chembl_path = REV_DIR / "chembl_ids_cache.json"
    ot_path     = REV_DIR / "ot_ipf_targets_cache.json"

    if not chembl_path.exists():
        print("ERROR: chembl_ids_cache.json not found. Run 12_prioritization_model.py first.")
        return
    if not ot_path.exists():
        print("ERROR: ot_ipf_targets_cache.json not found. Run 12_prioritization_model.py first.")
        return

    chembl_ids  = json.loads(chembl_path.read_text())
    ipf_scores  = json.loads(ot_path.read_text())
    print(f"ChEMBL IDs available: {len(chembl_ids)}")
    print(f"OT IPF target scores: {len(ipf_scores)} targets")

    # Fetch Ensembl targets for all drugs with ChEMBL IDs
    print("\nFetching drug targets (Ensembl IDs) from Open Targets...")
    ensembl_cache = REV_DIR / "drug_targets_ensembl.json"
    drug_targets  = fetch_ensembl_targets(chembl_ids, ensembl_cache)

    # Summary
    n_with_targets = sum(1 for t in drug_targets.values() if t)
    print(f"  Drugs with ≥1 target: {n_with_targets}/{len(drug_targets)}")

    # Print positive control targets
    print("\n=== Positive control targets ===")
    for pc in POSITIVE_CONTROLS:
        match = next((d for d in drug_targets if pc.lower() in d.lower()), None)
        if match:
            targets = drug_targets[match]
            ipf_hits = {t: ipf_scores[t] for t in targets if t in ipf_scores}
            print(f"  {pc}: {len(targets)} targets, {len(ipf_hits)} with IPF association")
            if ipf_hits:
                top = sorted(ipf_hits.items(), key=lambda x: -x[1])[:5]
                for eid, score in top:
                    print(f"    {eid}  OT score={score:.4f}")

    # Update feature matrix
    feat = pd.read_csv(REV_DIR / "features.csv", index_col=0)
    gen_support = compute_genetic_support(feat.index.tolist(), drug_targets, ipf_scores)
    feat["genetic_support"] = gen_support
    feat.to_csv(REV_DIR / "features.csv")
    print(f"\nGenetic support updated:")
    print(f"  Non-zero entries: {(feat['genetic_support'] > 0).sum()}/{len(feat)}")
    print(f"  Max score: {feat['genetic_support'].max():.4f}")
    print(f"  Mean score (non-zero): {feat.loc[feat['genetic_support']>0,'genetic_support'].mean():.4f}")

    # Re-run weighted rank combination
    combined = pd.read_csv(REV_DIR / "combined_scores.csv")
    drugs = combined["drug"].tolist()

    final_score = weighted_rank_score(feat)
    result = pd.DataFrame({
        "drug":                combined["drug"].values,
        "final_score":         final_score.reindex(combined["drug"].values).values,
        "trace_score":         feat.reindex(combined["drug"].values)["trace_score"].values,
        "genetic_support":     feat.reindex(combined["drug"].values)["genetic_support"].values,
        "pathway_concordance": feat.reindex(combined["drug"].values)["pathway_concordance"].values,
        "sig_reproducibility": feat.reindex(combined["drug"].values)["sig_reproducibility"].values,
        "n_cell_lines":        feat.reindex(combined["drug"].values)["n_cell_lines"].values,
        "baseline_score":      combined["baseline_score"].values,
    }).sort_values("final_score", ascending=False)
    result["final_rank"] = range(1, len(result) + 1)
    result.to_csv(REV_DIR / "model_scores.csv", index=False)

    n = len(result)

    # Positive control summary
    print(f"\n=== Positive control ranks (with genetic support) ===")
    for pc in POSITIVE_CONTROLS:
        rows = result[result["drug"].str.lower().str.contains(pc.lower(), na=False)]
        if not rows.empty:
            r = rows.iloc[0]
            print(f"  {pc:15}  rank {int(r['final_rank']):4}/{n} "
                  f"({r['final_rank']/n*100:.1f}th pct)  "
                  f"TRACE={r['trace_score']:.4f}  "
                  f"genetic={r['genetic_support']:.4f}")

    # Ablation across weight configs
    print(f"\n=== Ablation (positive control ranks under different weights) ===")
    configs = {
        "trace_only  (1.0, 0.0, 0.0)": (1.0, 0.0, 0.0),
        "balanced    (0.5, 0.3, 0.2)": (0.5, 0.3, 0.2),
        "gen_heavy   (0.3, 0.5, 0.2)": (0.3, 0.5, 0.2),
    }
    abl_lines = ["Ablation with genetic support populated:\n"]
    for name, (wt, wg, wr) in configs.items():
        s = weighted_rank_score(feat, wt, wg, wr)
        ranked = s.rank(ascending=False)
        abl_lines.append(f"  {name}")
        for pc in POSITIVE_CONTROLS:
            match = feat.index[feat.index.str.lower().str.contains(pc.lower(), na=False)]
            if len(match):
                r = int(ranked[match[0]])
                abl_lines.append(f"    {pc:15} rank {r:4}/{n} ({r/n*100:.1f}th pct)")
        abl_lines.append("")
    abl_text = "\n".join(abl_lines)
    (REV_DIR / "ablation_summary.txt").write_text(abl_text)
    print(abl_text)

    # Final candidate list
    is_pc = result["drug"].str.lower().apply(
        lambda d: any(pc.lower() in d for pc in POSITIVE_CONTROLS)
    )
    final = result[~is_pc].head(50).copy()
    final.to_csv(REV_DIR / "final_candidates.csv", index=False)

    print(f"=== Top 20 novel candidates (with genetic support) ===")
    print(f"{'Rank':>4}  {'Drug':30}  {'Final':>6}  {'TRACE':>7}  {'Genetic':>8}  {'n_cells':>7}")
    print("-" * 70)
    for _, r in final.head(20).iterrows():
        print(f"  {int(r['final_rank']):>4}  {r['drug']:30}  "
              f"{r['final_score']:>6.4f}  {r['trace_score']:>7.4f}  "
              f"{r['genetic_support']:>8.4f}  {int(r['n_cell_lines']):>7}")

    # Plot: TRACE vs genetic support coloured by final rank
    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(result["trace_score"], result["genetic_support"],
                    c=result["final_rank"], cmap="RdYlGn_r",
                    s=10, alpha=0.5, vmin=1, vmax=n)
    plt.colorbar(sc, ax=ax, label="Final rank (low = better)")
    for pc in POSITIVE_CONTROLS:
        rows = result[result["drug"].str.lower().str.contains(pc.lower(), na=False)]
        if not rows.empty:
            r = rows.iloc[0]
            ax.scatter(r["trace_score"], r["genetic_support"],
                       s=120, marker="*", c="#d62728", zorder=6)
            ax.annotate(pc, (r["trace_score"], r["genetic_support"]),
                        fontsize=8, ha="left")
    # Annotate top 10 candidates
    for _, r in final.head(10).iterrows():
        ax.annotate(r["drug"], (r["trace_score"], r["genetic_support"]),
                    fontsize=6, alpha=0.7, ha="right")
    ax.set_xlabel("TRACE score")
    ax.set_ylabel("Genetic support (max OT IPF score)")
    ax.set_title("Drug ranking: TRACE score vs genetic target support")
    fig.tight_layout()
    fig.savefig(REV_DIR / "feature_importances.png", dpi=150)
    plt.close(fig)

    print(f"\n  → results/reversal/final_candidates.csv")
    print(f"  → results/reversal/model_scores.csv")
    print(f"  → results/reversal/feature_importances.png")
    print("\nAim 2 complete. Next: Aim 3 — ClinicalTrials.gov + held-out validation.")


if __name__ == "__main__":
    main()
