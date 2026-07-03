
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

L1000_DIR = Path("results/l1000")
REV_DIR   = Path("results/reversal")
META_DIR  = Path("results/meta")

POSITIVE_CONTROLS = ["pirfenidone", "nintedanib"]

IPF_PATHWAY_GENES = set([
    "7040","7042","7043","4087","4088","4089","4090","7048","7049","7050",
    "4312","4313","4314","4316","4317","4318","4319","4320","1277","1278",
    "5291","5290","5293","5294","207","208","10000","2475","7157",
    "5594","5595","5596","5597","5598","5599","1147","4217","4216",
    "1499","1950","6932","7471","7473","3725","4040","90","8321",
    "6608","6612","2735","5727","6649",
])

OT_API = "https://api.platform.opentargets.org/api/v4/graphql"


def fetch_ipf_targets(cache: Path) -> dict[str, float]:
    if cache.exists():
        return json.loads(cache.read_text())
    print("  Fetching IPF target scores from Open Targets (EFO_0000768)...")
    query = """
    query { disease(efoId: "EFO_0000768") {
      associatedTargets(page: {index: 0, size: 500}) {
        rows { target { id } score }
      }
    }}"""
    try:
        r = requests.post(OT_API, json={"query": query}, timeout=30)
        r.raise_for_status()
        rows = r.json()["data"]["disease"]["associatedTargets"]["rows"]
        scores = {row["target"]["id"]: row["score"] for row in rows}
        cache.write_text(json.dumps(scores))
        print(f"  {len(scores)} IPF-associated targets")
        return scores
    except Exception as e:
        print(f"  [WARN] {e}  — using empty set")
        return {}


CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data"


def fetch_chembl_ids(drugs: list[str], cache: Path) -> dict[str, str]:
    if cache.exists():
        return json.loads(cache.read_text())
    print(f"  Looking up ChEMBL IDs for {len(drugs)} drugs...")
    chembl_map = {}
    for i, drug in enumerate(drugs):
        if i % 25 == 0:
            print(f"    {i}/{len(drugs)}", end="\r", flush=True)
        try:
            r = requests.get(
                f"{CHEMBL_API}/molecule/search",
                params={"q": drug, "format": "json", "limit": 1},
                timeout=10,
            )
            mols = r.json().get("molecules", [])
            if mols:
                chembl_map[drug] = mols[0]["molecule_chembl_id"]
            time.sleep(0.15)
        except Exception:
            pass
    print(f"    {len(chembl_map)} ChEMBL IDs resolved")
    cache.write_text(json.dumps(chembl_map))
    return chembl_map


def fetch_drug_targets(chembl_ids: dict[str, str], cache: Path) -> dict[str, list[str]]:
    if cache.exists():
        return json.loads(cache.read_text())
    print(f"  Fetching targets for {len(chembl_ids)} drugs via ChEMBL...")
    drug_targets = {}
    for i, (drug, chembl_id) in enumerate(chembl_ids.items()):
        if i % 25 == 0:
            print(f"    {i}/{len(chembl_ids)}", end="\r", flush=True)
        try:
            r = requests.get(
                f"{CHEMBL_API}/mechanism",
                params={"molecule_chembl_id": chembl_id, "format": "json", "limit": 50},
                timeout=10,
            )
            mechs = r.json().get("mechanisms", [])
            targets = [m["target_chembl_id"] for m in mechs if m.get("target_chembl_id")]
            drug_targets[drug] = targets
            time.sleep(0.15)
        except Exception:
            drug_targets[drug] = []
    print(f"    Done.          ")
    cache.write_text(json.dumps(drug_targets))
    return drug_targets


def compute_genetic_features(
    drugs: list[str],
    drug_targets: dict[str, list[str]],
    ipf_target_scores: dict[str, float],
    ipf_pathway_genes_entrez: set[str],
) -> pd.DataFrame:
    rows = []
    for drug in drugs:
        targets = set(drug_targets.get(drug, []))
        if not targets:
            rows.append({"drug": drug, "genetic_support": 0.0, "pathway_concordance": 0.0, "n_targets": 0})
            continue
        gen_scores = [ipf_target_scores.get(t, 0.0) for t in targets]
        genetic_support = max(gen_scores)
        ipf_target_set = set(ipf_target_scores.keys())
        pathway_frac = len(targets & ipf_target_set) / len(targets)
        rows.append({
            "drug": drug,
            "genetic_support": genetic_support,
            "pathway_concordance": pathway_frac,
            "n_targets": len(targets),
        })
    return pd.DataFrame(rows).set_index("drug")


def weighted_rank_score(feat: pd.DataFrame,
                        w_trace: float = 0.5,
                        w_genetic: float = 0.3,
                        w_repro: float = 0.2) -> pd.Series:
    n = len(feat)
    pct = pd.DataFrame(index=feat.index)
    pct["trace"]   = feat["trace_score"].rank(ascending=True) / n
    pct["genetic"] = feat["genetic_support"].rank(ascending=True) / n
    pct["repro"]   = feat["sig_reproducibility"].rank(ascending=True) / n
    return (w_trace * pct["trace"] + w_genetic * pct["genetic"] + w_repro * pct["repro"])


def main() -> None:
    scores   = pd.read_csv(REV_DIR / "combined_scores.csv")
    drugs    = scores["drug"].tolist()
    sig_info = pd.read_csv(L1000_DIR / "sm_sig_info.csv", low_memory=False)
    n_cells  = sig_info.groupby("pert_iname")["cell_id"].nunique().rename("n_cell_lines")

    feat_cache = REV_DIR / "features.csv"
    if feat_cache.exists():
        feat = pd.read_csv(feat_cache, index_col=0)
        print(f"Loaded cached features: {feat.shape}")
    else:
        print("Computing signature reproducibility (slow, ~2 min)...")
        sig_mat = pd.read_csv(L1000_DIR / "drug_signatures_landmark.csv.gz", index_col=0)
        sig_mat.index = sig_mat.index.astype(str)
        from sklearn.metrics.pairwise import cosine_similarity as cos_sim
        repro = {}
        for i, drug in enumerate(drugs):
            if i % 200 == 0: print(f"  {i}/{len(drugs)}", end="\r", flush=True)
            drug_sigs = sig_info[sig_info["pert_iname"] == drug]["sig_id"].tolist()
            present = [s for s in drug_sigs if s in sig_mat.columns]
            if len(present) < 2: repro[drug] = 0.0; continue
            sub = sig_mat[present].T.fillna(0).values
            sims = cos_sim(sub)
            repro[drug] = float(sims[np.triu_indices(len(sims), k=1)].mean())
        print(f"  Done.          ")
        repro_s = pd.Series(repro, name="sig_reproducibility")

        feat = scores.set_index("drug")[["trace_score", "baseline_score"]].copy()
        feat["sig_reproducibility"] = repro_s.reindex(drugs)
        feat["n_cell_lines"] = n_cells.reindex(drugs)
        feat["genetic_support"]     = 0.0
        feat["pathway_concordance"] = 0.0
        feat.to_csv(feat_cache)

    trace_top100 = scores.nlargest(100, "trace_score")["drug"].tolist()
    query_drugs  = list(set(trace_top100 + POSITIVE_CONTROLS))

    print(f"\nFetching genetic support for {len(query_drugs)} top candidates...")
    ipf_targets   = fetch_ipf_targets(REV_DIR / "ot_ipf_targets_cache.json")
    chembl_ids    = fetch_chembl_ids(query_drugs, REV_DIR / "chembl_ids_cache.json")
    drug_targets  = fetch_drug_targets(chembl_ids, REV_DIR / "drug_targets_cache.json")
    gen_feat      = compute_genetic_features(query_drugs, drug_targets,
                                             ipf_targets, IPF_PATHWAY_GENES)

    for col in ["genetic_support", "pathway_concordance", "n_targets"]:
        feat.loc[gen_feat.index, col] = gen_feat[col]
    feat.to_csv(feat_cache)

    print("\n=== Positive control genetic support ===")
    for pc in POSITIVE_CONTROLS:
        rows = feat[feat.index.str.lower().str.contains(pc.lower(), na=False)]
        if not rows.empty:
            r = rows.iloc[0]
            targets = drug_targets.get(pc, drug_targets.get(
                next((k for k in drug_targets if pc.lower() in k.lower()), pc), []))
            print(f"  {pc}: genetic_support={r['genetic_support']:.4f}  "
                  f"pathway_concordance={r['pathway_concordance']:.4f}  "
                  f"n_targets={int(r.get('n_targets',0))}  targets_sample={targets[:4]}")

    print("\nComputing weighted rank combination...")
    final_score = weighted_rank_score(feat)
    result = pd.DataFrame({
        "drug":              feat.index,
        "final_score":       final_score.values,
        "trace_score":       feat["trace_score"].values,
        "genetic_support":   feat["genetic_support"].values,
        "pathway_concordance": feat["pathway_concordance"].values,
        "sig_reproducibility": feat["sig_reproducibility"].values,
        "n_cell_lines":      feat["n_cell_lines"].values,
        "baseline_score":    feat["baseline_score"].values,
    }).sort_values("final_score", ascending=False)
    result["final_rank"] = range(1, len(result) + 1)
    result.to_csv(REV_DIR / "model_scores.csv", index=False)

    n = len(result)
    print(f"\n=== Positive control final ranks ===")
    for pc in POSITIVE_CONTROLS:
        rows = result[result["drug"].str.lower().str.contains(pc.lower(), na=False)]
        if not rows.empty:
            r = rows.iloc[0]
            print(f"  {pc:15}  rank {int(r['final_rank']):4}/{n} "
                  f"({r['final_rank']/n*100:.1f}th pct)  "
                  f"score={r['final_score']:.4f}  "
                  f"trace={r['trace_score']:.4f}  genetic={r['genetic_support']:.4f}")

    print("\n=== Ablation: sensitivity to weight changes ===")
    weight_configs = {
        "trace_only     (1.0, 0.0, 0.0)": (1.0, 0.0, 0.0),
        "balanced       (0.5, 0.3, 0.2)": (0.5, 0.3, 0.2),
        "genetic_heavy  (0.3, 0.5, 0.2)": (0.3, 0.5, 0.2),
        "repro_heavy    (0.4, 0.3, 0.3)": (0.4, 0.3, 0.3),
    }
    abl_lines = ["Ablation: positive control ranks under different weight configurations\n"]
    for config_name, (wt, wg, wr) in weight_configs.items():
        s = weighted_rank_score(feat, wt, wg, wr).rank(ascending=False)
        abl_lines.append(f"  {config_name}")
        for pc in POSITIVE_CONTROLS:
            rows = feat[feat.index.str.lower().str.contains(pc.lower(), na=False)]
            if not rows.empty:
                rank = int(s[rows.index[0]])
                abl_lines.append(f"    {pc:15} rank {rank:4}/{n} ({rank/n*100:.1f}th pct)")
        abl_lines.append("")
    abl_text = "\n".join(abl_lines)
    (REV_DIR / "ablation_summary.txt").write_text(abl_text)
    print(abl_text)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    sc = ax.scatter(result["trace_score"], result["genetic_support"],
                    c=result["final_rank"], cmap="RdYlGn_r", s=8, alpha=0.5,
                    vmin=1, vmax=n)
    plt.colorbar(sc, ax=ax, label="Final rank")
    for pc in POSITIVE_CONTROLS:
        rows = result[result["drug"].str.lower().str.contains(pc.lower(), na=False)]
        if not rows.empty:
            r = rows.iloc[0]
            ax.scatter(r["trace_score"], r["genetic_support"],
                       s=100, marker="*", c="#d62728", zorder=6)
            ax.annotate(pc, (r["trace_score"], r["genetic_support"]), fontsize=7)
    ax.set_xlabel("TRACE score")
    ax.set_ylabel("Genetic support (max OT score)")
    ax.set_title("TRACE score vs genetic support")

    ax = axes[1]
    top50 = result.head(50)
    colors = ["#d62728" if any(pc.lower() in d.lower() for pc in POSITIVE_CONTROLS)
              else "#1f77b4" for d in top50["drug"]]
    ax.barh(range(len(top50)), top50["final_score"].values[::-1],
            color=colors[::-1])
    ax.set_yticks(range(len(top50)))
    ax.set_yticklabels(top50["drug"].values[::-1], fontsize=6)
    ax.set_xlabel("Final score")
    ax.set_title("Top 50 candidates")

    fig.tight_layout()
    fig.savefig(REV_DIR / "feature_importances.png", dpi=150)
    plt.close(fig)

    is_pc = result["drug"].str.lower().apply(
        lambda d: any(pc.lower() in d for pc in POSITIVE_CONTROLS)
    )
    final = result[~is_pc].head(50).copy()
    final.to_csv(REV_DIR / "final_candidates.csv", index=False)

    print(f"\n=== Top 20 novel candidates ===")
    print(f"{'Rank':>4}  {'Drug':30}  {'Final':>6}  {'TRACE':>7}  {'Genetic':>8}  {'n_cells':>7}")
    print("-" * 70)
    for _, r in final.head(20).iterrows():
        print(f"  {int(r['final_rank']):>4}  {r['drug']:30}  "
              f"{r['final_score']:>6.4f}  {r['trace_score']:>7.4f}  "
              f"{r['genetic_support']:>8.4f}  {int(r['n_cell_lines']):>7}")

    print(f"\n  Final candidates → results/reversal/final_candidates.csv")
    print("Next: Aim 3 — ClinicalTrials.gov + FAERS validation of top candidates.")


if __name__ == "__main__":
    main()
