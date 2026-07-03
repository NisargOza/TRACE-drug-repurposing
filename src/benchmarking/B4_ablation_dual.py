
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as spstats

ROOT        = Path(__file__).resolve().parents[2]
L1K         = ROOT / "results" / "l1000"
EMB         = ROOT / "results" / "embedding"
META        = ROOT / "results" / "meta"
REV         = ROOT / "results" / "reversal"
BENCH       = ROOT / "results" / "benchmarking"
ACTIVES_DIR = ROOT / "data" / "known_actives"
BENCH.mkdir(parents=True, exist_ok=True)


def load_actives(disease: str) -> set[str]:
    path = ACTIVES_DIR / f"{disease.lower()}_actives.txt"
    if not path.exists():
        print(f"  WARNING: {path} not found")
        return set()
    return {l.strip().lower()
            for l in path.read_text().splitlines() if l.strip()}


def median_rank(score_series: pd.Series,
                actives: set[str],
                higher_is_better: bool = True) -> tuple[float, int]:
    df = (score_series.dropna()
                      .reset_index()
                      .rename(columns={"index": "drug",
                                       score_series.name or 0: "score"}))
    if "drug" not in df.columns:
        df.columns = ["drug", "score"]
    df = df.sort_values("score", ascending=not higher_is_better).reset_index(drop=True)
    df["rank"] = df.index + 1
    hits = df[df["drug"].str.lower().isin(actives)]
    if hits.empty:
        return np.nan, 0
    return float(np.median(hits["rank"])), len(hits)


def cosine_reversal(disease_vec: np.ndarray,
                    drug_matrix: np.ndarray) -> np.ndarray:
    d_norm  = np.linalg.norm(disease_vec) + 1e-10
    c_norms = np.linalg.norm(drug_matrix, axis=0)
    c_norms[c_norms == 0] = 1e-10
    return -(disease_vec @ drug_matrix) / (d_norm * c_norms)


def compute_arms(disease: str,
                 drug_mat: pd.DataFrame) -> dict[str, pd.Series]:
    drug_mat = drug_mat.copy()
    drug_mat.index = drug_mat.index.astype(str)
    drugs = drug_mat.columns.tolist()
    arms  = {}

    cons_path = (META / "consensus_signature.csv" if disease == "IPF"
                 else META / "ra_consensus_signature.csv")
    cons = pd.read_csv(cons_path, index_col=0)
    cons.index = cons.index.astype(str)
    common_lfc = cons.index.intersection(drug_mat.index)
    dis_lfc    = cons.loc[common_lfc, "meta_log2FC"].values.astype(float)
    dm_lfc     = drug_mat.loc[common_lfc].values.astype(float)

    skip_net_arr = cosine_reversal(dis_lfc, dm_lfc)
    arms["skip_network"] = pd.Series(skip_net_arr, index=drugs)

    if disease == "IPF":
        trace_df = pd.read_csv(REV / "trace_scores.csv")
        arms["skip_vae"] = trace_df.set_index("drug")["trace_score"]

        vae_df = pd.read_csv(EMB / "vae_trace_scores.csv")
        arms["full"] = vae_df.set_index("drug")["vae_score"]

        base_df = pd.read_csv(REV / "baseline_scores.csv")
        arms["baseline"] = -base_df.set_index("drug")["baseline_score"]

    else:
        b3_path = BENCH / "ra_drug_scores.csv"
        if b3_path.exists():
            ra_scores = pd.read_csv(b3_path)
            arms["skip_vae"] = ra_scores.set_index("drug")["pearson"]
            arms["full"]     = ra_scores.set_index("drug")["pearson"]
            arms["baseline"] = -ra_scores.set_index("drug")["cmap"]
        else:
            print(f"  WARNING: {b3_path.name} missing — run B3 first")

    return arms


def main() -> None:
    print("Loading L1000 drug matrix ...")
    drug_mat = pd.read_csv(
        L1K / "drug_signatures_landmark.csv.gz", index_col=0
    )
    drug_mat.index = drug_mat.index.astype(str)
    n_drugs = drug_mat.shape[1]
    print(f"  {drug_mat.shape[0]:,} genes × {n_drugs:,} drugs")

    records = []
    for disease in ["IPF", "RA"]:
        actives = load_actives(disease)
        print(f"\n{disease}: {len(actives)} known actives: {sorted(actives)}")

        arms = compute_arms(disease, drug_mat)
        for arm_name, scores in arms.items():
            med, n_found = median_rank(scores, actives, higher_is_better=True)
            pct = round(med / n_drugs * 100, 1) if not np.isnan(med) else np.nan
            records.append({
                "disease":             disease,
                "arm":                 arm_name,
                "median_rank_actives": med,
                "pct_rank":            pct,
                "n_actives_found":     n_found,
                "n_drugs":             n_drugs,
            })
            print(f"  {arm_name:15} median rank {int(med) if not np.isnan(med) else 'N/A'}"
                  f"/{n_drugs}  ({pct}th pct)  [{n_found} actives found]")

    ablation = pd.DataFrame(records)
    ablation.to_csv(BENCH / "ablation_dual_disease.csv", index=False)

    lines = [
        "Dual-Disease Ablation — Median Rank of Known Actives",
        "=" * 62,
        f"{'Disease':6} {'Arm':16} {'Median rank':>12} {'%ile':>7} {'Actives':>8}",
        "-" * 62,
    ]
    for _, row in ablation.iterrows():
        mr  = (f"{int(row['median_rank_actives'])}"
               if not np.isnan(row["median_rank_actives"]) else "N/A")
        pct = (f"{row['pct_rank']:.1f}%"
               if not np.isnan(row["pct_rank"]) else "N/A")
        lines.append(
            f"{row['disease']:6} {row['arm']:16} {mr:>12}/{int(row['n_drugs'])}"
            f" {pct:>7}  {int(row['n_actives_found']):>4} found"
        )
    summary = "\n".join(lines)
    (BENCH / "ablation_dual_disease_summary.txt").write_text(summary)
    print(f"\n{summary}")
    print("\nNext: python src/benchmarking/B5_benchmark_auroc.py")


if __name__ == "__main__":
    main()