
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as spstats

ROOT  = Path(__file__).resolve().parents[2]
L1K   = ROOT / "results" / "l1000"
EMB   = ROOT / "results" / "embedding"
META  = ROOT / "results" / "meta"
BENCH = ROOT / "results" / "benchmarking"
BENCH.mkdir(parents=True, exist_ok=True)

N_TOP = 150


def cmap_score_single(disease_lfc: np.ndarray,
                      drug_sig: np.ndarray,
                      n_top: int = N_TOP) -> float:
    n = len(disease_lfc)
    n_top = min(n_top, n // 4)
    if n_top < 5:
        return np.nan

    sorted_idx = np.argsort(disease_lfc)
    dn_query = set(sorted_idx[:n_top].tolist())
    up_query = set(sorted_idx[-n_top:].tolist())

    drug_rank = spstats.rankdata(-drug_sig)
    miss_step = 1.0 / max(n - n_top, 1)

    def ks_one_tail(query_set: set) -> float:
        hw     = np.array([abs(disease_lfc[g]) for g in query_set])
        hw_sum = hw.sum()
        if hw_sum == 0:
            return 0.0
        query_list = list(query_set)
        order      = np.argsort([drug_rank[g] for g in query_list])
        genes_ord  = [query_list[o] for o in order]

        cum, max_dev = 0.0, 0.0
        hit_seen     = 0
        rank_sorted  = np.argsort(drug_rank)

        for pos, gene_idx in enumerate(rank_sorted):
            if gene_idx in query_set:
                cum      += abs(disease_lfc[gene_idx]) / hw_sum
                hit_seen += 1
            else:
                cum -= miss_step
            if abs(cum) > abs(max_dev):
                max_dev = cum
        return max_dev

    ks_up = ks_one_tail(up_query)
    ks_dn = ks_one_tail(dn_query)

    if np.sign(ks_up) != np.sign(ks_dn):
        return float(ks_up - ks_dn) / 2.0
    return 0.0


def pearson_reversal(disease_vec: np.ndarray,
                     drug_matrix: np.ndarray) -> np.ndarray:
    d     = disease_vec - disease_vec.mean()
    d_std = d.std()
    if d_std == 0:
        return np.zeros(drug_matrix.shape[1])
    dm      = drug_matrix - drug_matrix.mean(axis=0)
    dm_std  = dm.std(axis=0)
    dm_std[dm_std == 0] = 1.0
    r = (d @ dm) / (d_std * dm_std * len(d))
    return -r


def score_disease(label: str,
                  net_scores_path: Path,
                  consensus_path: Path,
                  drug_mat: pd.DataFrame) -> pd.DataFrame:
    print(f"\n--- {label} ---")

    net = pd.read_csv(net_scores_path, index_col=0)
    net.index = net.index.astype(str)
    dis_net = net["rwr_net"]

    drug_mat.index = drug_mat.index.astype(str)
    common_net = dis_net.index.intersection(drug_mat.index)
    print(f"  Network genes on drug matrix: {len(common_net):,}")
    dis_net_vec = dis_net.loc[common_net].values.astype(float)
    dm_net      = drug_mat.loc[common_net].values.astype(float)

    print(f"  Computing Pearson scores ({drug_mat.shape[1]:,} drugs) ...")
    pearson = pearson_reversal(dis_net_vec, dm_net)

    cons = pd.read_csv(consensus_path, index_col=0)
    cons.index = cons.index.astype(str)
    common_lfc = cons.index.intersection(drug_mat.index)
    print(f"  Consensus genes on drug matrix: {len(common_lfc):,}")

    dis_lfc = cons.loc[common_lfc, "meta_log2FC"].values.astype(float)
    dm_lfc  = drug_mat.loc[common_lfc].values.astype(float)

    drugs = drug_mat.columns.tolist()
    print(f"  Computing CMap scores ...")
    cmap = np.zeros(len(drugs))
    for j in range(len(drugs)):
        if j % 500 == 0:
            print(f"    {j:,}/{len(drugs):,}", end="\r", flush=True)
        cmap[j] = cmap_score_single(dis_lfc, dm_lfc[:, j])
    print(f"    {len(drugs):,}/{len(drugs):,} done    ")

    return pd.DataFrame({"drug": drugs, "pearson": pearson, "cmap": cmap})


def main() -> None:
    print("Loading L1000 drug matrix ...")
    drug_mat = pd.read_csv(
        L1K / "drug_signatures_landmark.csv.gz", index_col=0
    )
    drug_mat.index = drug_mat.index.astype(str)
    print(f"  Drug matrix: {drug_mat.shape[0]:,} genes × {drug_mat.shape[1]:,} drugs")

    diseases = [
        (
            "IPF",
            EMB  / "ipf_network_scores.csv",
            META / "consensus_signature.csv",
        ),
        (
            "RA",
            EMB  / "ra_network_scores.csv",
            META / "ra_consensus_signature.csv",
        ),
    ]

    all_scores = {}
    for label, net_path, cons_path in diseases:
        if not net_path.exists():
            print(f"WARNING: {net_path.name} missing — skipping {label}")
            continue
        if not cons_path.exists():
            print(f"WARNING: {cons_path.name} missing — skipping {label}")
            continue

        scores = score_disease(label, net_path, cons_path, drug_mat)
        scores.to_csv(BENCH / f"{label.lower()}_drug_scores.csv", index=False)
        print(f"  Saved → results/benchmarking/{label.lower()}_drug_scores.csv")
        all_scores[label] = scores

    if len(all_scores) == 2:
        ipf = all_scores["IPF"].rename(
            columns={"pearson": "ipf_pearson", "cmap": "ipf_cmap"})
        ra  = all_scores["RA"].rename(
            columns={"pearson": "ra_pearson",  "cmap": "ra_cmap"})
        merged = ipf.merge(ra, on="drug", how="outer")
        merged.to_csv(BENCH / "dual_disease_scores.csv", index=False)
        print(f"\nMerged → results/benchmarking/dual_disease_scores.csv "
              f"({len(merged):,} drugs)")

    print("\nNext: python src/benchmarking/B4_ablation_dual.py")


if __name__ == "__main__":
    main()