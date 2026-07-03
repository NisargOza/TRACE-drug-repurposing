
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

EMB_DIR   = Path("results/embedding")
L1000_DIR = Path("results/l1000")
REV_DIR   = Path("results/reversal")
DE_DIR    = Path("results/de")
DATA_RAW  = Path("data/raw")

POSITIVE_CONTROLS = ["pirfenidone", "nintedanib"]


def build_lung_reference(landmark_entrez: list[int]) -> pd.Series:
    cache = DATA_RAW / "lung_reference_landmark.csv"
    if cache.exists():
        ref = pd.read_csv(cache, index_col=0).squeeze()
        if ref.notna().any():
            print(f"  [cache] {cache.name} ({ref.notna().sum()} genes)")
            return ref

    print("  Computing lung reference from GSE213001 normal controls...")
    counts_path = Path("data/processed/GSE213001/counts_raw.csv.gz")
    meta_path   = Path("data/processed/GSE213001/metadata.csv")
    e2g_path    = DE_DIR / "GSE213001_ensembl2entrez.csv"

    if not counts_path.exists():
        print("  [WARN] counts_raw.csv.gz not found")
        return pd.Series(dtype=float)

    meta = pd.read_csv(meta_path, index_col=0)
    ctrl_accessions = meta[meta["condition"] == "control"].index.tolist()
    ctrl = meta.loc[ctrl_accessions, "Sample_title"].tolist()
    print(f"  Normal lung controls: {len(ctrl)}")

    counts = pd.read_csv(counts_path, index_col=0)
    ctrl = [c for c in ctrl if c in counts.columns]
    counts = counts[ctrl]
    print(f"  Counts shape: {counts.shape}")

    e2g = pd.read_csv(e2g_path)
    e2g.columns = ["ensembl", "entrez"]
    e2g = e2g.dropna().drop_duplicates("ensembl").drop_duplicates("entrez")
    e2g["entrez"] = e2g["entrez"].astype(int).astype(str)

    counts.index = counts.index.astype(str)
    shared_ens = e2g["ensembl"][e2g["ensembl"].isin(counts.index)]
    sub = counts.loc[shared_ens.values].copy()
    sub.index = e2g.set_index("ensembl").loc[shared_ens.values, "entrez"].values

    sub = sub[~sub.index.duplicated(keep="first")]

    lm_str = [str(g) for g in landmark_entrez]
    sub = sub.reindex([g for g in lm_str if g in sub.index])

    ref = np.log1p(sub).median(axis=1)
    ref.index = ref.index.astype(str)
    ref.name = "lung_log_median"
    ref.to_csv(cache)
    print(f"  Lung reference: {ref.notna().sum()} landmark genes saved to {cache.name}")
    return ref


def build_cell_baselines(sig_info: pd.DataFrame,
                         drug_sig_path: Path) -> pd.DataFrame:
    cache = DATA_RAW / "l1000_cell_baselines_landmark.csv"
    if cache.exists():
        print(f"  [cache] {cache.name}")
        return pd.read_csv(cache, index_col=0)

    print("  Computing per-cell-line mean LFC profile as baseline proxy…")
    parquet_path = DATA_RAW / "l1000/all_signatures_landmark.parquet"
    if not parquet_path.exists():
        print(f"  [WARN] Parquet not found")
        return pd.DataFrame()

    import pyarrow.parquet as pq
    parquet_cols = set(pq.ParquetFile(parquet_path).schema_arrow.names)
    cl_sig_map = sig_info.set_index("sig_id")["cell_id"]
    valid_sigs  = [s for s in cl_sig_map.index if s in parquet_cols]
    print(f"  Drug sigs in parquet: {len(valid_sigs):,}")

    print("  Loading full parquet for cell-line mean profiles…")
    all_sigs = pd.read_parquet(parquet_path, columns=valid_sigs)
    all_sigs.index = all_sigs.index.astype(str)

    baselines = {}
    for cl, grp in sig_info[sig_info["sig_id"].isin(valid_sigs)].groupby("cell_id"):
        sids = [s for s in grp["sig_id"].values if s in all_sigs.columns]
        if len(sids) >= 5:
            baselines[cl] = all_sigs[sids].mean(axis=1).values

    if not baselines:
        print("  [WARN] No baselines computed")
        return pd.DataFrame()

    result = pd.DataFrame(baselines, index=all_sigs.index).T
    result.to_csv(cache)
    print(f"  Cell-line baselines: {result.shape} → {cache.name}")
    return result


def compute_weights(lung_ref: pd.Series,
                    cell_baselines: pd.DataFrame) -> pd.Series:
    if lung_ref.empty or lung_ref.notna().sum() < 50 or cell_baselines.empty:
        print("  [WARN] Insufficient data — using lung-adjacency heuristics")
        heur = {
            "A549":   2.0, "HCC515": 2.0,
            "HA1E":   1.0, "HEPG2":  1.0,
            "HT29":   0.5, "YAPC":   0.5, "HELA": 0.5,
            "MCF7":   0.3, "A375":   0.3, "PC3":  0.3,
        }
        all_cls = cell_baselines.index.tolist() if not cell_baselines.empty else list(heur)
        w = pd.Series({cl: heur.get(cl, 0.5) for cl in all_cls})
        return (w / w.sum()).rename("lung_weight")

    common = lung_ref.dropna().index.intersection(cell_baselines.columns)
    if len(common) < 50:
        print(f"  [WARN] Only {len(common)} common genes — falling back to heuristics")
        return compute_weights(pd.Series(), cell_baselines)

    lv = lung_ref[common].values.astype(float)
    corrs = {}
    for cl in cell_baselines.index:
        cv = cell_baselines.loc[cl, common].values.astype(float)
        r, _ = stats.spearmanr(lv, cv)
        corrs[cl] = float(r) if not np.isnan(r) else 0.0

    w = pd.Series(corrs)
    w = w.clip(lower=0)
    w = w + 0.05
    w = (w / w.sum()).rename("lung_weight")
    return w


def weighted_drug_signatures(sig_info: pd.DataFrame,
                              parquet_path: Path,
                              weights: pd.Series) -> pd.DataFrame:
    print("  Loading full parquet for drug signature weighting…")
    all_sigs = pd.read_parquet(parquet_path)
    all_sigs.index = all_sigs.index.astype(str)

    drug_groups = sig_info.groupby("pert_iname")["sig_id"].apply(list)
    weighted = {}
    for drug, sig_ids in drug_groups.items():
        present = [s for s in sig_ids if s in all_sigs.columns]
        if not present:
            continue
        cl_series = sig_info.set_index("sig_id").loc[present, "cell_id"]
        w_arr = np.array([weights.get(cl, weights.mean()) for cl in cl_series.values],
                         dtype=np.float32)
        w_arr /= w_arr.sum()
        weighted[drug] = (all_sigs[present].values * w_arr).sum(axis=1)

    return pd.DataFrame(weighted, index=all_sigs.index)


def compute_weighted_trace(weighted_sigs: pd.DataFrame,
                           network: pd.DataFrame) -> pd.DataFrame:
    common = network.index.intersection(weighted_sigs.index)
    ipf_v  = network.loc[common, "rwr_net"].values.astype(float)
    dm     = weighted_sigs.loc[common].fillna(0).values.astype(float)

    ipf_n  = np.linalg.norm(ipf_v) + 1e-10
    d_nrms = np.linalg.norm(dm, axis=0)
    d_nrms[d_nrms == 0] = 1e-10

    scores = -(ipf_v @ dm) / (ipf_n * d_nrms)
    df = pd.DataFrame({"drug": weighted_sigs.columns, "weighted_trace": scores})
    df = df.sort_values("weighted_trace", ascending=False).reset_index(drop=True)
    df["weighted_rank"] = range(1, len(df) + 1)
    return df


def main() -> None:
    parquet_path = DATA_RAW / "l1000/all_signatures_landmark.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet not found: {parquet_path}")

    sig_info = pd.read_csv(L1000_DIR / "sm_sig_info.csv", low_memory=False)
    network  = pd.read_csv(EMB_DIR / "ipf_network_scores.csv", index_col=0)
    network.index = network.index.astype(str)

    lg = pd.read_csv(L1000_DIR / "landmark_genes.csv")
    landmark_entrez = lg["pr_gene_id"].tolist()

    print("\n1. Building lung tissue reference (GSE213001 normal controls)…")
    lung_ref = build_lung_reference(landmark_entrez)

    print("\n2. Computing L1000 cell-line expression baselines…")
    cell_baselines = build_cell_baselines(sig_info, L1000_DIR / "drug_signatures_landmark.csv.gz")

    print("\n3. Computing Spearman lung-similarity weights…")
    weights = compute_weights(lung_ref, cell_baselines)
    weights.to_csv(EMB_DIR / "cellline_lung_similarity.csv")

    print("\n  Cell-line lung-similarity weights (top 15 by weight):")
    for cl, w in weights.sort_values(ascending=False).head(15).items():
        print(f"    {cl:12}  {w:.4f}")

    print("\n4. Computing weighted drug signatures…")
    weighted_sigs = weighted_drug_signatures(sig_info, parquet_path, weights)
    out_sigs = L1000_DIR / "drug_signatures_weighted.csv.gz"
    weighted_sigs.to_csv(out_sigs, compression="gzip")
    print(f"  Saved {weighted_sigs.shape} → {out_sigs.name}")

    print("\n5. Recomputing weighted TRACE scores…")
    wtrace = compute_weighted_trace(weighted_sigs, network)
    wtrace.to_csv(REV_DIR / "weighted_trace_scores.csv", index=False)
    print(f"  Saved {len(wtrace)} drugs → weighted_trace_scores.csv")

    n = len(wtrace)
    try:
        unweighted = pd.read_csv(REV_DIR / "trace_scores.csv")
        print(f"\n=== Positive control ranks (network TRACE vs. weighted TRACE) ===")
        for pc in POSITIVE_CONTROLS:
            wr = wtrace[wtrace["drug"].str.lower().str.contains(pc, na=False)]
            ur = unweighted[unweighted["drug"].str.lower().str.contains(pc, na=False)]
            if not wr.empty and not ur.empty:
                w_rank = int(wr.iloc[0]["weighted_rank"])
                u_rank = int(ur.iloc[0]["trace_rank"])
                chg = "improved" if w_rank < u_rank else ("worse" if w_rank > u_rank else "same")
                print(f"  {pc:15}  unweighted {u_rank:4}/{n}  →  weighted {w_rank:4}/{n}  ({chg})")
    except FileNotFoundError:
        pass

    print(f"\n=== Top 15 weighted TRACE candidates ===")
    print(f"{'Rank':>4}  {'Drug':30}  {'Score':>8}")
    for _, r in wtrace.head(15).iterrows():
        print(f"  {int(r.weighted_rank):>4}  {r.drug:30}  {r.weighted_trace:>8.4f}")

    print(f"\nOutputs saved to {EMB_DIR}/ and {L1000_DIR}/")


if __name__ == "__main__":
    main()
