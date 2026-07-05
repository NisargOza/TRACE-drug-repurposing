from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT   = Path(__file__).resolve().parents[2]
META   = ROOT / "results" / "meta"
EMB    = ROOT / "results" / "embedding"
DATA   = ROOT / "data" / "raw"
EMB.mkdir(exist_ok=True)

ALPHA    = 0.85
MAX_ITER = 100
TOL      = 1e-6

EDGE_FILE = DATA / "string_entrez_edges_700.csv.gz"


def rwr(W: sp.csr_matrix, seed: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    p0 = np.abs(seed).astype(float)
    if p0.sum() == 0:
        return p0
    p0 /= p0.sum()
    p = p0.copy()
    for _ in range(MAX_ITER):
        p_new = (1 - alpha) * W.dot(p) + alpha * p0
        if np.abs(p_new - p).sum() < TOL:
            return p_new
        p = p_new
    return p


def build_network() -> tuple[list[str], sp.csr_matrix]:
    print(f"  Loading pre-computed Entrez edges from {EDGE_FILE.name} ...")
    edges = pd.read_csv(EDGE_FILE)
    edges["entrez1"] = edges["entrez1"].astype(str)
    edges["entrez2"] = edges["entrez2"].astype(str)
    print(f"  {len(edges):,} edges")

    nodes    = sorted(set(edges["entrez1"]) | set(edges["entrez2"]))
    node_idx = {n: i for i, n in enumerate(nodes)}
    N        = len(nodes)
    print(f"  Network: {N:,} nodes")

    row  = edges["entrez1"].map(node_idx).values
    col  = edges["entrez2"].map(node_idx).values
    data = np.ones(len(row), dtype=float)
    A    = sp.csr_matrix(
        (np.concatenate([data, data]),
         (np.concatenate([row, col]), np.concatenate([col, row]))),
        shape=(N, N), dtype=float,
    )
    col_sums = np.array(A.sum(axis=0)).flatten()
    col_sums[col_sums == 0] = 1.0
    W = A.multiply(1.0 / col_sums)
    return nodes, W


def main() -> None:
    sig_path = META / "uc_consensus_signature.csv"
    if not sig_path.exists():
        raise FileNotFoundError(
            f"{sig_path} not found. Run D1_uc_geo_de.py first."
        )

    sig = pd.read_csv(sig_path, index_col=0)
    sig.index = sig.index.astype(str)
    print(f"UC consensus signature: {len(sig):,} genes")

    print("Building STRING network ...")
    nodes, W = build_network()

    seed_series = pd.Series(0.0, index=nodes)
    overlap     = sig.index.intersection(pd.Index(nodes))
    seed_series[overlap] = sig.loc[overlap, "meta_log2FC"]
    print(f"Consensus genes on network: {len(overlap):,} / {len(sig):,}")

    seed_arr = seed_series.values.astype(float)
    up_seed  = np.clip(seed_arr,  0, None)
    dn_seed  = np.clip(-seed_arr, 0, None)

    print(f"Running signed RWR (alpha={ALPHA}) ...")
    rwr_up  = rwr(W, up_seed)
    rwr_dn  = rwr(W, dn_seed)
    rwr_net = rwr_up - rwr_dn

    out = pd.DataFrame({
        "entrez_id":     nodes,
        "rwr_up":        rwr_up,
        "rwr_down":      dn_seed,
        "rwr_net":       rwr_net,
        "in_consensus":  pd.Index(nodes).isin(overlap),
        "consensus_lfc": seed_series.values,
    }).set_index("entrez_id")

    out = out.sort_values("rwr_net", ascending=False)
    out_path = EMB / "uc_network_scores.csv"
    out.to_csv(out_path)
    print(f"\nSaved → {out_path}")
    print(f"Top 5 UC network nodes: {out.head(5).index.tolist()}")
    print("\nNext: python src/benchmarking/B3_dual_disease_scoring.py")


if __name__ == "__main__":
    main()
