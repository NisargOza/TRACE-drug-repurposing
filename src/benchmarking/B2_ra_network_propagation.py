
import subprocess
import csv
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT   = Path(__file__).resolve().parents[2]
META   = ROOT / "results" / "meta"
EMB    = ROOT / "results" / "embedding"
DATA   = ROOT / "data" / "raw"
EMB.mkdir(exist_ok=True)

ALPHA      = 0.85
MAX_ITER   = 100
TOL        = 1e-6
MIN_SCORE  = 700

STRING_FILE = DATA / "string_human_ppi.txt.gz"
INFO_FILE   = DATA / "string_human_info.txt.gz"


def rwr(W: sp.csr_matrix, seed: np.ndarray,
        alpha: float = ALPHA) -> np.ndarray:
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


def load_string_network() -> pd.DataFrame:
    if not STRING_FILE.exists():
        raise FileNotFoundError(
            f"{STRING_FILE} not found.\n"
            "Run src/embedding/07_network_propagation.py first — it downloads "
            "the STRING raw files that B2 reuses."
        )
    print(f"  Loading STRING edges (score >= {MIN_SCORE}) ...")
    edges = pd.read_csv(STRING_FILE, sep=" ", compression="gzip",
                        usecols=["protein1", "protein2", "combined_score"])
    edges = edges[edges["combined_score"] >= MIN_SCORE].copy()
    print(f"  {len(edges):,} edges after score filter")
    return edges


def load_string_id_map() -> pd.Series:
    info = pd.read_csv(INFO_FILE, sep="\t", compression="gzip",
                       usecols=["#string_protein_id", "preferred_name"])
    info.columns = ["string_id", "gene_symbol"]
    return info.set_index("string_id")["gene_symbol"]


def map_symbol_to_entrez(symbols: list[str]) -> dict[str, str]:
    sym_file = Path(tempfile.mktemp(suffix=".txt"))
    out_file = Path(tempfile.mktemp(suffix=".csv"))
    sym_file.write_text("\n".join(symbols))

    r_code = f"""
suppressPackageStartupMessages({{
  library(org.Hs.eg.db); library(AnnotationDbi)
}})
syms <- readLines("{sym_file.as_posix()}")
res  <- AnnotationDbi::select(org.Hs.eg.db, keys=syms,
          columns="ENTREZID", keytype="SYMBOL")
res  <- res[!is.na(res$ENTREZID),]
res  <- res[!duplicated(res$SYMBOL),]
write.csv(res, "{out_file.as_posix()}", row.names=FALSE)
"""
    subprocess.run(["Rscript", "--vanilla", "-e", r_code],
                   capture_output=True, check=True)
    mapping = {}
    with open(out_file) as f:
        for row in csv.DictReader(f):
            mapping[row["SYMBOL"]] = row["ENTREZID"]
    sym_file.unlink(missing_ok=True)
    out_file.unlink(missing_ok=True)
    return mapping


def build_network() -> tuple[list[str], sp.csr_matrix]:
    edges   = load_string_network()
    sym_map = load_string_id_map()

    edges["gene1"] = edges["protein1"].map(sym_map)
    edges["gene2"] = edges["protein2"].map(sym_map)
    edges = edges.dropna(subset=["gene1", "gene2"])

    print("  Mapping gene symbols -> Entrez (via org.Hs.eg.db) ...")
    all_syms   = list(set(edges["gene1"]) | set(edges["gene2"]))
    sym2entrez = map_symbol_to_entrez(all_syms)

    edges["entrez1"] = edges["gene1"].map(sym2entrez)
    edges["entrez2"] = edges["gene2"].map(sym2entrez)
    edges = edges.dropna(subset=["entrez1", "entrez2"])
    edges["entrez1"] = edges["entrez1"].astype(str)
    edges["entrez2"] = edges["entrez2"].astype(str)

    nodes    = sorted(set(edges["entrez1"]) | set(edges["entrez2"]))
    node_idx = {n: i for i, n in enumerate(nodes)}
    N        = len(nodes)
    print(f"  Network: {N:,} nodes, {len(edges):,} edges")

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
    sig_path = META / "ra_consensus_signature.csv"
    if not sig_path.exists():
        raise FileNotFoundError(
            f"{sig_path} not found.\n"
            "Run Rscript src/benchmarking/ra_meta_analysis.R first."
        )

    sig = pd.read_csv(sig_path, index_col=0)
    sig.index = sig.index.astype(str)
    print(f"RA consensus signature: {len(sig):,} genes")

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
        "rwr_down":      rwr_dn,
        "rwr_net":       rwr_net,
        "in_consensus":  pd.Index(nodes).isin(overlap),
        "consensus_lfc": seed_series.values,
    }).set_index("entrez_id")

    out = out.sort_values("rwr_net", ascending=False)
    out_path = EMB / "ra_network_scores.csv"
    out.to_csv(out_path)
    print(f"\nSaved → {out_path}")
    print(f"Top 5 RA network nodes: {out.head(5).index.tolist()}")
    print("\nNext: python src/benchmarking/B3_dual_disease_scoring.py")


if __name__ == "__main__":
    main()