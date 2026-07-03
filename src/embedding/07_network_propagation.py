
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import scipy.sparse as sp

DATA_RAW   = Path("data/raw")
EMB_DIR    = Path("results/embedding")
EMB_DIR.mkdir(parents=True, exist_ok=True)

STRING_URL  = "https://stringdb-downloads.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz"
STRING_FILE = DATA_RAW / "string_human_ppi.txt.gz"
INFO_URL    = "https://stringdb-downloads.org/download/protein.info.v12.0/9606.protein.info.v12.0.txt.gz"
INFO_FILE   = DATA_RAW / "string_human_info.txt.gz"

MIN_SCORE   = 700
ALPHA       = 0.85
MAX_ITER    = 100
TOL         = 1e-6


def download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  [skip] {dest.name}")
        return
    print(f"  Downloading {dest.name}...", end=" ", flush=True)
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in r.iter_content(1 << 20):
            f.write(chunk)
    print(f"done ({dest.stat().st_size / 1e6:.0f} MB)")


def load_string_network() -> pd.DataFrame:
    print(f"  Loading STRING PPI (score >= {MIN_SCORE})...")
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
    import subprocess, tempfile, csv

    sym_file  = Path(tempfile.mktemp(suffix=".txt"))
    out_file  = Path(tempfile.mktemp(suffix=".csv"))
    sym_file.write_text("\n".join(symbols))

    r_code = f"""
suppressPackageStartupMessages({{
  library(org.Hs.eg.db); library(AnnotationDbi)
}})
syms <- readLines("{sym_file}")
res  <- AnnotationDbi::select(org.Hs.eg.db, keys=syms,
          columns="ENTREZID", keytype="SYMBOL")
res  <- res[!is.na(res$ENTREZID),]
res  <- res[!duplicated(res$SYMBOL),]
write.csv(res, "{out_file}", row.names=FALSE)
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


def rwr(W: sp.csr_matrix, seed_scores: np.ndarray,
        alpha: float = ALPHA, max_iter: int = MAX_ITER,
        tol: float = TOL) -> np.ndarray:
    p0 = np.abs(seed_scores).astype(float)
    if p0.sum() == 0:
        return p0
    p0 /= p0.sum()
    p = p0.copy()
    for _ in range(max_iter):
        p_new = (1 - alpha) * W.dot(p) + alpha * p0
        if np.abs(p_new - p).sum() < tol:
            p = p_new
            break
        p = p_new
    return p


def main() -> None:
    print("Downloading STRING network files...")
    download(STRING_URL, STRING_FILE)
    download(INFO_URL, INFO_FILE)

    edges   = load_string_network()
    sym_map = load_string_id_map()

    edges["gene1"] = edges["protein1"].map(sym_map)
    edges["gene2"] = edges["protein2"].map(sym_map)
    edges = edges.dropna(subset=["gene1", "gene2"])

    print("  Mapping gene symbols -> Entrez IDs (via org.Hs.eg.db)...")
    all_syms  = list(set(edges["gene1"]) | set(edges["gene2"]))
    sym2entrez = map_symbol_to_entrez(all_syms)
    edges["entrez1"] = edges["gene1"].map(sym2entrez)
    edges["entrez2"] = edges["gene2"].map(sym2entrez)
    edges = edges.dropna(subset=["entrez1", "entrez2"])
    edges["entrez1"] = edges["entrez1"].astype(str)
    edges["entrez2"] = edges["entrez2"].astype(str)

    nodes = sorted(set(edges["entrez1"]) | set(edges["entrez2"]))
    node_idx = {n: i for i, n in enumerate(nodes)}
    N = len(nodes)
    print(f"  Network: {N:,} nodes, {len(edges):,} edges (after ID mapping)")

    row = edges["entrez1"].map(node_idx).values
    col = edges["entrez2"].map(node_idx).values
    data = np.ones(len(row))
    A = sp.csr_matrix(
        (np.concatenate([data, data]),
         (np.concatenate([row, col]), np.concatenate([col, row]))),
        shape=(N, N), dtype=float
    )
    col_sums = np.array(A.sum(axis=0)).flatten()
    col_sums[col_sums == 0] = 1
    W = A.multiply(1.0 / col_sums)

    consensus = pd.read_csv("results/meta/consensus_signature.csv", index_col=0)
    consensus.index = consensus.index.astype(str)

    seed = pd.Series(0.0, index=nodes)
    overlap = consensus.index.intersection(pd.Index(nodes))
    seed[overlap] = consensus.loc[overlap, "meta_log2FC"]
    print(f"  Consensus genes on network: {len(overlap):,} / {len(consensus):,}")

    up_seed   = seed.clip(lower=0).values
    down_seed = (-seed).clip(lower=0).values

    print(f"  Running RWR (alpha={ALPHA}, max_iter={MAX_ITER})...")
    up_scores   = rwr(W, up_seed)
    down_scores = rwr(W, down_seed)

    net_scores = up_scores - down_scores

    result = pd.DataFrame({
        "entrez_id":         nodes,
        "rwr_up":            up_scores,
        "rwr_down":          down_scores,
        "rwr_net":           net_scores,
        "in_consensus":      pd.Index(nodes).isin(overlap),
        "consensus_lfc":     seed.values,
    }).set_index("entrez_id")

    result = result.sort_values("rwr_net", ascending=False)
    result.to_csv(EMB_DIR / "ipf_network_scores.csv")

    stats_lines = [
        f"STRING PPI network (score >= {MIN_SCORE})",
        f"  Nodes (Entrez):   {N:,}",
        f"  Edges:            {len(edges):,}",
        f"  Consensus genes on network: {len(overlap):,} / {len(consensus):,}",
        f"  RWR alpha:        {ALPHA}",
        f"  Top 10 up-propagated genes (net RWR score):",
    ]
    top10 = result.head(10)
    for eid, row in top10.iterrows():
        stats_lines.append(f"    Entrez {eid:>8}  net={row['rwr_net']:.5f}  lfc={row['consensus_lfc']:.3f}")
    stats_lines += [
        f"  Top 10 down-propagated genes:",
    ]
    bot10 = result.tail(10).iloc[::-1]
    for eid, row in bot10.iterrows():
        stats_lines.append(f"    Entrez {eid:>8}  net={row['rwr_net']:.5f}  lfc={row['consensus_lfc']:.3f}")

    stats_text = "\n".join(stats_lines)
    (EMB_DIR / "network_stats.txt").write_text(stats_text)
    print(f"\n{stats_text}")
    print(f"\n  Scores -> results/embedding/ipf_network_scores.csv")
    print("Next: download L1000 drug signatures and run the same propagation")
    print("to get tissue-aware drug embeddings for reversal scoring (RESEARCH.md §2a).")


if __name__ == "__main__":
    main()
