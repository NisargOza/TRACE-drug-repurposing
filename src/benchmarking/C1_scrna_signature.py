"""
scRNA-seq AT2→AT1 alveolar transition signature arm (C1).

Downloads two IPF single-cell atlases and computes the AT2→AT1 transition
differential-expression signature, which captures the blocked alveolar
epithelial differentiation step that drives IPF fibrosis.

Datasets:
  GSE135893  Habermann 2020, Sci Adv — IPF lung scRNA-seq (10x Chromium)
             114,397 cells; AT1=771, AT2=9,311, KRT5-/KRT17+=485, transitional=1,160
             celltype col: 'celltype'; donor col: 'Sample_Name'; disease: 'Diagnosis'
  GSE136831  Adams 2020, Sci Adv — multi-tissue IPF lung scRNA-seq
             celltype col: 'Manuscript_Identity' (ATI, ATII); donor: 'Subject_Identity'

Method A (pseudo-bulk, preferred): per-donor mean of AT2 vs AT1, scipy t-test,
  BH FDR correction. Used when >=3 donors have both AT1 and AT2 cells.
Method B (fallback): per-cell t-test between AT1 and AT2 clusters.

Note: MTX files are large (~1-2 GB compressed). First run takes ~30-60 min.
Checkpoints in results/scrna/ allow fast reruns.

Outputs:
  results/scrna/at2_at1_transition_signature.csv
  results/scrna/at2_at1_network_scores.csv
  results/scrna/at2_at1_drug_scores.csv
  results/scrna/scrna_vs_bulk_comparison.csv
  results/scrna/scrna_summary.txt

Usage:
  python src/benchmarking/C1_scrna_signature.py
"""

import csv
import gzip
import io
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats
from statsmodels.stats.multitest import multipletests

ROOT   = Path(__file__).resolve().parents[2]
RAW    = ROOT / "data/raw/scrna"
OUT    = ROOT / "results/scrna"
L1K    = ROOT / "results/l1000"
BENCH  = ROOT / "results/benchmarking"
RAW.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

ALPHA = 0.85; MAX_ITER = 100; TOL = 1e-6  # RWR params — match 07_network_propagation.py

BASE135 = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE135nnn/GSE135893/suppl/"
BASE136 = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE136nnn/GSE136831/suppl/"

DATASETS = {
    "GSE135893": {
        "mtx":      (BASE135 + "GSE135893_matrix.mtx.gz",               RAW / "GSE135893_matrix.mtx.gz"),
        "barcodes": (BASE135 + "GSE135893_barcodes.tsv.gz",             RAW / "GSE135893_barcodes.tsv.gz"),
        "genes":    (BASE135 + "GSE135893_genes.tsv.gz",                RAW / "GSE135893_genes.tsv.gz"),
        "metadata": (BASE135 + "GSE135893_IPF_metadata.csv.gz",         RAW / "GSE135893_metadata.csv.gz"),
        "ct_col":   "celltype",     # column with cell type label
        "donor_col":"Sample_Name",  # column with donor/patient ID
        "disease_col": "Diagnosis", # column with disease label (IPF/Control)
        "at2_label":   "AT2",
        "at1_label":   "AT1",
    },
    "GSE136831": {
        "mtx":      (BASE136 + "GSE136831_RawCounts_Sparse.mtx.gz",                    RAW / "GSE136831_matrix.mtx.gz"),
        "barcodes": (BASE136 + "GSE136831_AllCells.cellBarcodes.txt.gz",               RAW / "GSE136831_barcodes.tsv.gz"),
        "genes":    (BASE136 + "GSE136831_AllCells.GeneIDs.txt.gz",                    RAW / "GSE136831_genes.tsv.gz"),
        "metadata": (BASE136 + "GSE136831_AllCells.Samples.CellType.MetadataTable.txt.gz",
                     RAW / "GSE136831_metadata.tsv.gz"),
        "ct_col":   "Manuscript_Identity",
        "donor_col":"Subject_Identity",
        "disease_col": "Disease_Identity",
        "at2_label":   "ATII",
        "at1_label":   "ATI",
    },
}


def download(url: str, dest: Path) -> bool:
    if dest.exists():
        print(f"    [cached] {dest.name}")
        return True
    print(f"    Downloading {dest.name} ...", end=" ", flush=True)
    try:
        with urlopen(url, timeout=600) as r:
            total = int(r.headers.get("Content-Length", 0))
            data  = bytearray()
            chunk_size = 1 << 20
            while True:
                chunk = r.read(chunk_size)
                if not chunk:
                    break
                data.extend(chunk)
                if total:
                    print(f"\r    Downloading {dest.name} ... {len(data)/1e6:.0f}/{total/1e6:.0f} MB",
                          end="", flush=True)
        dest.write_bytes(data)
        print(f"\r    Downloaded {dest.name}: {len(data)/1e6:.1f} MB")
        return True
    except Exception as e:
        print(f"\n    FAILED: {e}")
        return False


def load_metadata(path: Path, ct_col: str, donor_col: str, disease_col: str,
                  at2_label: str, at1_label: str) -> pd.DataFrame:
    """Load metadata CSV/TSV, return df with barcode index."""
    opener = gzip.open if str(path).endswith(".gz") else open
    sep = "\t" if "tsv" in str(path) or "txt" in str(path) else ","
    with opener(path, "rt", encoding="utf-8", errors="replace") as f:
        df = pd.read_csv(f, sep=sep, index_col=0, quoting=csv.QUOTE_ALL if sep=="," else csv.QUOTE_MINIMAL)
    # Normalize index (strip quotes)
    df.index = df.index.str.strip('"')
    for col in [ct_col, donor_col, disease_col]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip('"')
    print(f"    Metadata: {len(df):,} cells, columns: {list(df.columns[:6])}")
    at2 = df[df[ct_col] == at2_label]; at1 = df[df[ct_col] == at1_label]
    print(f"    AT2 ({at2_label}): {len(at2):,}, AT1 ({at1_label}): {len(at1):,}")
    return df


def stream_mtx_subset(mtx_path: Path, barcodes_path: Path, genes_path: Path,
                      target_barcodes: set) -> tuple[sp.csr_matrix, list, list]:
    """Stream-read MTX.gz, keep only columns for target barcodes. Returns (mat, genes, sub_barcodes)."""
    barcodes = []
    opener = gzip.open if str(barcodes_path).endswith(".gz") else open
    with opener(barcodes_path, "rt") as f:
        for line in f:
            barcodes.append(line.strip().strip('"'))
    # Map barcode → original col index
    bc_to_orig_idx = {b: i for i, b in enumerate(barcodes)}
    # Map target barcodes to new (compact) column indices
    sub_barcodes = [b for b in barcodes if b in target_barcodes]
    orig_to_new  = {bc_to_orig_idx[b]: j for j, b in enumerate(sub_barcodes)}

    genes = []
    with opener(genes_path, "rt") as f:
        for line in f:
            genes.append(line.strip().split("\t")[0].strip('"'))  # first field

    print(f"    Streaming MTX ({mtx_path.stat().st_size/1e6:.0f} MB) → {len(sub_barcodes)} target cells ...")
    rows, cols, vals = [], [], []
    n_lines = 0
    opener_mtx = gzip.open if str(mtx_path).endswith(".gz") else open
    with opener_mtx(mtx_path, "rt") as f:
        header_done = False
        for line in f:
            if line.startswith("%"):
                continue
            if not header_done:
                header_done = True  # skip dims line
                continue
            parts = line.split()
            g_idx = int(parts[0]) - 1  # 1-indexed
            c_idx = int(parts[1]) - 1
            if c_idx in orig_to_new:
                rows.append(g_idx)
                cols.append(orig_to_new[c_idx])
                vals.append(float(parts[2]))
            n_lines += 1
            if n_lines % 5_000_000 == 0:
                print(f"      ... {n_lines//1e6:.0f}M entries read", flush=True)

    n_genes = len(genes); n_cells = len(sub_barcodes)
    mat = sp.csr_matrix((vals, (rows, cols)), shape=(n_genes, n_cells), dtype=np.float32)
    print(f"    Matrix shape: {mat.shape}, nnz: {mat.nnz:,}")
    return mat, genes, sub_barcodes


def compute_pseudobulk_de(mat: sp.csr_matrix, genes: list,
                          meta: pd.DataFrame, sub_barcodes: list,
                          ct_col: str, donor_col: str,
                          at2_label: str, at1_label: str) -> pd.DataFrame:
    """Pseudo-bulk DE (AT1 vs AT2) using per-donor averages."""
    bc_to_j = {b: j for j, b in enumerate(sub_barcodes)}
    # Group by donor × cell type, compute log1p mean
    pb: dict[tuple, np.ndarray] = {}
    for barcode, row in meta.iterrows():
        if barcode not in bc_to_j:
            continue
        ct = row[ct_col]; donor = row[donor_col]
        if ct not in (at2_label, at1_label):
            continue
        j = bc_to_j[barcode]
        key = (donor, ct)
        if key not in pb:
            pb[key] = []
        pb[key].append(j)

    # Average per (donor, ct)
    donors_with_both = set(d for d, ct in pb if (d, at2_label) in pb and (d, at1_label) in pb)
    print(f"    Donors with both AT1+AT2: {len(donors_with_both)}")

    if len(donors_with_both) >= 2:
        # Method A: pseudo-bulk
        mat_dense = mat.toarray().astype(np.float32)
        np.log1p(mat_dense, out=mat_dense)
        at2_means = np.array([mat_dense[:, pb[(d, at2_label)]].mean(axis=1)
                              for d in donors_with_both])
        at1_means = np.array([mat_dense[:, pb[(d, at1_label)]].mean(axis=1)
                              for d in donors_with_both])
        lfc = at1_means.mean(axis=0) - at2_means.mean(axis=0)
        t_stat, pvals = stats.ttest_rel(at1_means, at2_means, axis=0)
        method = f"pseudo-bulk, {len(donors_with_both)} donors (Method A)"
        del mat_dense
    else:
        # Method B: per-cell t-test
        print("    WARNING: <2 donors with both AT1+AT2; using per-cell t-test (Method B)")
        all_at2 = [j for (d, ct), jlist in pb.items() if ct == at2_label for j in jlist]
        all_at1 = [j for (d, ct), jlist in pb.items() if ct == at1_label for j in jlist]
        mat_at2 = np.log1p(mat[:, all_at2].toarray().astype(float))
        mat_at1 = np.log1p(mat[:, all_at1].toarray().astype(float))
        lfc = mat_at1.mean(axis=1) - mat_at2.mean(axis=1)
        t_stat, pvals = stats.ttest_ind(mat_at1.T, mat_at2.T, axis=0)
        method = "per-cell t-test (Method B)"

    _, padj, _, _ = multipletests(np.nan_to_num(pvals, nan=1.0), method="fdr_bh")
    sig = ((padj < 0.05) & (np.abs(lfc) > 0.5)).sum()
    print(f"    {method}: {sig:,}/{len(lfc):,} significant (FDR<0.05, |LFC|>0.5)")
    return pd.DataFrame({"gene_symbol": genes, "log2FC": lfc, "pvalue": pvals, "padj": padj})


def map_symbols_to_entrez(symbols: list) -> dict:
    cache = RAW / "symbol_to_entrez.csv"
    if cache.exists():
        df = pd.read_csv(cache, dtype=str)
        return dict(zip(df["SYMBOL"], df["ENTREZID"]))
    print("  Mapping symbols → Entrez (org.Hs.eg.db) ...")
    sym_f = Path(tempfile.mktemp(suffix=".txt")); out_f = Path(tempfile.mktemp(suffix=".csv"))
    sym_f.write_text("\n".join(symbols))
    r_code = f"""suppressPackageStartupMessages({{library(org.Hs.eg.db); library(AnnotationDbi)}})
syms <- readLines("{sym_f}")
res  <- AnnotationDbi::select(org.Hs.eg.db, keys=syms, columns="ENTREZID", keytype="SYMBOL")
res  <- res[!is.na(res$ENTREZID),]; res <- res[!duplicated(res$SYMBOL),]
write.csv(res, "{out_f}", row.names=FALSE)"""
    subprocess.run(["Rscript", "--vanilla", "-e", r_code], capture_output=True, check=True)
    df = pd.read_csv(out_f, dtype=str); df.to_csv(cache, index=False)
    sym_f.unlink(missing_ok=True); out_f.unlink(missing_ok=True)
    return dict(zip(df["SYMBOL"], df["ENTREZID"]))


def rwr(W: sp.csr_matrix, seed: np.ndarray) -> np.ndarray:
    p0 = np.abs(seed).astype(float)
    if p0.sum() == 0:
        return p0
    p0 /= p0.sum(); p = p0.copy()
    for _ in range(MAX_ITER):
        p_new = (1 - ALPHA) * W.dot(p) + ALPHA * p0
        if np.abs(p_new - p).sum() < TOL:
            return p_new
        p = p_new
    return p


def build_string_network() -> tuple[sp.csr_matrix, list]:
    ppi   = ROOT / "data/raw/string_human_ppi.txt.gz"
    info  = ROOT / "data/raw/string_human_info.txt.gz"
    print("  Building STRING network ...")
    edges = pd.read_csv(ppi, sep=" ", compression="gzip",
                        usecols=["protein1", "protein2", "combined_score"])
    edges = edges[edges["combined_score"] >= 700]
    sym_map = pd.read_csv(info, sep="\t", compression="gzip",
                          usecols=["#string_protein_id", "preferred_name"]
                         ).rename(columns={"#string_protein_id": "sid", "preferred_name": "sym"}).set_index("sid")["sym"]
    edges = edges.copy()
    edges["g1"] = edges["protein1"].map(sym_map); edges["g2"] = edges["protein2"].map(sym_map)
    edges = edges.dropna(subset=["g1", "g2"])
    sym2e = map_symbols_to_entrez(list(set(edges["g1"]) | set(edges["g2"])))
    edges["e1"] = edges["g1"].map(sym2e); edges["e2"] = edges["g2"].map(sym2e)
    edges = edges.dropna(subset=["e1", "e2"])
    nodes = sorted(set(edges["e1"]) | set(edges["e2"]))
    idx = {n: i for i, n in enumerate(nodes)}; N = len(nodes)
    r = edges["e1"].map(idx).values; c = edges["e2"].map(idx).values
    A = sp.csr_matrix((np.r_[np.ones(len(r)), np.ones(len(r))],
                       (np.r_[r, c], np.r_[c, r])), shape=(N, N))
    col_sums = np.array(A.sum(axis=0)).flatten(); col_sums[col_sums == 0] = 1
    print(f"  Network: {N:,} nodes, {len(edges):,} edges")
    return A.multiply(1.0 / col_sums), nodes


def pearson_reversal(net_vec: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Negated Pearson correlation of disease net vector vs each drug column."""
    d = (net_vec - net_vec.mean()) / (net_vec.std() + 1e-12)
    D_c = D - D.mean(axis=0); D_s = D.std(axis=0) + 1e-12
    return -(d @ D_c) / (len(net_vec) * D_s)


def cmap_ks(sig: np.ndarray, drug: np.ndarray, n_top: int = 150) -> float:
    n = len(sig); n_top = min(n_top, n // 4)
    ranked = np.argsort(sig)[::-1]
    up_q = set(ranked[:n_top]); dn_q = set(ranked[-n_top:])
    d_rank = np.argsort(drug)[::-1]
    def ks(q):
        hit = np.array([1 if i in q else 0 for i in d_rank])
        n_h = hit.sum()
        if n_h == 0: return 0.0
        dev = np.cumsum(hit) / n_h - np.cumsum(1 - hit) / max(n - n_h, 1)
        return float(dev[np.abs(dev).argmax()])
    ks_up = ks(up_q); ks_dn = ks(dn_q)
    return (ks_up - ks_dn) / 2 if np.sign(ks_up) != np.sign(ks_dn) else 0.0


def main() -> None:
    if (OUT / ".at2_at1_computed").exists():
        print("[skip] Signature already computed. Delete results/scrna/.at2_at1_computed to redo.")
        _report_candidates()
        return

    # ── Steps 1-4: Per-dataset DE ─────────────────────────────────────────────
    all_de: list[pd.DataFrame] = []
    sig_path = OUT / "at2_at1_transition_signature.csv"

    if not sig_path.exists():
        for acc, cfg in DATASETS.items():
            print(f"\n{'='*55}\n{acc}")
            ck_acc = OUT / f".{acc}_de_done"
            de_csv = OUT / f"{acc}_at2_at1_de.csv"

            if de_csv.exists() and ck_acc.exists():
                print(f"  [cached] DE for {acc}")
                all_de.append(pd.read_csv(de_csv))
                continue

            # Download files
            ok = all(download(url, dest) for url, dest in [cfg["metadata"]])
            if not ok:
                print(f"  WARNING: metadata download failed for {acc} — skipping")
                continue

            meta = load_metadata(cfg["metadata"][1], cfg["ct_col"], cfg["donor_col"],
                                 cfg["disease_col"], cfg["at2_label"], cfg["at1_label"])
            target_barcodes = set(meta[meta[cfg["ct_col"]].isin(
                [cfg["at2_label"], cfg["at1_label"]])].index)
            print(f"  Target cells (AT1+AT2): {len(target_barcodes):,}")

            # Download MTX files
            for key in ("barcodes", "genes", "mtx"):
                download(cfg[key][0], cfg[key][1])

            mat, genes, sub_barcodes = stream_mtx_subset(
                cfg["mtx"][1], cfg["barcodes"][1], cfg["genes"][1], target_barcodes)

            de = compute_pseudobulk_de(mat, genes, meta, sub_barcodes,
                                       cfg["ct_col"], cfg["donor_col"],
                                       cfg["at2_label"], cfg["at1_label"])
            de["dataset"] = acc
            de.to_csv(de_csv, index=False)
            ck_acc.touch()
            all_de.append(de)
            del mat  # free memory

        if not all_de:
            print("\nNo DE results — writing empty outputs.")
            for f in ["at2_at1_transition_signature.csv", "at2_at1_network_scores.csv",
                      "at2_at1_drug_scores.csv", "scrna_vs_bulk_comparison.csv"]:
                (OUT / f).write_text("drug,pearson,cmap\n")
            (OUT / "scrna_summary.txt").write_text("No expression data available.\n")
            return

        # IVW Z-score meta-analysis across datasets (same as IPF meta-analysis)
        combined = pd.concat(all_de, ignore_index=True)
        from scipy.special import ndtri
        from scipy.stats import norm
        records = []
        for gene, grp in combined.groupby("gene_symbol"):
            lfc = grp["log2FC"].values; p = grp["pvalue"].clip(lower=1e-300).values
            z   = np.abs(ndtri(p / 2)) * np.sign(lfc)
            se  = np.where(np.abs(z) > 0, np.abs(lfc) / np.abs(z), 1.0)
            w   = 1 / (se**2 + 1e-12)
            ml  = (w * lfc).sum() / w.sum()
            mz  = (w * z).sum() / np.sqrt((w**2).sum())
            records.append({"gene_symbol": gene, "log2FC": ml, "pvalue": float(2 * norm.sf(abs(mz))),
                            "n_datasets": len(grp)})
        pooled = pd.DataFrame(records)
        _, padj, _, _ = multipletests(pooled["pvalue"].fillna(1).values, method="fdr_bh")
        pooled["padj"] = padj

        # Map to Entrez
        print("\nMapping gene symbols → Entrez ...")
        sym2e = map_symbols_to_entrez(pooled["gene_symbol"].tolist())
        pooled["gene_id"] = pooled["gene_symbol"].map(sym2e)
        pooled = pooled.dropna(subset=["gene_id"])
        pooled = (pooled.loc[pooled.groupby("gene_id")["log2FC"].apply(lambda x: x.abs().idxmax())]
                  .set_index("gene_id"))
        pooled.to_csv(sig_path)
        print(f"Saved signature: {sig_path} ({len(pooled):,} genes, "
              f"{(pooled['padj']<0.05).sum():,} FDR<0.05)")
    else:
        print(f"[cached] Transition signature ({sig_path.name})")
        pooled = pd.read_csv(sig_path, index_col=0)

    # ── Step 7: RWR network propagation ───────────────────────────────────────
    net_path = OUT / "at2_at1_network_scores.csv"
    if not net_path.exists():
        print("\nRunning RWR network propagation ...")
        W, nodes = build_string_network()
        sig = pooled["log2FC"].astype(float); sig.index = sig.index.astype(str)
        seed = pd.Series(0.0, index=nodes)
        overlap = sig.index.intersection(pd.Index(nodes))
        seed[overlap] = sig.loc[overlap]
        print(f"  AT2→AT1 genes on network: {len(overlap):,}/{len(sig):,}")
        up_rwr = rwr(W, seed.clip(lower=0).values)
        dn_rwr = rwr(W, (-seed).clip(lower=0).values)
        pd.DataFrame({"rwr_up": up_rwr, "rwr_down": dn_rwr, "rwr_net": up_rwr - dn_rwr},
                     index=nodes).to_csv(net_path)
        print(f"Saved: {net_path}")
    else:
        print(f"[cached] Network scores ({net_path.name})")

    net_df = pd.read_csv(net_path, index_col=0)

    # ── Step 8: Drug scoring ───────────────────────────────────────────────────
    drug_path = OUT / "at2_at1_drug_scores.csv"
    if not drug_path.exists():
        print("\nScoring L1000 drugs ...")
        drug_mat = pd.read_csv(L1K / "drug_signatures_landmark.csv.gz", index_col=0)
        drug_mat.index = drug_mat.index.astype(str); net_df.index = net_df.index.astype(str)
        ovlp    = net_df.index.intersection(drug_mat.index)
        net_vec = net_df.loc[ovlp, "rwr_net"].values
        D_mat   = drug_mat.loc[ovlp].values.astype(float)
        pearson_s = pearson_reversal(net_vec, D_mat)
        cmap_s    = np.array([cmap_ks(net_vec, D_mat[:, j]) for j in range(D_mat.shape[1])])
        drug_df = pd.DataFrame({"pearson": pearson_s, "cmap": cmap_s}, index=drug_mat.columns)
        drug_df.to_csv(drug_path)
        print(f"Saved: {drug_path}")
    else:
        drug_df = pd.read_csv(drug_path, index_col=0)

    # ── Step 9: Compare to bulk ────────────────────────────────────────────────
    _report_candidates(drug_df, pooled)
    (OUT / ".at2_at1_computed").touch()
    print("\nC1 complete. Rerun B5 to include 'scRNA AT2→AT1 (Pearson)' arm.")


def _report_candidates(drug_df: pd.DataFrame = None, sig: pd.DataFrame = None) -> None:
    if drug_df is None:
        if not (OUT / "at2_at1_drug_scores.csv").exists():
            return
        drug_df = pd.read_csv(OUT / "at2_at1_drug_scores.csv", index_col=0)
    bulk = pd.read_csv(ROOT / "results/reversal/reversal_primary_ranking.csv").set_index("drug")
    merged = drug_df.join(bulk[["net_trace", "reversal_rank"]], how="inner")
    sc_rank = drug_df["pearson"].rank(ascending=False)
    from scipy.stats import spearmanr
    r, p = spearmanr(sc_rank.loc[merged.index], merged["reversal_rank"])
    merged["scrna_rank"] = sc_rank.loc[merged.index]
    merged.to_csv(OUT / "scrna_vs_bulk_comparison.csv")

    priority = ["romidepsin", "JNJ-26481585", "dasatinib", "atorvastatin", "nintedanib"]
    top20 = drug_df.sort_values("pearson", ascending=False).head(20)
    lines = [f"Spearman r (scRNA vs bulk Net-TRACE): {r:.3f}  p={p:.3e}",
             "", "Top-20 scRNA Pearson reversers:"] + [
             f"  {i+1:3}. {d} ({s:.4f})" for i, (d, s) in enumerate(top20["pearson"].items())] + [
             "", "Priority candidates in scRNA arm:"] + [
             f"  {d}: rank {int(sc_rank.get(d, float('nan')))}/{len(drug_df)}"
             for d in priority if d in sc_rank.index]
    txt = "\n".join(lines)
    (OUT / "scrna_summary.txt").write_text(txt)
    print("\n" + txt)


if __name__ == "__main__":
    main()
