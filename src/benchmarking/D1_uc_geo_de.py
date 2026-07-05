import gzip
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

ROOT   = Path(__file__).resolve().parents[2]
DATA   = ROOT / "data/raw/uc"
DE_OUT = ROOT / "results/de/uc"
META   = ROOT / "results/meta"
DATA.mkdir(parents=True, exist_ok=True)
DE_OUT.mkdir(parents=True, exist_ok=True)
META.mkdir(parents=True, exist_ok=True)

DATASETS = [
    {
        "acc":      "GSE87466",
        "url":      ("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE87nnn/"
                     "GSE87466/matrix/GSE87466_series_matrix.txt.gz"),
        "gpl":      "GPL13158",
        "uc_char":  "ulcerative colitis",
        "ctrl_char":"disease: normal",
        "title_kw": None,
    },
    {
        "acc":      "GSE59071",
        "url":      ("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE59nnn/"
                     "GSE59071/matrix/GSE59071_series_matrix.txt.gz"),
        "gpl":      "GPL6244",
        "uc_char":  None,
        "ctrl_char":None,
        "title_kw": ("UC_colon_active", "control_colon"),
    },
    {
        "acc":      "GSE38713",
        "url":      ("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE38nnn/"
                     "GSE38713/matrix/GSE38713_series_matrix.txt.gz"),
        "gpl":      "GPL570",
        "uc_char":  "disease extension: left",
        "ctrl_char":"disease extension: --",
        "title_kw": None,
    },
]

GPL_ANNOT_URLS = {
    "GPL570":   ("https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPLnnn/"
                 "GPL570/annot/GPL570.annot.gz"),
    "GPL13158": ("https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPL13nnn/"
                 "GPL13158/annot/GPL13158.annot.gz"),
    "GPL6244":  ("https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPL6nnn/"
                 "GPL6244/annot/GPL6244.annot.gz"),
}


def download(url: str, dest: Path) -> bool:
    if dest.exists():
        print(f"  [cached] {dest.name}")
        return True
    print(f"  Downloading {dest.name} ...", end=" ", flush=True)
    try:
        with urlopen(url, timeout=600) as r, open(dest, "wb") as f:
            f.write(r.read())
        print(f"done ({dest.stat().st_size / 1e6:.1f} MB)")
        return True
    except Exception as exc:
        print(f"FAILED: {exc}")
        return False


def parse_matrix(path: Path) -> tuple[pd.DataFrame, list[str], list[list[str]]]:
    titles: list[str] = []
    chars_per_sample: list[list[str]] = []
    char_lines: list[list[str]] = []
    rows: list[str] = []
    header: list[str] = []
    in_tab = False

    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if in_tab:
                if "series_matrix_table_end" in line:
                    break
                rows.append(line)
                continue
            if "ID_REF" in line:
                header = [v.strip('"') for v in line.split("\t")]
                in_tab = True
                continue
            if line.startswith("!Sample_title\t"):
                titles = [v.strip('"') for v in line.split("\t")[1:]]
            elif line.startswith("!Sample_characteristics_ch1\t"):
                vals = [v.strip('"').lower() for v in line.split("\t")[1:]]
                char_lines.append(vals)

    n = len(titles)
    chars_per_sample = [[] for _ in range(n)]
    for cline in char_lines:
        for i, val in enumerate(cline[:n]):
            chars_per_sample[i].append(val)

    df = (
        pd.DataFrame([r.split("\t") for r in rows], columns=header)
        .set_index("ID_REF")
    )
    df.index = df.index.str.strip('"')
    df = df.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    return df, titles, chars_per_sample


def classify(titles: list[str], chars_per_sample: list[list[str]],
             ds: dict) -> tuple[list[int], list[int]]:
    uc_idx, ctrl_idx = [], []
    if ds["title_kw"] is not None:
        uc_kw, ctrl_kw = ds["title_kw"]
        for i, t in enumerate(titles):
            if uc_kw.lower() in t.lower():
                uc_idx.append(i)
            elif ctrl_kw.lower() in t.lower():
                ctrl_idx.append(i)
    else:
        uc_kw   = ds["uc_char"].lower()
        ctrl_kw = ds["ctrl_char"].lower()
        for i, clist in enumerate(chars_per_sample):
            combined = " | ".join(clist)
            if uc_kw in combined:
                uc_idx.append(i)
            elif ctrl_kw in combined:
                ctrl_idx.append(i)
    return uc_idx, ctrl_idx


def load_probe_map(gpl: str) -> dict[str, int]:
    cache = DATA / f"{gpl}_entrez.csv"
    if cache.exists() and cache.stat().st_size > 100:
        gdf = pd.read_csv(cache, dtype=str)
    else:
        url = GPL_ANNOT_URLS.get(gpl)
        if not url:
            return {}
        annot_path = DATA / f"{gpl}.annot.gz"
        if not download(url, annot_path):
            return {}
        rows_a, hdr, in_t = [], [], False
        with gzip.open(annot_path, "rt", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if "platform_table_begin" in line.lower():
                    in_t = True
                    continue
                if "platform_table_end" in line.lower():
                    break
                if not in_t:
                    continue
                parts = line.split("\t")
                if not hdr:
                    hdr = parts
                    continue
                if len(parts) >= len(hdr):
                    rows_a.append(dict(zip(hdr, parts)))
        if not rows_a:
            print(f"  ERROR: {gpl}.annot.gz parsed 0 rows")
            return {}
        gdf = pd.DataFrame(rows_a)
        gdf.to_csv(cache, index=False)

    cu = {c.upper().strip(): c for c in gdf.columns}
    id_col = (cu.get("ID") or cu.get("PROBE SET ID") or cu.get("PROBE_SET_ID"))
    e_col  = (cu.get("ENTREZ_GENE_ID") or cu.get("ENTREZ GENE")
              or cu.get("ENTREZ_GENE") or cu.get("GENE_ID")
              or cu.get("GENE ID") or cu.get("GENE"))
    if not id_col or not e_col:
        print(f"  WARNING: missing Probe/Entrez columns in {gpl}; got {list(gdf.columns)[:8]}")
        return {}

    probe_map: dict[str, int] = {}
    for _, row in gdf.iterrows():
        try:
            probe_map[str(row[id_col]).strip()] = int(
                str(row[e_col]).strip().split("///")[0].strip()
            )
        except (ValueError, TypeError):
            pass
    print(f"  {gpl}: {len(probe_map):,} probe→Entrez mappings")
    return probe_map


def run_de(expr: pd.DataFrame, uc_idx: list[int],
           ctrl_idx: list[int], gpl: str, acc: str) -> pd.DataFrame:
    if expr.median().median() > 100:
        expr = np.log2(expr.clip(lower=1))

    uc_mat   = expr.iloc[:, uc_idx].values.astype(float)
    ctrl_mat = expr.iloc[:, ctrl_idx].values.astype(float)
    valid = (
        (np.isnan(uc_mat).mean(axis=1) < 0.5) &
        (np.isnan(ctrl_mat).mean(axis=1) < 0.5)
    )
    uc_f, ctrl_f = uc_mat[valid], ctrl_mat[valid]
    probes = expr.index[valid]

    t_stat, pvals = stats.ttest_ind(uc_f, ctrl_f, axis=1, nan_policy="omit")
    lfc           = np.nanmean(uc_f, axis=1) - np.nanmean(ctrl_f, axis=1)
    _, padj, _, _ = multipletests(np.nan_to_num(pvals, nan=1.0), method="fdr_bh")

    n_uc, n_ctrl = uc_f.shape[1], ctrl_f.shape[1]
    pooled_var   = (
        np.nanvar(uc_f, axis=1, ddof=1) / n_uc
        + np.nanvar(ctrl_f, axis=1, ddof=1) / n_ctrl
    )
    se = np.sqrt(np.where(pooled_var > 0, pooled_var, np.nan))

    de = pd.DataFrame({
        "probe":   probes,
        "log2FC":  lfc,
        "t_stat":  t_stat,
        "se":      se,
        "pvalue":  pvals,
        "padj":    padj,
        "n_uc":    n_uc,
        "n_ctrl":  n_ctrl,
    })

    probe_map = load_probe_map(gpl)
    if not probe_map:
        print(f"  WARNING {acc}: no probe map")
        return pd.DataFrame()
    de["entrez"] = de["probe"].map(probe_map)
    de = de.dropna(subset=["entrez"]).copy()
    de["entrez"] = de["entrez"].astype(int)

    best_idx = (
        de.groupby("entrez")["log2FC"]
        .apply(lambda x: x.abs().idxmax())
        .values
    )
    de = de.loc[best_idx].set_index("entrez")

    sig = ((de["padj"] < 0.05) & (de["log2FC"].abs() > 0.5)).sum()
    print(f"  {acc}: {len(de):,} genes, {sig:,} significant (padj<0.05, |LFC|>0.5), "
          f"n_uc={n_uc}, n_ctrl={n_ctrl}")
    return de


def meta_analyze(de_frames: list[tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    all_genes = sorted(
        set().union(*[set(df.index) for _, df in de_frames])
    )
    lfc_matrix = pd.DataFrame(index=all_genes)
    se_matrix  = pd.DataFrame(index=all_genes)
    n_matrix   = pd.DataFrame(index=all_genes)

    for acc, df in de_frames:
        lfc_matrix[acc] = df["log2FC"].reindex(all_genes)
        se_matrix[acc]  = df["se"].reindex(all_genes)
        n_matrix[acc]   = (df["n_uc"] + df["n_ctrl"]).reindex(all_genes)

    lfc_arr = lfc_matrix.values.astype(float)
    se_arr  = se_matrix.values.astype(float)
    n_arr   = n_matrix.values.astype(float)

    inv_var = np.where(se_arr > 0, 1.0 / se_arr**2, np.nan)
    w_sum   = np.nansum(inv_var, axis=1)
    meta_lfc = np.nansum(inv_var * lfc_arr, axis=1) / np.where(w_sum > 0, w_sum, np.nan)
    meta_se  = np.sqrt(1.0 / np.where(w_sum > 0, w_sum, np.nan))
    meta_z   = meta_lfc / meta_se

    pvals = 2 * stats.norm.sf(np.abs(meta_z))
    pvals = np.where(np.isnan(meta_z), 1.0, pvals)
    _, padj, _, _ = multipletests(pvals, method="fdr_bh")

    n_datasets    = np.sum(~np.isnan(lfc_arr), axis=1)
    n_concordant  = np.sum(lfc_arr * np.sign(meta_lfc)[:, None] > 0, axis=1)

    out = pd.DataFrame({
        "meta_log2FC":      meta_lfc,
        "meta_SE":          meta_se,
        "meta_z":           meta_z,
        "meta_pvalue":      pvals,
        "meta_padj":        padj,
        "n_datasets":       n_datasets,
        "n_concordant":     n_concordant,
        "frac_concordant":  np.where(n_datasets > 0, n_concordant / n_datasets, np.nan),
        "replicated":       (n_datasets >= 2) & (n_concordant / np.where(n_datasets > 0, n_datasets, 1) >= 0.5),
    }, index=all_genes)
    out.index.name = "entrez_id"
    return out.dropna(subset=["meta_log2FC"])


def main() -> None:
    de_frames = []

    for ds in DATASETS:
        acc = ds["acc"]
        print(f"\n{'='*60}")
        print(f"Processing {acc} ({ds['gpl']})")

        out_path = DE_OUT / f"{acc}_uc_de.csv"
        if out_path.exists():
            print(f"  [cached] {out_path.name}")
            de_frames.append((acc, pd.read_csv(out_path, index_col="entrez")))
            continue

        local = DATA / f"{acc}_series_matrix.txt.gz"
        if not download(ds["url"], local):
            print(f"  SKIP {acc}: download failed")
            continue

        print(f"  Parsing {acc} ...")
        expr, titles, chars = parse_matrix(local)
        print(f"  {expr.shape[0]:,} probes × {expr.shape[1]} samples")

        uc_idx, ctrl_idx = classify(titles, chars, ds)
        print(f"  UC: {len(uc_idx)}   Control: {len(ctrl_idx)}")
        if len(uc_idx) < 5 or len(ctrl_idx) < 3:
            print(f"  WARNING: too few samples — skipping")
            continue

        de = run_de(expr, uc_idx, ctrl_idx, ds["gpl"], acc)
        if de.empty:
            continue

        de.to_csv(out_path)
        de_frames.append((acc, de))

    if not de_frames:
        print("No datasets processed — exiting")
        return

    print(f"\n{'='*60}")
    print(f"Meta-analyzing {len(de_frames)} datasets ...")
    consensus = meta_analyze(de_frames)

    sig_up   = ((consensus["meta_padj"] < 0.05) & (consensus["meta_log2FC"] > 0)).sum()
    sig_down = ((consensus["meta_padj"] < 0.05) & (consensus["meta_log2FC"] < 0)).sum()
    print(f"  {len(consensus):,} genes in meta-analysis")
    print(f"  FDR<0.05: {sig_up} up, {sig_down} down")

    out = META / "uc_consensus_signature.csv"
    consensus.to_csv(out)
    print(f"  Saved → {out}")
    print("\nNext: python src/benchmarking/D2_uc_network_propagation.py")


if __name__ == "__main__":
    main()
