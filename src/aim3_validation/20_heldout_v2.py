"""
IMPROVE item 3: Proper held-out case/control validation using GSE47460.

GSE47460 = Lung Genomics Research Consortium (LGRC), 430 samples,
whole-lung homogenate, Agilent GPL14550 microarray.
Sample titles encode disease: LT*_CTRL (controls), LT*_ILD (ILD including IPF/UIP).

Validates that the consensus IPF signature's direction replicates in this
completely held-out dataset (not used in DE or meta-analysis).

Writes:
  results/aim3/heldout_v2_de.csv
  results/aim3/heldout_v2_concordance.txt
  results/figures/fig_heldout_v2.png
"""

from pathlib import Path
import gzip
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests
from urllib.request import urlopen

ROOT  = Path(__file__).resolve().parents[2]
DATA  = ROOT / "data"
AIM3  = ROOT / "results" / "aim3"
META  = ROOT / "results" / "meta"
OUT   = ROOT / "results" / "figures"
DATA.mkdir(exist_ok=True)
AIM3.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

MATRIX_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE47nnn/GSE47460/matrix/"
    "GSE47460-GPL14550_series_matrix.txt.gz"
)
LOCAL = DATA / "GSE47460_series_matrix.txt.gz"


def ensure_downloaded():
    if LOCAL.exists():
        print(f"Already downloaded: {LOCAL}")
        return True
    print(f"Downloading GSE47460...")
    try:
        with urlopen(MATRIX_URL, timeout=120) as r, open(LOCAL, "wb") as f:
            f.write(r.read())
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        return False


def parse_matrix(path: Path):
    """Parse GEO series matrix; return (expr DataFrame, sample_titles list)."""
    meta_rows: dict[str, list] = {}
    data_rows: list[str] = []
    header: list[str] = []
    in_table = False

    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("!") and not in_table:
                parts = line.split("\t")
                key   = parts[0].lstrip("!").strip()
                vals  = [v.strip('"') for v in parts[1:]]
                meta_rows.setdefault(key, []).extend(vals)
            elif (line.startswith('"ID_REF"') or line.startswith("ID_REF")) and not in_table:
                header = [v.strip('"') for v in line.split("\t")]
                in_table = True
            elif in_table:
                if "series_matrix_table_end" in line:
                    break
                data_rows.append(line)

    if not header or not data_rows:
        return None, []

    df = pd.DataFrame(
        [r.split("\t") for r in data_rows],
        columns=header,
    ).set_index("ID_REF")
    # Strip quotes from probe ID index
    df.index = df.index.str.strip('"')
    df = df.apply(pd.to_numeric, errors="coerce").dropna(how="all")

    titles = meta_rows.get("Sample_title", [])
    return df, titles


def classify(titles: list[str]) -> tuple[list[int], list[int]]:
    """Classify samples by title suffix: _CTRL → control, _ILD → ILD."""
    ipf_idx  = [i for i, t in enumerate(titles) if "_ild"  in t.lower()]
    ctrl_idx = [i for i, t in enumerate(titles) if "_ctrl" in t.lower()]
    return ipf_idx, ctrl_idx


def main():
    if not ensure_downloaded():
        (AIM3 / "heldout_v2_concordance.txt").write_text(
            "GSE47460 download failed — held-out v2 pending."
        )
        return

    print("Parsing series matrix...")
    expr, titles = parse_matrix(LOCAL)
    if expr is None:
        print("Parse failed"); return

    print(f"Expression: {expr.shape}  (probes × samples)")
    print(f"Titles (first 5): {titles[:5]}")

    # Align sample columns with titles
    n_cols = expr.shape[1]
    # Sometimes titles list has a different count; trim to match
    if len(titles) > n_cols:
        titles = titles[:n_cols]
    elif len(titles) < n_cols:
        titles = titles + ["unknown"] * (n_cols - len(titles))

    ipf_idx, ctrl_idx = classify(titles)
    print(f"ILD samples: {len(ipf_idx)},  Control samples: {len(ctrl_idx)}")

    if not ipf_idx or not ctrl_idx:
        (AIM3 / "heldout_v2_concordance.txt").write_text(
            f"Sample classification failed.\nTitles[:5]: {titles[:5]}"
        )
        return

    # ── Differential expression ────────────────────────────────────────────────
    ild_mat  = expr.iloc[:, ipf_idx].values.astype(float)
    ctrl_mat = expr.iloc[:, ctrl_idx].values.astype(float)

    valid = (np.isnan(ild_mat).mean(1) < 0.5) & (np.isnan(ctrl_mat).mean(1) < 0.5)
    ild_f  = ild_mat[valid];  ctrl_f = ctrl_mat[valid]
    probes = expr.index[valid]

    t_stat, pvals = stats.ttest_ind(ild_f, ctrl_f, axis=1, nan_policy="omit")
    logfc = np.nanmean(ild_f, 1) - np.nanmean(ctrl_f, 1)
    _, padj, _, _ = multipletests(np.nan_to_num(pvals, nan=1.0), method="fdr_bh")

    de = pd.DataFrame({"probe": probes, "logFC": logfc, "pval": pvals, "padj": padj})
    de.to_csv(AIM3 / "heldout_v2_de.csv", index=False)
    print(f"DE: {(padj < 0.05).sum()} sig probes (padj<0.05)")

    # ── Map GPL14550 probe IDs → Entrez gene IDs ──────────────────────────────
    # Download GPL14550 SOFT file and extract GENE_ID column
    gpl_path = DATA / "GPL14550_annotation.csv"
    probe2entrez: dict[str, str] = {}
    if not gpl_path.exists():
        print("Downloading GPL14550 annotation (streaming)...")
        try:
            gpl_url = (
                "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPL14nnn/"
                "GPL14550/soft/GPL14550_family.soft.gz"
            )
            import csv, io
            records = []
            header_row: list[str] = []
            in_tab = False

            with urlopen(gpl_url, timeout=300) as resp:
                with gzip.GzipFile(fileobj=resp) as gz:
                    text_stream = io.TextIOWrapper(gz, encoding="utf-8", errors="replace")
                    for line in text_stream:
                        line = line.rstrip("\n")
                        if "platform_table_begin" in line:
                            in_tab = True
                            continue
                        if "platform_table_end" in line:
                            break
                        if not in_tab:
                            continue
                        if not header_row:
                            header_row = line.split("\t")
                            continue
                        parts = line.split("\t")
                        if len(parts) >= len(header_row):
                            records.append(dict(zip(header_row, parts)))

            if records:
                pd.DataFrame(records).to_csv(gpl_path, index=False)
                print(f"Saved {len(records)} probe annotations → {gpl_path}")
            else:
                print("No records parsed from GPL file")
        except Exception as e:
            print(f"GPL14550 download failed: {e}")

    if gpl_path.exists():
        gpl = pd.read_csv(gpl_path, dtype=str)
        id_col = next((c for c in gpl.columns if c in ("ID","PROBE_ID","ID_REF")), None)
        # Use exact-match priority: ENTREZ_GENE_ID > GENE_ID > GENE
        col_upper = {c.upper(): c for c in gpl.columns}
        entrez_col = (
            col_upper.get("ENTREZ_GENE_ID")
            or col_upper.get("GENE_ID")
            or col_upper.get("GENE")
            or next((c for c in gpl.columns if "ENTREZ" in c.upper()), None)
        )
        if id_col and entrez_col:
            for _, row in gpl.iterrows():
                pid   = str(row[id_col]).strip()
                eids  = str(row[entrez_col]).strip()
                # Take first Entrez ID (some probes map to multiple genes)
                first = eids.split("///")[0].strip().split(";")[0].strip()
                if first and first not in ("", "nan", "---"):
                    probe2entrez[pid] = first
            print(f"Probe→Entrez map: {len(probe2entrez)} entries")

    # ── Concordance with consensus ─────────────────────────────────────────────
    cons = pd.read_csv(META / "consensus_signature.csv", index_col=0)

    # Build integer-keyed probe→Entrez mapping
    p2e_int: dict[str, int] = {}
    for k, v in probe2entrez.items():
        try:
            p2e_int[str(k).strip()] = int(str(v).strip())
        except (ValueError, TypeError):
            pass
    print(f"p2e_int size: {len(p2e_int)}")

    # Map and deduplicate
    eid_list = [p2e_int.get(str(p), None) for p in de["probe"]]
    de2 = de.copy()
    de2["eid"] = eid_list
    de2 = de2.dropna(subset=["eid"]).copy()
    de2["eid"] = de2["eid"].astype(int)
    print(f"de2 mapped: {len(de2)}")
    # Deduplicate: keep max |logFC| per gene
    best_idx = de2.groupby("eid")["logFC"].apply(lambda x: x.abs().idxmax()).values
    de_mapped = de2.loc[best_idx].copy()
    de_mapped.index = de_mapped["eid"]
    print(f"de_mapped after dedup: {len(de_mapped)}")

    print(f"de_mapped size: {len(de_mapped)}, de_mapped.index sample: {list(de_mapped.index[:3])}")
    print(f"cons.index sample: {list(cons.index[:3])}, cons.index dtype: {cons.index.dtype}")
    overlap_genes = set(de_mapped.index.tolist()) & set(cons.index.tolist())

    if len(overlap_genes) < 50:
        msg = (
            f"Held-out validation (GSE47460)\n"
            f"Expression: {expr.shape}\n"
            f"ILD samples: {len(ipf_idx)}, Controls: {len(ctrl_idx)}\n"
            f"DE sig probes: {(padj < 0.05).sum()}\n"
            f"Probe-Entrez overlap: {len(overlap_genes)} (too small for concordance — "
            f"GPL14550 probe map needed for full Entrez mapping)\n"
            f"\nNote: The {(padj < 0.05).sum()} differentially expressed probes confirm "
            f"a significant ILD vs. control transcriptomic difference exists in this "
            f"independent dataset. Probe→Entrez mapping required for direction concordance "
            f"with the consensus signature."
        )
        (AIM3 / "heldout_v2_concordance.txt").write_text(msg)
        print(msg)
        return

    # Concordance computation
    common = list(overlap_genes)
    cons_lfc = cons.loc[common, "meta_log2FC"].values.astype(float)
    held_lfc = de_mapped.loc[common, "logFC"].values.astype(float)
    print(f"Concordance on {len(common)} genes")

    up_c   = cons_lfc > 0
    up_h   = held_lfc > 0
    n_conc = (up_c == up_h).sum()
    pct    = n_conc / len(common) * 100

    # Replicated consensus only
    if "replicated" in cons.columns:
        rep_mask = cons.loc[common, "replicated"].values
    elif "meta_padj" in cons.columns:
        rep_mask = cons.loc[common, "meta_padj"].values < 0.05
    else:
        rep_mask = np.ones(len(common), dtype=bool)

    n_conc_rep = (up_c[rep_mask] == up_h[rep_mask]).sum()
    pct_rep    = n_conc_rep / max(rep_mask.sum(), 1) * 100

    lines = [
        "Held-out Validation v2 — GSE47460 (LGRC dataset)",
        "=" * 60,
        f"Expression:        {expr.shape[0]} probes × {expr.shape[1]} samples",
        f"ILD samples:       {len(ipf_idx)}",
        f"Control samples:   {len(ctrl_idx)}",
        f"DE sig probes:     {(padj < 0.05).sum()}",
        f"Probe–Entrez overlap: {len(common)}",
        f"",
        f"Direction concordance (all overlap): {n_conc}/{len(common)} = {pct:.1f}%",
        f"Direction concordance (sig consensus): {n_conc_rep}/{rep_mask.sum()} = {pct_rep:.1f}%",
        f"",
        f"Random expectation: 50%",
        f"Result: {'PASS (>65%)' if pct > 65 else 'MODEST' if pct > 55 else 'FAIL/POOR'}",
    ]
    (AIM3 / "heldout_v2_concordance.txt").write_text("\n".join(lines))
    print("\n".join(lines))

    # ── Figure ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor("#f9f9f9"); ax.set_facecolor("#f9f9f9")
    ax.scatter(cons_lfc, held_lfc, alpha=0.3, s=6, color="#2166ac")
    ax.axhline(0, color="#aaa", lw=0.8); ax.axvline(0, color="#aaa", lw=0.8)
    ax.set_xlabel("Consensus meta-LFC (training datasets)", fontsize=10)
    ax.set_ylabel("Held-out LFC (GSE47460, LGRC)", fontsize=10)
    ax.set_title(f"Held-out direction concordance: {pct:.0f}%\n(n={len(common)} genes)", fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT / "fig_heldout_v2.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved fig_heldout_v2.png")


if __name__ == "__main__":
    main()
