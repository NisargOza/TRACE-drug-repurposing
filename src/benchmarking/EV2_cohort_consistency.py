from pathlib import Path
import gzip
import re
import urllib.request

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT  = Path(__file__).resolve().parents[2]
DATA  = ROOT / "data/raw/GSE47460"
BENCH = ROOT / "results/benchmarking"
L1K   = ROOT / "results/l1000"
OUT   = ROOT / "results/external_validation"
ACT   = ROOT / "data/known_actives"
OUT.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)

GEO_FTP   = "https://ftp.ncbi.nlm.nih.gov/geo"
ACCESSION = "GSE47460"
PLATFORM  = "GPL14550"
SCORE_COL = "ipf_pearson"
TOP_KS    = [25, 50, 100, 200]


def _geo_prefix(acc: str) -> str:
    return acc[:-3] + "nnn"


def _download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  [cached] {dest.name}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {dest.name} ...", end=" ", flush=True)
    urllib.request.urlretrieve(url, dest)
    sz = dest.stat().st_size / 1e6
    print(f"done ({sz:.1f} MB)")


def download_data() -> tuple[Path, Path]:
    prefix = _geo_prefix(ACCESSION)
    matrix_url = (f"{GEO_FTP}/series/{prefix}/{ACCESSION}/matrix/"
                  f"{ACCESSION}-{PLATFORM}_series_matrix.txt.gz")
    matrix_dest = DATA / f"{ACCESSION}-{PLATFORM}_series_matrix.txt.gz"
    _download(matrix_url, matrix_dest)

    ann_url = (f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi"
               f"?acc={PLATFORM}&targ=self&form=text&view=data")
    ann_dest = DATA / f"{PLATFORM}_annotation.txt"
    _download(ann_url, ann_dest)

    return matrix_dest, ann_dest


def parse_series_matrix(path: Path) -> tuple[dict, pd.DataFrame]:
    samples: dict = {}
    expr_rows: list = []
    in_table = False
    header = None

    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("!Sample_geo_accession"):
                for g in re.findall(r'"([^"]+)"', line):
                    samples[g] = {}
            elif line.startswith("!Sample_title"):
                for g, t in zip(samples, re.findall(r'"([^"]+)"', line)):
                    samples[g]["title"] = t
            elif line.startswith("!Sample_characteristics_ch1"):
                vals = re.findall(r'"([^"]+)"', line)
                for g, v in zip(samples, vals):
                    samples[g].setdefault("chars", []).append(v)
            elif line == "!series_matrix_table_begin":
                in_table = True
            elif line == "!series_matrix_table_end":
                break
            elif in_table:
                parts = line.split("\t")
                if header is None:
                    header = [p.strip('"') for p in parts]
                else:
                    expr_rows.append(parts)

    expr = pd.DataFrame(expr_rows, columns=header)
    probe_col = header[0]
    expr = expr.rename(columns={probe_col: "probe_id"}).set_index("probe_id")
    expr.index = expr.index.str.strip('"')
    expr = expr.apply(pd.to_numeric, errors="coerce")
    return samples, expr


def parse_platform(path: Path) -> pd.Series:
    records: list = []
    in_table = False
    header = None

    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("!platform_table_begin"):
                in_table = True
                header = None
            elif line.startswith("!platform_table_end"):
                break
            elif in_table:
                parts = line.split("\t")
                if header is None:
                    header = parts
                else:
                    records.append(dict(zip(header, parts)))

    df = pd.DataFrame(records)
    if "ID" not in df.columns or "GENE" not in df.columns:
        raise ValueError(
            f"{PLATFORM} annotation missing ID or GENE columns; got: {list(df.columns)}"
        )

    df = df[["ID", "GENE"]].copy()
    df["GENE"] = df["GENE"].str.split("///").str[0].str.strip()
    df = df[(df["GENE"] != "") & df["GENE"].notna()]
    return df.set_index("ID")["GENE"]


def classify_samples(samples: dict) -> tuple[list, list]:
    ipf, ctrl = [], []
    for gsm, info in samples.items():
        text = " ".join(info.get("chars", [])).lower()
        if "interstitial lung disease" in text:
            ipf.append(gsm)
        elif "disease state: control" in text:
            ctrl.append(gsm)
    return ipf, ctrl


def compute_de(expr: pd.DataFrame,
               ipf_cols: list, ctrl_cols: list) -> pd.DataFrame:
    ipf_mat  = expr[[c for c in ipf_cols  if c in expr.columns]].values
    ctrl_mat = expr[[c for c in ctrl_cols if c in expr.columns]].values

    log2fc = np.nanmean(ipf_mat, axis=1) - np.nanmean(ctrl_mat, axis=1)
    t_vals, p_vals = [], []
    for i in range(len(expr)):
        a = ipf_mat[i][~np.isnan(ipf_mat[i])]
        b = ctrl_mat[i][~np.isnan(ctrl_mat[i])]
        if len(a) >= 3 and len(b) >= 3:
            t, p = stats.ttest_ind(a, b, equal_var=False)
            t_vals.append(float(t)); p_vals.append(float(p))
        else:
            t_vals.append(np.nan); p_vals.append(np.nan)

    de = pd.DataFrame({"log2FC": log2fc, "t_stat": t_vals, "pvalue": p_vals},
                      index=expr.index).dropna()
    _, padj, _, _ = multipletests(de["pvalue"].values, method="fdr_bh")
    de["padj"] = padj
    return de


def probes_to_genes(de: pd.DataFrame, probe_map: pd.Series) -> pd.DataFrame:
    de = de.copy()
    de["entrez"] = probe_map.reindex(de.index)
    de = de.dropna(subset=["entrez"])
    de["abs_t"] = de["t_stat"].abs()
    gene_de = (de.sort_values("abs_t", ascending=False)
                 .groupby("entrez")
                 .first()
                 .reset_index()
                 .rename(columns={"entrez": "gene_id"})
                 .set_index("gene_id"))
    return gene_de


def compute_reversal_scores(gene_de: pd.DataFrame,
                            drug_mat: pd.DataFrame) -> pd.Series:
    common = gene_de.index.intersection(drug_mat.index)
    print(f"  Landmark genes in held-out signature: {len(common)}/978")
    sig = gene_de.loc[common, "log2FC"].values.astype(float)
    sig = (sig - sig.mean()) / (sig.std() + 1e-8)
    mat = drug_mat.loc[common].values.astype(float)

    scores = []
    for j in range(mat.shape[1]):
        col = mat[:, j]
        ok = ~np.isnan(col)
        if ok.sum() >= 50:
            r, _ = stats.pearsonr(sig[ok], col[ok])
            scores.append(float(-r))
        else:
            scores.append(np.nan)
    return pd.Series(scores, index=drug_mat.columns)


def jaccard_top_k(a: pd.Series, b: pd.Series, k: int) -> float:
    top_a = set(a.nlargest(k).index)
    top_b = set(b.nlargest(k).index)
    return len(top_a & top_b) / len(top_a | top_b)


def make_figure(orig: pd.Series, held: pd.Series,
                actives: set, spearman_r: float, spearman_p: float) -> None:
    both = pd.DataFrame({"original": orig, "heldout": held}).dropna()

    labels = both.index.isin(actives).astype(int)
    has_auroc = labels.sum() >= 2

    ncols = 3 if has_auroc else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4.5), dpi=150)
    fig.patch.set_facecolor("white")

    ax = axes[0]
    ax.scatter(both["original"], both["heldout"],
               s=6, alpha=0.4, linewidths=0, color="#4a7fb5", rasterized=True)
    for drug in actives:
        if drug in both.index:
            ax.scatter(both.loc[drug, "original"], both.loc[drug, "heldout"],
                       s=80, marker="*", color="#d7191c", zorder=5, linewidths=0)
            ax.annotate(drug, (both.loc[drug, "original"], both.loc[drug, "heldout"]),
                        fontsize=7, color="#d7191c",
                        xytext=(5, 3), textcoords="offset points")
    ax.set_xlabel("TRACE score — original meta-analysis", fontsize=9)
    ax.set_ylabel("TRACE score — held-out GSE47460", fontsize=9)
    ax.set_title(f"Cross-cohort drug score correlation\nSpearman r = {spearman_r:.3f}  (p = {spearman_p:.2e})",
                 fontsize=9.5, fontweight="bold")
    ax.set_facecolor("#f8f9fa")
    for sp in ax.spines.values(): sp.set_edgecolor("#ddd")

    ax = axes[1]
    ks = TOP_KS
    jaccards  = [jaccard_top_k(orig, held, k) for k in ks]
    random_j  = [k / (2 * len(both) - k) for k in ks]
    x = np.arange(len(ks))
    w = 0.35
    ax.bar(x - w/2, jaccards,  width=w, color="#2c7bb6", label="Observed", alpha=0.85)
    ax.bar(x + w/2, random_j,  width=w, color="#aaa",    label="Random",   alpha=0.70)
    ax.set_xticks(x); ax.set_xticklabels([f"Top-{k}" for k in ks], fontsize=8.5)
    ax.set_ylabel("Jaccard index", fontsize=9)
    ax.set_title("Top-K ranking overlap\n(original vs held-out)", fontsize=9.5, fontweight="bold")
    ax.legend(fontsize=8, framealpha=0.8)
    ax.set_facecolor("#f8f9fa")
    for sp in ax.spines.values(): sp.set_edgecolor("#ddd")

    if has_auroc:
        from sklearn.metrics import roc_curve, roc_auc_score
        ax = axes[2]
        fpr, tpr, _ = roc_curve(labels, both["heldout"].values)
        auroc = roc_auc_score(labels, both["heldout"].values)
        ax.plot(fpr, tpr, color="#d7191c", lw=2, label=f"AUROC = {auroc:.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4)
        ax.set_xlabel("False positive rate", fontsize=9)
        ax.set_ylabel("True positive rate", fontsize=9)
        ax.set_title("ROC — held-out scores vs\nknown IPF actives", fontsize=9.5, fontweight="bold")
        ax.legend(fontsize=8, framealpha=0.8)
        ax.set_facecolor("#f8f9fa")
        for sp in ax.spines.values(): sp.set_edgecolor("#ddd")

    plt.suptitle("External Validation — Cross-Cohort Consistency (GSE47460 LGRC, held-out)",
                 fontsize=11, fontweight="bold", y=1.02)
    plt.tight_layout(pad=1.2)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"EV2_cohort_consistency.{ext}",
                    dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Figure saved → results/external_validation/EV2_cohort_consistency.png")


def main() -> None:
    print(f"Step 1: Downloading {ACCESSION} (LGRC, held-out cohort) ...")
    matrix_path, platform_path = download_data()

    print("\nStep 2: Parsing series matrix ...")
    samples, expr = parse_series_matrix(matrix_path)
    print(f"  {expr.shape[0]} probes × {expr.shape[1]} samples")

    print("\nStep 3: Classifying samples ...")
    ipf_cols, ctrl_cols = classify_samples(samples)
    print(f"  IPF: {len(ipf_cols)}   Control: {len(ctrl_cols)}")
    if len(ipf_cols) < 5 or len(ctrl_cols) < 5:
        raise RuntimeError("Too few samples classified — check series matrix parsing")

    print(f"\nStep 4: Parsing {PLATFORM} platform annotation ...")
    probe_map = parse_platform(platform_path)
    print(f"  {len(probe_map)} probes with Entrez IDs")

    print("\nStep 5: Differential expression (Welch t-test per probe) ...")
    de = compute_de(expr, ipf_cols, ctrl_cols)
    n_sig = (de["padj"] < 0.05).sum()
    print(f"  {len(de)} probes tested, {n_sig} FDR < 0.05")

    print("\nStep 6: Collapsing probes to Entrez gene IDs ...")
    gene_de = probes_to_genes(de, probe_map)
    print(f"  {len(gene_de)} unique genes in held-out DE signature")

    print("\nStep 7: Loading L1000 landmark drug matrix ...")
    drug_mat = pd.read_csv(L1K / "drug_signatures_landmark.csv.gz", index_col=0)
    drug_mat.index = drug_mat.index.astype(str)
    gene_de.index  = gene_de.index.astype(str)
    print(f"  Drug matrix: {drug_mat.shape[0]} genes × {drug_mat.shape[1]} drugs")

    print("\nStep 8: Computing held-out reversal scores ...")
    held_scores = compute_reversal_scores(gene_de, drug_mat)
    held_scores = held_scores.dropna()

    print("\nStep 9: Loading original TRACE scores ...")
    orig_df = pd.read_csv(BENCH / "dual_disease_scores.csv", index_col="drug")
    orig_scores = orig_df[SCORE_COL].rename(lambda x: x.lower().strip())
    held_scores.index = held_scores.index.str.lower().str.strip()

    both = pd.DataFrame({"original": orig_scores, "heldout": held_scores}).dropna()
    print(f"  Matched drugs: {len(both)}")

    r, p = stats.spearmanr(both["original"], both["heldout"])
    print(f"\n── Cross-cohort consistency ──────────────────────────────────")
    print(f"  Spearman r = {r:.4f}  (p = {p:.2e})")
    for k in TOP_KS:
        j = jaccard_top_k(both["original"], both["heldout"], k)
        rand_j = k / (2 * len(both) - k)
        print(f"  Jaccard Top-{k:<4d}: {j:.4f}  ({j/rand_j:.2f}× random baseline)")

    actives = {l.strip().lower() for l in (ACT / "ipf_preclinical_actives.txt").read_text().splitlines() if l.strip()}

    out_df = both.copy()
    out_df["is_known_active"] = out_df.index.isin(actives)
    out_df.to_csv(OUT / "EV2_drug_scores.csv")
    pd.DataFrame([{
        "cohort": "GSE47460 (LGRC)",
        "n_ipf": len(ipf_cols),
        "n_ctrl": len(ctrl_cols),
        "n_matched_drugs": len(both),
        "spearman_r": round(r, 4),
        "spearman_p": float(p),
        "jaccard_top25":  round(jaccard_top_k(both["original"], both["heldout"], 25),  4),
        "jaccard_top50":  round(jaccard_top_k(both["original"], both["heldout"], 50),  4),
        "jaccard_top100": round(jaccard_top_k(both["original"], both["heldout"], 100), 4),
        "jaccard_top200": round(jaccard_top_k(both["original"], both["heldout"], 200), 4),
    }]).to_csv(OUT / "EV2_summary.csv", index=False)

    print("\nGenerating figure ...")
    make_figure(both["original"], both["heldout"], actives, r, p)
    print(f"\nEV2 complete. Outputs → {OUT}")


if __name__ == "__main__":
    main()
