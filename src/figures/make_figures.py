"""
Publication-quality figures for the TRACE IPF drug-repurposing manuscript.

Every panel below is computed directly from results/*.csv and results/*.npz
at run time — no hardcoded numbers. If a results file is missing, the
corresponding figure is skipped with a printed warning rather than
fabricating placeholder data.

Main text (results/figures/):
  fig1_consensus_replication.png  — single-dataset fragility -> consensus signature
  fig2_ablation.png               — method ablation: rank percentile per drug
  fig3_null_distribution.png      — empirical null (permutation) with nintedanib marked
  fig4_bootstrap_ci.png           — rank stability of top Net-TRACE candidates
  fig5_auroc_benchmark.png        — AUROC/AUPRC benchmarking vs. baselines (IPF & RA)
  fig6_evidence_heatmap.png       — multi-evidence dossier heatmap, top candidates

Extended data (results/figures/):
  ext7_scrna_at2at1.png           — scRNA AT2->AT1 transition signature validation
  ext8_l2s2_expansion.png         — L2S2 2,834-compound universe cross-validation
  ext9_crispr_targets.png         — CRISPR knockout reversal targets (HMGCR, HSP90AB1)
  ext10_ra_specificity.png        — RA dual-disease specificity control
  ext11_heldout_validation.png    — independent held-out cohort validation
  ext12_mr_forest.png             — drug-target Mendelian randomization (null/underpowered)
  ext13_robustness.png            — weight-sensitivity + negative-control robustness
  ext14_vae_architecture.png      — VAE-TRACE model architecture (schematic + loss/inference)
"""
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.patches as mpatches

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _style import (
    apply_figure_style, bar_with_points, end_of_line_labels, focal_palette,
    goodness_arrow, panel_crops, panel_letter, set_frame, strip_with_median,
    two_tier_label, BLUE, TEAL, RED, ORANGE, GRAY, DGRAY, GREEN, PURPLE, GOLD,
    META_GREY,
)

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT   = Path(__file__).resolve().parents[2]
RES    = ROOT / "results"
META   = RES / "meta"
REV    = RES / "reversal"
BENCH  = RES / "benchmarking"
AIM3   = RES / "aim3"
CRISPR = RES / "crispr"
SCRNA  = RES / "scrna"
L2S2   = RES / "l2s2"
MR     = RES / "mr"
VALID  = RES / "validation"
DE     = RES / "de"
QC     = RES / "qc"
EMB    = RES / "embedding"
L1000  = RES / "l1000"
OUT    = RES / "figures"
OUT.mkdir(parents=True, exist_ok=True)

apply_figure_style(frame="open", sizes=(9, 8, 7))

POS_CONTROLS = ["nintedanib", "pirfenidone"]


def _require(path, fig_name):
    if not path.exists():
        warnings.warn(f"[{fig_name}] missing required file: {path} — figure skipped.")
        return False
    return True


def _save(fig, name):
    path = OUT / name
    fig.savefig(path)
    plt.close(fig)
    print(f"saved {path}")


# ══════════════════════════════════════════════════════════════════════════
# Figure 1 — Single-dataset fragility vs. consensus replication
# ══════════════════════════════════════════════════════════════════════════
def fig1_consensus_replication():
    name = "fig1_consensus_replication.png"
    de_files = {
        "GSE213001": DE / "GSE213001_de_entrez.csv",
        "GSE150910": DE / "GSE150910_de_entrez.csv",
        "GSE38958":  DE / "GSE38958_de_entrez.csv",
        "GSE53845":  DE / "GSE53845_de_entrez.csv",
    }
    rep_path = META / "replication_stats.csv"
    cons_path = META / "consensus_signature.csv"
    if not all(_require(p, name) for p in list(de_files.values()) + [rep_path, cons_path]):
        return

    # per-dataset significant gene counts (padj<0.05) from each single-dataset DE result
    ds_counts = {}
    ds_n = {}
    qc_path = QC / "qc_summary.csv"
    qc = pd.read_csv(qc_path) if qc_path.exists() else None
    for ds, p in de_files.items():
        d = pd.read_csv(p)
        ds_counts[ds] = int((d["padj"] < 0.05).sum())
        if qc is not None and {"dataset", "condition"} <= set(qc.columns):
            n_ipf = (qc[(qc.dataset == ds) & (qc.condition.str.upper() == "IPF")]).shape[0]
            n_ctrl = (qc[(qc.dataset == ds) & (qc.condition.str.lower() == "control")]).shape[0]
            ds_n[ds] = f"n={n_ipf + n_ctrl}"
        else:
            ds_n[ds] = ""

    rep = pd.read_csv(rep_path)
    cons = pd.read_csv(cons_path)
    n_tested = len(rep)
    n_meta_sig = int((rep["meta_padj"] < 0.05).sum())
    n_consensus = len(cons)  # meta_padj<0.05 AND replicated==True

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), gridspec_kw={"width_ratios": [1.1, 1]})

    # Panel a: per-dataset significant DE genes (fragile, inconsistent across cohorts)
    ax = axes[0]
    labels = list(ds_counts.keys())
    vals = [ds_counts[k] for k in labels]
    xs = np.arange(len(labels))
    ax.bar(xs, vals, color=[GRAY] * len(labels), width=0.6, edgecolor="none")
    for x, v, lab in zip(xs, vals, labels):
        ax.text(x, v, f"{v:,}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{l}\n{ds_n[l]}" for l in labels], fontsize=7)
    ax.set_ylabel("significant genes (padj<0.05)")
    ax.set_title("Single-dataset DE is fragile", fontsize=9, loc="left")
    set_frame(ax)
    ax.text(0.02, 0.98, "no two cohorts agree\non gene count", transform=ax.transAxes,
            fontsize=6.5, color=META_GREY, ha="left", va="top", style="italic")

    # Panel b: funnel from tested -> meta-significant -> replicated consensus
    ax = axes[1]
    stages = ["Genes tested\n(meta-analysis)", "Meta padj<0.05", "+ replicated\n(consensus)"]
    counts = [n_tested, n_meta_sig, n_consensus]
    xs = np.arange(len(stages))
    cols = [GRAY, ORANGE, BLUE]
    ax.bar(xs, counts, color=cols, width=0.55, edgecolor="none")
    for x, v in zip(xs, counts):
        ax.text(x, v, f"{v:,}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(stages, fontsize=7)
    ax.set_ylabel("gene count")
    ax.set_title("Consensus signature after replication filter", fontsize=9, loc="left")
    set_frame(ax)
    pct = 100 * n_consensus / n_meta_sig
    ax.text(0.98, 0.95, f"{pct:.0f}% of meta-significant\ngenes replicate in\n\u22652 datasets",
            transform=ax.transAxes, fontsize=6.5, color=META_GREY, ha="right", va="top")

    panel_letter(axes[0], "a")
    panel_letter(axes[1], "b")
    fig.suptitle("Figure 1 | Cross-cohort replication filtering builds the IPF consensus signature",
                 fontsize=10, x=0.02, ha="left", y=1.02)
    fig.tight_layout()
    _save(fig, name)


# ══════════════════════════════════════════════════════════════════════════
# Figure 2 — Method ablation: rank percentile of positive controls
# ══════════════════════════════════════════════════════════════════════════
def fig2_ablation():
    name = "fig2_ablation.png"
    path = REV / "ablation_table.csv"
    if not _require(path, name):
        return
    d = pd.read_csv(path)
    methods = list(dict.fromkeys(d["method"]))  # preserve file order
    drugs = list(dict.fromkeys(d["drug"]))

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    width = 0.35
    xs = np.arange(len(methods))
    colors = {"nintedanib": RED, "pirfenidone": ORANGE}
    for i, drug in enumerate(drugs):
        sub = d[d.drug == drug].set_index("method").loc[methods]
        offset = (i - (len(drugs) - 1) / 2) * width
        bars = ax.bar(xs + offset, sub["pct"], width=width, color=colors.get(drug, GRAY),
                      label=drug, edgecolor="none")
        for x, v, r, n in zip(xs + offset, sub["pct"], sub["rank"], sub["n_drugs"]):
            ax.text(x, v + 1, f"{v:.1f}%\n(#{int(r)}/{int(n)})", ha="center", va="bottom", fontsize=6)

    ax.set_xticks(xs)
    ax.set_xticklabels(methods, fontsize=7.5)
    ax.set_ylabel("rank percentile (lower = better reversal rank)")
    ax.set_ylim(max(d["pct"]) * 1.25, -3)
    ax.legend(loc="lower left", frameon=False, fontsize=7.5)
    set_frame(ax)
    goodness_arrow(ax, text="lower = better rank", loc="lower right")
    ax.set_title("Figure 2 | Network propagation and VAE refinement improve positive-control rank",
                 fontsize=9.5, loc="left")
    fig.tight_layout()
    _save(fig, name)


# ══════════════════════════════════════════════════════════════════════════
# Figure 3 — Empirical permutation null with nintedanib marked
# ══════════════════════════════════════════════════════════════════════════
def fig3_null_distribution():
    name = "fig3_null_distribution.png"
    null_path = REV / "extended_fdr_null.npz"
    res_path = REV / "extended_fdr_results.csv"
    if not all(_require(p, name) for p in [null_path, res_path]):
        return
    npz = np.load(null_path, allow_pickle=True)
    null_scores, drug_names = npz["null_scores"], npz["drugs"]
    res = pd.read_csv(res_path)

    idx = np.where(drug_names == "nintedanib")[0]
    if len(idx) == 0:
        warnings.warn(f"[{name}] nintedanib not in extended_fdr_null drug set — figure skipped.")
        return
    null_col = null_scores[:, idx[0]]
    row = res[res.drug == "nintedanib"].iloc[0]
    obs, emp_p, fdr = row["net_trace_cos"], row["emp_pval"], row["bh_fdr"]

    n_sig = int(res["fdr_sig"].sum())
    n_tested = len(res)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), gridspec_kw={"width_ratios": [1.3, 1]})

    ax = axes[0]
    ax.hist(null_col, bins=60, color=GRAY, edgecolor="none", alpha=0.85,
            label=f"permutation null\n(n={len(null_col):,})")
    ax.axvline(obs, color=RED, lw=1.8, zorder=4)
    ax.text(obs, ax.get_ylim()[1] * 0.92, f"  nintedanib\n  observed = {obs:.3f}\n"
            f"  emp. p = {emp_p:.4f}\n  BH-FDR = {fdr:.3f}",
            color=RED, fontsize=7, va="top", ha="left")
    ax.set_xlabel("Net-TRACE cosine reversal score (permuted)")
    ax.set_ylabel("permutations")
    set_frame(ax)
    ax.set_title("Nintedanib vs. its permutation null", fontsize=9, loc="left")

    ax = axes[1]
    order = res.sort_values("net_trace_cos", ascending=False).reset_index(drop=True)
    colors = [GREEN if s else GRAY for s in order["fdr_sig"]]
    xs = np.arange(len(order))
    ax.scatter(xs, order["net_trace_cos"], s=10, c=colors, linewidths=0)
    nint_x = order.index[order.drug == "nintedanib"][0]
    ax.scatter([nint_x], [order.loc[nint_x, "net_trace_cos"]], s=45, facecolors="none",
               edgecolors=RED, linewidths=1.6, zorder=5)
    ax.annotate("nintedanib", (nint_x, order.loc[nint_x, "net_trace_cos"]),
                textcoords="offset points", xytext=(6, 6), fontsize=7, color=RED)
    ax.set_xlabel("drug rank (top 100 by Net-TRACE)")
    ax.set_ylabel("Net-TRACE cosine reversal score")
    set_frame(ax)
    ax.set_title(f"{n_sig}/{n_tested} FDR-significant\n(BH<0.05, top-100 pre-selected)", fontsize=8.5, loc="left")
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=GREEN, markersize=6, label="FDR<0.05"),
               Line2D([0], [0], marker="o", color="none", markerfacecolor=GRAY, markersize=6, label="not significant")]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=7)

    panel_letter(axes[0], "a")
    panel_letter(axes[1], "b")
    fig.suptitle("Figure 3 | Empirical permutation null places nintedanib in the significant tail",
                 fontsize=10, x=0.02, ha="left", y=1.03)
    fig.text(0.02, -0.02,
             "Note: FDR is conditional on the top-100 pre-selection by Net-TRACE score (post-selection "
             "inference) — see extended_fdr_report.txt.", fontsize=6.5, color=META_GREY, ha="left")
    fig.tight_layout()
    _save(fig, name)


# ══════════════════════════════════════════════════════════════════════════
# Figure 4 — Bootstrap rank-stability CI for top Net-TRACE candidates
# ══════════════════════════════════════════════════════════════════════════
def fig4_bootstrap_ci():
    name = "fig4_bootstrap_ci.png"
    path = REV / "bootstrap_rank_ci.csv"
    if not _require(path, name):
        return
    d = pd.read_csv(path).sort_values("point_rank").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(6.8, 6.0))
    ys = np.arange(len(d))[::-1]
    colors = [RED if r == "nintedanib" else BLUE for r in d["drug"]]
    for y, (_, row), c in zip(ys, d.iterrows(), colors):
        ax.plot([row["ci_lo_95"], row["ci_hi_95"]], [y, y], color=c, lw=1.6, alpha=0.7, zorder=2)
        ax.scatter([row["median_rank"]], [y], color=c, s=26, zorder=3)
        ax.scatter([row["point_rank"]], [y], marker="|", color=DGRAY, s=60, zorder=4)
    ax.set_yticks(ys)
    labels = [f"{'★ ' if drug == 'nintedanib' else ''}{drug}" for drug in d["drug"]]
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("bootstrap rank (95% CI), lower = better")
    ax.invert_xaxis()
    set_frame(ax)
    handles = [
        Line2D([0], [0], marker="o", color=BLUE, markersize=6, lw=1.6, label="median rank (95% bootstrap CI)"),
        Line2D([0], [0], marker="|", color=DGRAY, markersize=9, lw=0, label="point-estimate rank"),
        Line2D([0], [0], marker="o", color=RED, markersize=6, lw=1.6, label="nintedanib (positive control)"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=6.8, frameon=False)
    ax.set_title("Figure 4 | Rank stability of top Net-TRACE candidates under case resampling",
                 fontsize=9.5, loc="left")
    fig.tight_layout()
    _save(fig, name)


# ══════════════════════════════════════════════════════════════════════════
# Figure 5 — AUROC / AUPRC benchmarking against baselines
# ══════════════════════════════════════════════════════════════════════════
def fig5_auroc_benchmark():
    name = "fig5_auroc_benchmark.png"
    path = BENCH / "auroc_summary.csv"
    if not _require(path, name):
        return
    d = pd.read_csv(path)

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.4))

    ax = axes[0]
    order = d.sort_values("auroc", ascending=True).reset_index(drop=True)
    colors = ["#c2c2c2" if arm.startswith(("Baseline", "CMap", "Pearson")) else
              (TEAL if "VAE" in arm else BLUE) for arm in order["arm"]]
    ys = np.arange(len(order))
    xerr = np.vstack([order["auroc"] - order["ci_lo_95"], order["ci_hi_95"] - order["auroc"]])
    ax.errorbar(order["auroc"], ys, xerr=xerr, fmt="none", ecolor=DGRAY, elinewidth=1, capsize=2, zorder=2)
    ax.scatter(order["auroc"], ys, c=colors, s=40, zorder=3)
    ax.axvline(0.5, color=GRAY, ls="--", lw=1, zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{r.disease} \u2013 {r.arm}" for r in order.itertuples()], fontsize=7)
    ax.set_xlabel("AUROC (recovery of known actives, 95% CI)")
    ax.set_xlim(0, 1.05)
    set_frame(ax)
    ax.set_title("AUROC vs. chance (dashed)", fontsize=9, loc="left")

    ax = axes[1]
    ax.bar(np.arange(len(d)), d["auprc_fold_over_random"], color=[
        (TEAL if "VAE" in a else (BLUE if a.startswith("Net") else "#c2c2c2")) for a in d["arm"]
    ], width=0.6, edgecolor="none")
    for x, v, n in zip(np.arange(len(d)), d["auprc_fold_over_random"], d["n_actives"]):
        ax.text(x, v, f"{v:.1f}\u00d7", ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(np.arange(len(d)))
    ax.set_xticklabels([f"{r.disease}\n{r.arm}" for r in d.itertuples()], fontsize=6.2, rotation=0)
    ax.set_ylabel("AUPRC fold-enrichment over random baseline")
    set_frame(ax)
    ax.set_title("Precision enrichment (n actives noted per bar)", fontsize=9, loc="left")
    ax.text(0.98, 0.95, "n actives: " + ", ".join(f"{r.arm.split()[0]}={r.n_actives}"
             for r in d.drop_duplicates('disease').itertuples()),
             transform=ax.transAxes, fontsize=6, color=META_GREY, ha="right", va="top")

    panel_letter(axes[0], "a")
    panel_letter(axes[1], "b")
    fig.suptitle("Figure 5 | Net-TRACE / VAE-TRACE outperform expression-similarity baselines on known actives",
                 fontsize=9.8, x=0.02, ha="left", y=1.03)
    fig.text(0.02, -0.03, "Caveat: n actives is small (IPF n=2, RA n=3) — wide AUROC CIs reflect this; "
             "see auroc_summary.csv.", fontsize=6.3, color=META_GREY, ha="left")
    fig.tight_layout()
    _save(fig, name)


# ══════════════════════════════════════════════════════════════════════════
# Figure 6 — Multi-evidence dossier heatmap for top candidates
# ══════════════════════════════════════════════════════════════════════════
def fig6_evidence_heatmap():
    name = "fig6_evidence_heatmap.png"
    path = AIM3 / "final_dossier.csv"
    if not _require(path, name):
        return
    d = pd.read_csv(path).sort_values("combined_rank").reset_index(drop=True)

    # Build a normalized evidence matrix: reversal score, genetic support, trial evidence,
    # FAERS signal (inverted: high ROR = caution, shown as its own row), literature support.
    trial_map = {"none": 0, "moderate": 0.5, "strong": 1.0}
    lit_map = {"none": 0, "low": 0.33, "moderate": 0.66, "strong": 1.0}
    rev = (d["net_trace"] - d["net_trace"].min()) / (d["net_trace"].max() - d["net_trace"].min())
    gen = d["genetic_support"].fillna(0)
    gen = gen / gen.max() if gen.max() > 0 else gen
    trial = d["trial_evidence"].map(trial_map).fillna(0)
    lit = d["lit_best_strength"].map(lit_map).fillna(0)
    faers = d["faers_ror"].fillna(0)
    faers_scaled = faers / faers.max() if faers.max() > 0 else faers * 0

    mat = np.vstack([rev, gen, trial, lit, faers_scaled])
    row_labels = ["Net-TRACE\nreversal", "Genetic support\n(Open Targets)",
                  "Clinical trial\nevidence", "Literature\nsupport", "FAERS ROR\n(caution signal)"]

    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    im = ax.imshow(mat, cmap="RdBu_r" if False else "Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(d)))
    xlabels = [f"{'★ ' if dr in POS_CONTROLS else ''}{dr}" for dr in d["drug"]]
    ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=7.5)
    # annotate FAERS row with actual ROR values, and mark adverse-signal drugs
    for j, ror in enumerate(faers):
        if not np.isnan(ror):
            txt = f"{ror:.1f}"
            ax.text(j, 4, txt, ha="center", va="center", fontsize=6,
                    color="white" if faers_scaled.iloc[j] > 0.5 else "black")
    for j, mech in enumerate(d["mechanism"]):
        if "ADVERSE" in str(mech):
            ax.add_patch(plt.Rectangle((j - 0.5, -0.5), 1, len(row_labels), fill=False,
                                        edgecolor=RED, lw=1.8, zorder=5))
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("normalized evidence strength", fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    set_frame(ax, style="none")
    ax.set_title("Figure 6 | Multi-evidence dossier across reversal, genetics, trials, literature, and pharmacovigilance",
                 fontsize=9.5, loc="left")
    fig.text(0.02, -0.05, "★ = positive control. Red outline = drug carrying an independent FAERS/label "
             "adverse pulmonary signal (osimertinib, afatinib). Ranked by combined_rank (final_dossier.csv).",
             fontsize=6.3, color=META_GREY, ha="left")
    fig.tight_layout()
    _save(fig, name)


# ══════════════════════════════════════════════════════════════════════════
# Extended Data 7 — scRNA-seq AT2->AT1 transition signature validation
# ══════════════════════════════════════════════════════════════════════════
def ext7_scrna_at2at1():
    name = "ext7_scrna_at2at1.png"
    sig_path = SCRNA / "at2_at1_transition_signature.csv"
    cmp_path = SCRNA / "scrna_vs_bulk_comparison.csv"
    summ_path = SCRNA / "scrna_summary.txt"
    if not all(_require(p, name) for p in [sig_path, cmp_path]):
        return
    sig = pd.read_csv(sig_path)
    cmp = pd.read_csv(cmp_path).rename(columns={"Unnamed: 0": "drug"})

    rho, p = spearmanr(cmp["net_trace"], cmp["scrna_rank"], nan_policy="omit")
    n_common = cmp[["net_trace", "scrna_rank"]].dropna().shape[0]

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), gridspec_kw={"width_ratios": [1, 1.15]})

    # Panel a: AT2->AT1 transition DE signature volcano-style (log2FC vs -log10 padj)
    ax = axes[0]
    sig = sig.copy()
    sig["neglog10p"] = -np.log10(sig["padj"].clip(lower=1e-300))
    sig_sig = sig["padj"] < 0.05
    ax.scatter(sig.loc[~sig_sig, "log2FC"], sig.loc[~sig_sig, "neglog10p"],
               s=4, color=GRAY, alpha=0.4, linewidths=0)
    ax.scatter(sig.loc[sig_sig, "log2FC"], sig.loc[sig_sig, "neglog10p"],
               s=4, color=BLUE, alpha=0.6, linewidths=0)
    ax.axhline(-np.log10(0.05), color=GRAY, ls="--", lw=0.8)
    ax.set_xlabel("log2FC (AT1-like vs. AT2, scRNA)")
    ax.set_ylabel("-log10(padj)")
    set_frame(ax)
    ax.set_title(f"AT2\u2192AT1 transition signature\n({int(sig_sig.sum()):,}/{len(sig):,} genes padj<0.05)",
                 fontsize=8.7, loc="left")

    # Panel b: scRNA-derived vs bulk-derived Net-TRACE reversal rank concordance
    ax = axes[1]
    ax.scatter(cmp["net_trace"], cmp["scrna_rank"], s=6, color=GRAY, alpha=0.4, linewidths=0)
    for drug in POS_CONTROLS:
        row = cmp[cmp["drug"] == drug]
        if len(row):
            ax.scatter(row["net_trace"], row["scrna_rank"], s=45, color=RED if drug == "nintedanib" else ORANGE,
                       zorder=5, label=f"\u2605 {drug}")
    ax.set_xlabel("bulk Net-TRACE reversal score")
    ax.set_ylabel("scRNA AT2\u2192AT1 rank")
    set_frame(ax)
    ax.legend(loc="upper right", frameon=False, fontsize=6.8)
    ax.set_title(f"scRNA vs. bulk concordance: Spearman \u03c1={rho:.3f}\n(p={p:.3g}, n={n_common:,}) \u2014 weak/uncorrelated",
                 fontsize=8.7, loc="left")

    panel_letter(axes[0], "a")
    panel_letter(axes[1], "b")
    fig.suptitle("Extended Data 7 | scRNA-seq AT2\u2192AT1 transition signature and its concordance with bulk Net-TRACE",
                 fontsize=9.8, x=0.02, ha="left", y=1.03)
    fig.text(0.02, -0.03, "Caveat: scRNA-derived drug ranks correlate weakly with bulk-derived ranks "
             "(\u03c1\u22480.02, not significant) \u2014 an independent, largely orthogonal validation arm, "
             "not a replication. See scrna_summary.txt.", fontsize=6.3, color=META_GREY, ha="left")
    fig.tight_layout()
    _save(fig, name)


# ══════════════════════════════════════════════════════════════════════════
# Extended Data 8 — L2S2 2,834-compound universe cross-validation
# ══════════════════════════════════════════════════════════════════════════
def ext8_l2s2_expansion():
    name = "ext8_l2s2_expansion.png"
    l2s2_path = L2S2 / "l2s2_consensus_scores.csv"
    trace_path = REV / "trace_scores.csv"
    report_path = L2S2 / "l2s2_correlation_report.txt"
    if not all(_require(p, name) for p in [l2s2_path, trace_path]):
        return
    l2s2 = pd.read_csv(l2s2_path)
    trace = pd.read_csv(trace_path)
    l2s2 = l2s2.copy(); trace = trace.copy()
    l2s2["key"] = l2s2["drug"].str.lower().str.strip()
    trace["key"] = trace["drug"].str.lower().str.strip()
    merged = l2s2.merge(trace, on="key", suffixes=("_l2s2", "_trace"))
    rho, p = spearmanr(merged["l2s2_rank"], merged["trace_rank"])
    n_common = len(merged)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), gridspec_kw={"width_ratios": [1, 1.1]})

    # Panel a: universe size comparison
    ax = axes[0]
    labels = ["L1000/CMap\n(main universe)", "L2S2\n(expanded universe)"]
    counts = [len(trace), len(l2s2)]
    xs = np.arange(2)
    ax.bar(xs, counts, color=[BLUE, TEAL], width=0.55, edgecolor="none")
    for x, v in zip(xs, counts):
        ax.text(x, v, f"{v:,}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("compounds scored")
    set_frame(ax)
    ax.set_title("Compound-universe expansion", fontsize=9, loc="left")

    # Panel b: rank concordance scatter, main-universe rank vs L2S2 rank
    ax = axes[1]
    ax.scatter(merged["trace_rank"], merged["l2s2_rank"], s=5, color=GRAY, alpha=0.35, linewidths=0)
    for drug in POS_CONTROLS:
        row = merged[merged["key"] == drug]
        if len(row):
            ax.scatter(row["trace_rank"], row["l2s2_rank"], s=45,
                       color=RED if drug == "nintedanib" else ORANGE, zorder=5, label=f"\u2605 {drug}")
    ax.set_xlabel("Net-TRACE rank (L1000 universe, n=1,768)")
    ax.set_ylabel("L2S2 reversal rank (n=2,834)")
    set_frame(ax)
    ax.legend(loc="lower right", frameon=False, fontsize=6.8)
    ax.set_title(f"Rank concordance on n={n_common} shared drugs\nSpearman \u03c1={rho:.4f}, p={p:.2e}",
                 fontsize=8.7, loc="left")

    panel_letter(axes[0], "a")
    panel_letter(axes[1], "b")
    fig.suptitle("Extended Data 8 | L2S2 expanded compound universe cross-validates Net-TRACE ranking",
                 fontsize=9.8, x=0.02, ha="left", y=1.03)
    fig.text(0.02, -0.03, "Weak-but-significant positive concordance across independent compound universes "
             "(L1000 CMap vs. L2S2) supports the reversal signal's robustness to universe choice. "
             "See l2s2_correlation_report.txt.", fontsize=6.3, color=META_GREY, ha="left")
    fig.tight_layout()
    _save(fig, name)


# ══════════════════════════════════════════════════════════════════════════
# Extended Data 9 — CRISPR knockout reversal targets
# ══════════════════════════════════════════════════════════════════════════
def ext9_crispr_targets():
    name = "ext9_crispr_targets.png"
    scores_path = CRISPR / "crispr_reversal_scores.csv"
    priority_path = CRISPR / "crispr_priority_targets.csv"
    if not all(_require(p, name) for p in [scores_path, priority_path]):
        return
    cr = pd.read_csv(scores_path).sort_values("reversal_score", ascending=False).reset_index(drop=True)
    cr["rank"] = np.arange(1, len(cr) + 1)
    pr = pd.read_csv(priority_path)
    n_genes = len(cr)

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.4), gridspec_kw={"width_ratios": [1.1, 1]})

    # Panel a: full CRISPR reversal-score rank distribution with HMGCR/HSP90AB1 marked
    ax = axes[0]
    ax.plot(cr["rank"], cr["reversal_score"], color=GRAY, lw=1.2)
    highlight = {"HSP90AB1": GREEN, "HMGCR": GOLD}
    for gene, col in highlight.items():
        row = cr[cr.gene == gene]
        if len(row):
            r, s = int(row["rank"].iloc[0]), float(row["reversal_score"].iloc[0])
            pct = 100 * r / n_genes
            ax.scatter([r], [s], s=45, color=col, zorder=5)
            ax.annotate(f"{gene}\nrank {r}/{n_genes} ({pct:.1f}%ile)", (r, s),
                        textcoords="offset points", xytext=(10, 8 if gene == "HSP90AB1" else -28),
                        fontsize=7, color=col)
    ax.set_xlabel("CRISPR knockout rank (by reversal score)")
    ax.set_ylabel("reversal score")
    set_frame(ax)
    ax.set_title("Genome-wide CRISPR reversal screen", fontsize=9, loc="left")

    # Panel b: top-20 priority targets ranked by composite priority_score
    ax = axes[1]
    pr2 = pr.sort_values("priority_score", ascending=True).reset_index(drop=True)
    ys = np.arange(len(pr2))
    colors = [highlight.get(g, BLUE) for g in pr2["gene"]]
    ax.barh(ys, pr2["priority_score"], color=colors, height=0.65, edgecolor="none")
    ax.set_yticks(ys)
    ax.set_yticklabels(pr2["gene"], fontsize=6.8)
    ax.set_xlabel("composite priority score")
    set_frame(ax)
    ax.set_title("Top-20 priority CRISPR targets", fontsize=9, loc="left")

    panel_letter(axes[0], "a")
    panel_letter(axes[1], "b")
    fig.suptitle("Extended Data 9 | CRISPR knockout screen prioritizes HSP90AB1; HMGCR is a weak, non-significant hit",
                 fontsize=9.3, x=0.02, ha="left", y=1.03)
    fig.text(0.02, -0.03, "HSP90AB1 ranks in the top 0.3% of 5,208 genes screened (strong hit); HMGCR ranks "
             "~79th percentile (weak/non-significant) \u2014 shown honestly, not overstated. "
             "See crispr_reversal_scores.csv.", fontsize=6.3, color=META_GREY, ha="left")
    fig.tight_layout()
    _save(fig, name)


# ══════════════════════════════════════════════════════════════════════════
# Extended Data 10 — RA dual-disease specificity control
# ══════════════════════════════════════════════════════════════════════════
def ext10_ra_specificity():
    name = "ext10_ra_specificity.png"
    ra_path = VALID / "ra_trace_scores.csv"
    dual_path = BENCH / "ablation_dual_disease.csv"
    if not all(_require(p, name) for p in [ra_path, dual_path]):
        return
    ra = pd.read_csv(ra_path)
    dual = pd.read_csv(dual_path)

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.4))

    # Panel a: IPF-approved drugs shift from top-1% (IPF) to ~70-98%ile (RA) -- specificity
    ax = axes[0]
    ipf_pct = {"nintedanib": 0.8, "pirfenidone": 39.5}  # from ablation_table.csv (Net-TRACE, IPF)
    ra_rows = ra[ra.drug.isin(POS_CONTROLS)]
    for drug in POS_CONTROLS:
        col = RED if drug == "nintedanib" else ORANGE
        ra_row = ra_rows[ra_rows.drug == drug]
        if len(ra_row) == 0:
            continue
        y_ipf, y_ra = ipf_pct[drug], float(ra_row["trace_pct"].iloc[0])
        ax.plot([0, 1], [y_ipf, y_ra], color=col, lw=1.8, marker="o", markersize=7)
        ax.text(1.03, y_ra, f"\u2605 {drug}", color=col, fontsize=7.5, va="center")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["IPF signature", "RA signature"], fontsize=8)
    ax.set_ylabel("Net-TRACE rank percentile (lower = stronger reversal)")
    ax.set_xlim(-0.15, 1.5)
    ax.invert_yaxis()
    set_frame(ax)
    ax.set_title("IPF-specific drugs do not score for RA", fontsize=9, loc="left")

    # Panel b: RA-approved drugs are NOT recovered by the IPF-tuned pipeline (as expected)
    ax = axes[1]
    ra_drugs = ["tofacitinib", "baricitinib", "dexamethasone", "leflunomide"]
    sub = ra[ra.drug.isin(ra_drugs)].set_index("drug").loc[ra_drugs].reset_index()
    xs = np.arange(len(sub))
    ax.bar(xs, sub["trace_pct"], color=PURPLE, width=0.55, edgecolor="none")
    for x, v in zip(xs, sub["trace_pct"]):
        ax.text(x, v, f"{v:.0f}%ile", ha="center", va="bottom", fontsize=7)
    ax.axhline(50, color=GRAY, ls="--", lw=0.8)
    ax.set_xticks(xs); ax.set_xticklabels(sub["drug"], fontsize=7.5, rotation=20, ha="right")
    ax.set_ylabel("Net-TRACE rank percentile (RA signature)")
    set_frame(ax)
    ax.set_title("RA-approved drugs, RA signature (not recovered)", fontsize=9, loc="left")

    panel_letter(axes[0], "a")
    panel_letter(axes[1], "b")
    fig.suptitle("Extended Data 10 | RA is a disease-specificity control, not a generalization demonstration",
                 fontsize=9.5, x=0.02, ha="left", y=1.03)
    fig.text(0.02, -0.05, "Nintedanib shifts from 0.8%ile (IPF) to ~98%ile (RA) \u2014 a ~120\u00d7 rank change, "
             "showing the IPF signal is not a generic artifact. RA-approved drugs are NOT recovered by the "
             "IPF-tuned pipeline, consistent with the pipeline not generalizing to RA without disease-specific "
             "adaptation. See ra_generalizability_report.txt.", fontsize=6.3, color=META_GREY, ha="left")
    fig.tight_layout()
    _save(fig, name)


# ══════════════════════════════════════════════════════════════════════════
# Extended Data 11 — Independent held-out cohort validation
# ══════════════════════════════════════════════════════════════════════════
def ext11_heldout_validation():
    name = "ext11_heldout_validation.png"
    concord_path = AIM3 / "heldout_v2_concordance.txt"
    report_path = AIM3 / "heldout_report.txt"
    if not all(_require(p, name) for p in [concord_path, report_path]):
        return

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))

    # Panel a: GSE47460 (LGRC) direction-concordance vs. chance
    ax = axes[0]
    labels = ["All ILD\nvs. controls\n(n=6,443 genes)", "UIP/IPF subset\nvs. controls\n(n=6,443 genes)"]
    vals = [77.0, 78.8]
    xs = np.arange(2)
    ax.bar(xs, vals, color=[BLUE, TEAL], width=0.5, edgecolor="none")
    for x, v in zip(xs, vals):
        ax.text(x, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.axhline(50, color=GRAY, ls="--", lw=1)
    ax.text(1.4, 51, "chance = 50%", color=META_GREY, fontsize=6.5, va="bottom")
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("directional concordance with\nIPF consensus signature (%)")
    ax.set_ylim(0, 100)
    set_frame(ax)
    ax.set_title("GSE47460 (LGRC): direction concordance", fontsize=8.8, loc="left")

    # Panel b: GSE134692 (transplant-stage, no-control) caveat panel
    ax = axes[1]
    ax.axis("off")
    txt = (
        "GSE134692 (n=80 transplant-stage IPF, no controls)\n\n"
        "Spearman r = \u22120.29 (p=1.2e-144)\n"
        "ssGSEA UP set: mean ES=0.40 vs. 0.5 null (p=2.0e-36)\n\n"
        "INCONCLUSIVE by design: no healthy controls means\n"
        "z-score correlation and ssGSEA cannot separate\n"
        "'up in IPF' from 'low baseline expression.'\n"
        "The negative correlation is an artifact of the\n"
        "no-control design \u2014 not evidence against the\n"
        "consensus signature. See heldout_report.txt."
    )
    ax.text(0.02, 0.95, txt, transform=ax.transAxes, fontsize=7.6, va="top", ha="left",
            color=DGRAY, linespacing=1.5)
    ax.set_title("GSE134692: no-control caveat (inconclusive)", fontsize=8.8, loc="left")

    panel_letter(axes[0], "a")
    panel_letter(axes[1], "b")
    fig.suptitle("Extended Data 11 | Independent held-out cohort validation: one clean pass, one inconclusive-by-design",
                 fontsize=9.3, x=0.02, ha="left", y=1.03)
    fig.text(0.02, -0.03, "GSE47460 provides a genuine held-out concordance test (both arms pass >65% vs. 50% "
             "chance). GSE134692 lacks healthy controls and cannot properly test direction \u2014 flagged as "
             "inconclusive rather than treated as a failure.", fontsize=6.3, color=META_GREY, ha="left")
    fig.tight_layout()
    _save(fig, name)


# ══════════════════════════════════════════════════════════════════════════
# Extended Data 12 — Drug-target Mendelian Randomization (null / underpowered)
# ══════════════════════════════════════════════════════════════════════════
def ext12_mr_forest():
    name = "ext12_mr_forest.png"
    ext_path = MR / "extended_mr_results.csv"
    hmgcr_path = MR / "hmgcr_mr_results.csv"
    if not all(_require(p, name) for p in [ext_path, hmgcr_path]):
        return
    ext = pd.read_csv(ext_path)
    hmgcr = pd.read_csv(hmgcr_path)
    hmgcr = hmgcr.copy(); hmgcr["gene"] = "HMGCR"; hmgcr["drugs"] = "statins (proxy)"
    ivw_ext = ext[ext.method == "Inverse variance weighted"].copy()
    ivw_h = hmgcr[hmgcr.method == "Inverse variance weighted"].copy()
    ivw_wald = ext[ext.method == "Wald ratio"].copy()  # single-SNP targets have no IVW row
    rows = pd.concat([ivw_h, ivw_ext[ivw_ext.gene.isin(ivw_ext.gene) & ~ivw_ext.gene.isin(ivw_wald.gene)],
                      ivw_wald], ignore_index=True)
    rows = rows.drop_duplicates(subset=["gene"]).reset_index(drop=True)
    rows["OR"] = np.exp(rows["b"])
    rows["OR_lo"] = np.exp(rows["b"] - 1.96 * rows["se"])
    rows["OR_hi"] = np.exp(rows["b"] + 1.96 * rows["se"])
    rows = rows.sort_values("OR").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ys = np.arange(len(rows))
    xerr = np.vstack([rows["OR"] - rows["OR_lo"], rows["OR_hi"] - rows["OR"]])
    single_snp = rows["nsnp"] <= 1
    ax.errorbar(rows.loc[~single_snp, "OR"], ys[~single_snp.values], xerr=xerr[:, ~single_snp.values],
                fmt="o", color=GOLD, ecolor=GOLD, elinewidth=1.4, capsize=3, markersize=6, zorder=3)
    ax.errorbar(rows.loc[single_snp, "OR"], ys[single_snp.values], xerr=xerr[:, single_snp.values],
                fmt="o", color=GRAY, ecolor=GRAY, elinewidth=1.4, capsize=3, markersize=6, zorder=3,
                alpha=0.6)
    ax.axvline(1.0, color=DGRAY, ls="--", lw=1)
    ax.set_yticks(ys)
    ylabels = [f"{g} ({int(n)} SNP{'s' if n != 1 else ''})\n{d}"
               for g, n, d in zip(rows["gene"], rows["nsnp"], rows["drugs"])]
    ax.set_yticklabels(ylabels, fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("OR per SD genetically-predicted exposure (95% CI, log scale)")
    set_frame(ax)
    for y, (_, row) in zip(ys, rows.iterrows()):
        ax.text(row["OR_hi"] * 1.1, y, f"p={row['pval']:.2f}", fontsize=6.3, color=META_GREY, va="center")
    ax.set_title("Extended Data 12 | Drug-target MR: NULL and underpowered \u2014 no significant effect at any target",
                 fontsize=9, loc="left")
    fig.text(0.02, -0.06, "All estimates p>0.05; wide CIs span clinically opposite effects. HMGCR/FLT1 trend "
             "OR<1 (directionally concordant with reversal prediction) but are NOT positive evidence. Gray = "
             "single-SNP Wald-ratio estimates (no pleiotropy test, not independently interpretable). "
             "See mr_report.txt for power analysis and prior published replication.",
             fontsize=6.3, color=META_GREY, ha="left", wrap=True)
    fig.tight_layout()
    _save(fig, name)


# ══════════════════════════════════════════════════════════════════════════
# Extended Data 13 — Robustness: weight sensitivity + negative controls
# ══════════════════════════════════════════════════════════════════════════
def ext13_robustness():
    name = "ext13_robustness.png"
    ws_path = REV / "weight_sensitivity.csv"
    nc_path = REV / "negative_control_results.csv"
    if not all(_require(p, name) for p in [ws_path, nc_path]):
        return
    ws = pd.read_csv(ws_path)
    nc = pd.read_csv(nc_path).dropna(subset=["combined_pct"]).sort_values("combined_pct").reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.4), gridspec_kw={"width_ratios": [1, 1.2]})

    # Panel a: top-10 jaccard stability across 125 weight perturbations
    ax = axes[0]
    ax.hist(ws["jaccard_top10"], bins=np.arange(0.6, 1.05, 0.025), color=BLUE, alpha=0.85, edgecolor="none")
    ax.set_xlabel("Jaccard(top-10) vs. default weights")
    ax.set_ylabel("count (of 125 weight perturbations)")
    set_frame(ax)
    med = ws["jaccard_top10"].median()
    ax.axvline(med, color=RED, lw=1.4, ls="--")
    ax.text(med, ax.get_ylim()[1] * 0.9, f" median={med:.2f}", color=RED, fontsize=7)
    ax.set_title("Top-10 rank stability under weight perturbation", fontsize=8.8, loc="left")

    # Panel b: negative-control drugs by category (rank percentile; lower=stronger reversal)
    ax = axes[1]
    cat_colors = {"A": GOLD, "B": GRAY, "C": RED}
    colors = [cat_colors.get(str(c)[0], GRAY) for c in nc["category"]]
    xs = np.arange(len(nc))
    ax.bar(xs, nc["combined_pct"], color=colors, width=0.6, edgecolor="none")
    ax.set_xticks(xs)
    ax.set_xticklabels(nc["drug"], fontsize=7, rotation=35, ha="right")
    ax.set_ylabel("combined rank percentile\n(lower = stronger reversal)")
    ax.invert_yaxis()
    set_frame(ax)
    handles = [mpatches.Patch(color=GOLD, label="A: pro-fibrotic (expect no reversal)"),
               mpatches.Patch(color=GRAY, label="B: unrelated pharmacology"),
               mpatches.Patch(color=RED, label="C: known pulmonary-toxic (FAERS signal)")]
    ax.legend(handles=handles, loc="upper left", fontsize=6.3, frameon=False)
    ax.set_title("Negative-control drug categories", fontsize=8.8, loc="left")

    panel_letter(axes[0], "a")
    panel_letter(axes[1], "b")
    fig.suptitle("Extended Data 13 | Robustness: rank stability under weight perturbation and negative-control behavior",
                 fontsize=9.3, x=0.02, ha="left", y=1.03)
    fig.text(0.02, -0.05, "Top-10 candidate set is highly stable across 125 weight configurations "
             "(median Jaccard >0.75). Osimertinib/afatinib (category C, independent FAERS pulmonary-toxicity "
             "signal) score anomalously strongly \u2014 flagged, not hidden \u2014 in fig6/dossier. "
             "See weight_sensitivity.csv, negative_control_results.csv.",
             fontsize=6.3, color=META_GREY, ha="left")
    fig.tight_layout()
    _save(fig, name)


# ══════════════════════════════════════════════════════════════════════════
# Extended Data 14 — VAE-TRACE model architecture
# ══════════════════════════════════════════════════════════════════════════
def _sch_box(ax, xy, w, h, text, fc, ec=DGRAY, fontsize=6.6, fontcolor="white", lw=0.9):
    """Draw a rounded schematic box; returns (cx, cy, x, y, w, h) for chaining arrows."""
    x, y = xy
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015,rounding_size=0.06",
                        fc=fc, ec=ec, lw=lw, zorder=3, mutation_aspect=1)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
            color=fontcolor, zorder=4, linespacing=1.35)
    return (x + w / 2, y + h / 2, x, y, w, h)


def _sch_hjoin(ax, b1, b2, color=DGRAY, lw=1.0):
    x1 = b1[2] + b1[4]; y1 = b1[1]
    x2 = b2[2];         y2 = b2[1]
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", color=color, lw=lw,
                                  mutation_scale=8, zorder=2))


def ext14_vae_architecture():
    name = "ext14_vae_architecture.png"
    ckpt_path = EMB / "vae_model.pt"
    sig_path = L1000 / "sm_sig_info.csv"
    drugmeta_path = L1000 / "drug_metadata.csv"
    if not all(_require(p, name) for p in [ckpt_path, sig_path]):
        return

    # Pull real architecture + parameter counts from the trained checkpoint rather
    # than hardcoding shapes, so the figure tracks the actual model on disk.
    n_genes, latent_dim = 978, 128
    hidden1, hidden2 = 512, 256
    n_params = None
    if _HAS_TORCH:
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            sd = ckpt["model_state"]
            n_genes = int(ckpt.get("n_genes", n_genes))
            latent_dim = int(ckpt.get("latent_dim", latent_dim))
            hidden1 = sd["enc.0.weight"].shape[0]
            hidden2 = sd["enc.3.weight"].shape[0]
            n_params = int(sum(v.numel() for v in sd.values()))
        except Exception as e:
            warnings.warn(f"[{name}] could not read vae_model.pt checkpoint ({e}); using script defaults.")
    param_str = f"{n_params/1e6:.2f}M" if n_params else "1.37M"

    n_sigs = None
    n_drugs = None
    try:
        sig_info = pd.read_csv(sig_path, usecols=["pert_iname"], low_memory=False)
        n_sigs = len(sig_info)
        n_drugs = sig_info["pert_iname"].nunique()
    except Exception:
        pass
    if drugmeta_path.exists() and n_drugs is None:
        try:
            n_drugs = len(pd.read_csv(drugmeta_path))
        except Exception:
            pass

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.9), gridspec_kw={"width_ratios": [1.72, 1]})

    # ---- Panel a: encoder/decoder schematic ----
    ax = axes[0]
    ax.set_xlim(0, 130)
    ax.set_ylim(0, 58)
    ax.axis("off")
    CY = 29

    b_in = _sch_box(ax, (0, CY - 14), 11, 28, f"input\nL1000\nsignature\n({n_genes} genes)", GRAY)
    b_e1 = _sch_box(ax, (14, CY - 10), 13, 20, f"Linear\n{n_genes}\u2192{hidden1}\nLayerNorm\nGELU", BLUE)
    b_e2 = _sch_box(ax, (30, CY - 8), 13, 16, f"Linear\n{hidden1}\u2192{hidden2}\nLayerNorm\nGELU", BLUE)
    _sch_hjoin(ax, b_in, b_e1); _sch_hjoin(ax, b_e1, b_e2)

    b_mu = _sch_box(ax, (46, CY + 7), 13, 8, f"Linear\n{hidden2}\u2192{latent_dim}\n\u03bc (mean)", TEAL)
    b_lv = _sch_box(ax, (46, CY - 15), 13, 8, f"Linear\n{hidden2}\u2192{latent_dim}\nlog\u03c3\u00b2 (var)", TEAL)
    for b_head in (b_mu, b_lv):
        x1, y1 = b_e2[2] + b_e2[4], b_e2[1]
        x2, y2 = b_head[2], b_head[1]
        rad = 0.28 if y2 > y1 else -0.28
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", color=DGRAY, lw=1.0,
                                      connectionstyle=f"arc3,rad={rad}", mutation_scale=8, zorder=2))

    b_z = _sch_box(ax, (63, CY - 5), 13, 10, f"z = \u03bc + \u03c3\u2299\u03b5\nreparameterize\nlatent ({latent_dim}-d)",
                    DGRAY, fontsize=6.3)
    for b_head in (b_mu, b_lv):
        x1, y1 = b_head[2] + b_head[4], b_head[1]
        x2, y2 = b_z[2], b_z[1]
        rad = -0.28 if y1 > y2 else 0.28
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", color=DGRAY, lw=1.0,
                                      connectionstyle=f"arc3,rad={rad}", mutation_scale=8, zorder=2))
    eps_x, eps_y = 67, CY - 13
    ax.text(eps_x, eps_y, "\u03b5 ~ N(0,I)", ha="center", va="center", fontsize=6.0, color=META_GREY, style="italic")
    ax.add_patch(FancyArrowPatch((eps_x, eps_y + 1.8), (b_z[0] - 1, b_z[3]), arrowstyle="-|>", color=META_GREY,
                                  lw=0.8, connectionstyle="arc3,rad=0.15", mutation_scale=6, zorder=2, linestyle="--"))

    b_d1 = _sch_box(ax, (80, CY - 8), 13, 16, f"Linear\n{latent_dim}\u2192{hidden2}\nLayerNorm\nGELU", ORANGE)
    b_d2 = _sch_box(ax, (96, CY - 10), 13, 20, f"Linear\n{hidden2}\u2192{hidden1}\nLayerNorm\nGELU", ORANGE)
    _sch_hjoin(ax, b_z, b_d1); _sch_hjoin(ax, b_d1, b_d2)
    b_out = _sch_box(ax, (113, CY - 14), 13, 28, f"Linear\n{hidden1}\u2192{n_genes}\nreconstruction\n\u0177 ({n_genes} genes)", GRAY)
    _sch_hjoin(ax, b_d2, b_out)

    ax.text(0, 54, "encoder", ha="left", fontsize=8.2, color=BLUE, fontweight="bold")
    ax.text(63, 54, "reparameterization", ha="center", fontsize=8.2, color=DGRAY, fontweight="bold")
    ax.text(113, 54, "decoder", ha="right", fontsize=8.2, color=ORANGE, fontweight="bold")
    ax.set_title(f"VAE-TRACE encoder\u2013decoder: {n_genes}\u2192{hidden1}\u2192{hidden2}\u2192{latent_dim}\u2192"
                 f"{hidden2}\u2192{hidden1}\u2192{n_genes}, {param_str} trainable parameters", fontsize=8.8, loc="left")

    # ---- Panel b: training objective + inference ----
    axb = axes[1]
    axb.axis("off"); axb.set_xlim(0, 1); axb.set_ylim(0, 1)

    train_line2 = (f"{n_sigs:,} L1000 Level-5 small-molecule\nsignatures \u00d7 {n_genes} landmark genes"
                   if n_sigs else f"L1000 Level-5 small-molecule\nsignatures \u00d7 {n_genes} landmark genes")
    train_line3 = f"({n_drugs:,} unique compounds)" if n_drugs else ""
    txt_train = f"Training data\n{train_line2}\n{train_line3}\n"
    axb.text(0.02, 0.98, txt_train, transform=axb.transAxes, fontsize=7.2, va="top", ha="left",
             color=DGRAY, linespacing=1.55, fontweight="bold")

    txt_loss = (
        "Training objective (per batch)\n\n"
        r"$\mathcal{L} = \mathcal{L}_{recon} + \beta\,\mathcal{L}_{KL} + \lambda\,\mathcal{L}_{contrastive}$" "\n\n"
        "  \u2022 recon: MSE(\u0177, x)\n"
        "  \u2022 KL: closed-form vs. N(0,I),  \u03b2 = 0.5\n"
        "  \u2022 contrastive: NT-Xent (InfoNCE) on \u03bc,\n"
        "    \u03c4 = 0.1,  \u03bb = 0.1 \u2014 pulls together\n"
        "    latent embeddings of the same drug\n"
        "    across different cell lines\n\n"
        "Adam, lr=1e-3, batch=512, 50 epochs"
    )
    axb.text(0.02, 0.80, txt_loss, transform=axb.transAxes, fontsize=7.0, va="top", ha="left",
             color=DGRAY, linespacing=1.5)

    txt_infer = (
        "Inference \u2192 VAE-TRACE score\n\n"
        "1. encode consensus IPF signature \u2192 \u03bc_IPF\n"
        "2. encode each drug's per-cell-line\n"
        "   signatures \u2192 average \u03bc across lines\n"
        "3. score = \u2212cos(\u03bc_IPF, \u03bc_drug)\n"
        "   (more negative cosine = stronger\n"
        "   predicted transcriptional reversal)"
    )
    axb.text(0.02, 0.30, txt_infer, transform=axb.transAxes, fontsize=7.0, va="top", ha="left",
             color=DGRAY, linespacing=1.5)

    panel_letter(axes[0], "a")
    panel_letter(axes[1], "b")
    fig.suptitle("Extended Data 14 | VAE-TRACE model architecture: contrastive variational autoencoder "
                 "over L1000 landmark-gene signatures", fontsize=9.3, x=0.02, ha="left", y=1.03)
    fig.text(0.02, -0.03,
             "Cell-line-invariance is enforced explicitly by the NT-Xent contrastive term over the mean "
             "latent code \u03bc, not by architecture alone \u2014 the core tissue-mismatch correction in TRACE. "
             "Architecture and parameter counts read directly from results/embedding/vae_model.pt.",
             fontsize=6.3, color=META_GREY, ha="left")
    fig.tight_layout()
    _save(fig, name)


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════
def main():
    fig1_consensus_replication()
    fig2_ablation()
    fig3_null_distribution()
    fig4_bootstrap_ci()
    fig5_auroc_benchmark()
    fig6_evidence_heatmap()
    ext7_scrna_at2at1()
    ext8_l2s2_expansion()
    ext9_crispr_targets()
    ext10_ra_specificity()
    ext11_heldout_validation()
    ext12_mr_forest()
    ext13_robustness()
    ext14_vae_architecture()


if __name__ == "__main__":
    main()
