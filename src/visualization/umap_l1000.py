"""
UMAP of the 978-gene L1000 landmark drug signatures.

Each point = one drug (1,768 total). Axes are the UMAP embedding of the
978-dimensional z-scored expression signature. Color = IPF net-TRACE
reversal score (Pearson; red = strong reversal). Annotated landmarks:
known IPF actives, RA actives, and top-scoring TRACE candidates.

Outputs:
  results/figures/umap_l1000_ipf.png
  results/figures/umap_l1000_ipf.svg

Usage:
  python src/visualization/umap_l1000.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
import umap
from sklearn.preprocessing import RobustScaler

ROOT    = Path(__file__).resolve().parents[2]
L1K     = ROOT / "results/l1000"
BENCH   = ROOT / "results/benchmarking"
ACT     = ROOT / "data/known_actives"
FIG_DIR = ROOT / "results/figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_matrix() -> tuple[pd.DataFrame, np.ndarray]:
    df  = pd.read_csv(L1K / "drug_signatures_landmark.csv.gz", index_col=0)
    mat = df.values.astype(np.float32)
    mat = RobustScaler().fit_transform(mat.T)   # shape: n_drugs × 978
    return df, mat


def load_scores(drug_names: list[str]) -> pd.Series:
    scores = pd.read_csv(BENCH / "dual_disease_scores.csv", index_col="drug")
    return scores["ipf_pearson"].reindex(drug_names).fillna(0)


def load_actives() -> tuple[set, set]:
    ipf = {l.strip().lower() for l in (ACT / "ipf_actives.txt").read_text().splitlines() if l.strip()}
    ra  = {l.strip().lower() for l in (ACT / "ra_actives.txt").read_text().splitlines() if l.strip()}
    return ipf, ra


def compute_umap(mat: np.ndarray, seed: int = 42) -> np.ndarray:
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.40,   # more spread → less crowding in hot cluster
        n_components=2,
        metric="cosine",
        random_state=seed,
        low_memory=False,
    )
    print("  Fitting UMAP on 1,768 × 978 matrix ...")
    return reducer.fit_transform(mat)


def make_figure(embedding: np.ndarray, drug_names: list[str],
                scores: pd.Series, ipf_actives: set, ra_actives: set) -> None:
    score_arr = scores.values

    # ── Top TRACE candidates to label ────────────────────────────────────────
    top_n = pd.Series(score_arr, index=drug_names).nlargest(10).index.tolist()
    label_drugs = set(ipf_actives) | set(ra_actives) | set(d.lower() for d in top_n)

    # ── Colour scale clipped at ±2 SD for visual clarity ──────────────────────
    vmax = np.percentile(np.abs(score_arr), 95)
    vmin = -vmax

    BG, TEXT, SPINE = "white", "#1a1a2e", "#cccccc"

    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor("#f7f7f9")

    # Background scatter — all drugs
    name_lower = [d.lower() for d in drug_names]
    mask_neutral = ~np.isin(name_lower, list(ipf_actives | ra_actives))
    sc = ax.scatter(
        embedding[mask_neutral, 0], embedding[mask_neutral, 1],
        c=score_arr[mask_neutral], cmap="RdYlBu_r",
        vmin=vmin, vmax=vmax,
        s=16, alpha=0.80, linewidths=0, rasterized=True,
    )

    # ── Draw markers for all labelled drugs first ─────────────────────────────
    for i, drug in enumerate(drug_names):
        if drug.lower() in ipf_actives:
            ax.scatter(embedding[i, 0], embedding[i, 1],
                       c=[[score_arr[i]]], cmap="RdYlBu_r", vmin=vmin, vmax=vmax,
                       s=220, marker="*", linewidths=1.2,
                       edgecolors="#b8860b", zorder=5)
        elif drug.lower() in ra_actives:
            ax.scatter(embedding[i, 0], embedding[i, 1],
                       c=[[score_arr[i]]], cmap="RdYlBu_r", vmin=vmin, vmax=vmax,
                       s=110, marker="D", linewidths=1.1,
                       edgecolors="#1a6ea8", zorder=5)

    for drug in top_n:
        if drug.lower() not in ipf_actives and drug.lower() not in ra_actives:
            idxs = [j for j, d in enumerate(name_lower) if d == drug.lower()]
            if not idxs:
                continue
            i = idxs[0]
            ax.scatter(embedding[i, 0], embedding[i, 1],
                       c=[[score_arr[i]]], cmap="RdYlBu_r", vmin=vmin, vmax=vmax,
                       s=90, marker="^", linewidths=0.8,
                       edgecolors="#333", zorder=4)

    # ── Unified label pass — adjustText moves all texts together ─────────────
    try:
        from adjustText import adjust_text
        has_adjust = True
    except ImportError:
        has_adjust = False

    all_texts, all_xs, all_ys = [], [], []

    for i, drug in enumerate(drug_names):
        dl = drug.lower()
        if dl in ipf_actives:
            t = ax.text(embedding[i, 0], embedding[i, 1], drug,
                        fontsize=9, color="#7a5c00", fontweight="bold", zorder=10,
                        path_effects=[pe.withStroke(linewidth=2.5, foreground=BG)])
            all_texts.append(t)
            all_xs.append(embedding[i, 0])
            all_ys.append(embedding[i, 1])
        elif dl in ra_actives:
            t = ax.text(embedding[i, 0], embedding[i, 1], drug,
                        fontsize=7.5, color="#1a4f78", zorder=10,
                        path_effects=[pe.withStroke(linewidth=2.5, foreground=BG)])
            all_texts.append(t)
            all_xs.append(embedding[i, 0])
            all_ys.append(embedding[i, 1])

    for drug in top_n:
        if drug.lower() not in ipf_actives and drug.lower() not in ra_actives:
            idxs = [j for j, d in enumerate(name_lower) if d == drug.lower()]
            if not idxs:
                continue
            i = idxs[0]
            t = ax.text(embedding[i, 0], embedding[i, 1], drug,
                        fontsize=7.5, color="#222", zorder=10,
                        path_effects=[pe.withStroke(linewidth=2.5, foreground=BG)])
            all_texts.append(t)
            all_xs.append(embedding[i, 0])
            all_ys.append(embedding[i, 1])

    if has_adjust and all_texts:
        adjust_text(
            all_texts,
            x=np.array(all_xs), y=np.array(all_ys),
            ax=ax,
            arrowprops=dict(arrowstyle="-", color="#999", lw=0.7),
            expand_text=(1.5, 1.5),
            expand_points=(1.5, 1.5),
        )

    # ── Colorbar ───────────────────────────────────────────────────────────────
    cbar = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02, shrink=0.75)
    cbar.set_label("IPF reversal score (Pearson)", color=TEXT, fontsize=9)
    cbar.ax.yaxis.set_tick_params(color=TEXT, labelcolor=TEXT, labelsize=8)
    cbar.outline.set_edgecolor(SPINE)

    # ── Legend ─────────────────────────────────────────────────────────────────
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="*", color="w", markerfacecolor="gray",
               markeredgecolor="#b8860b", markersize=13, label="IPF active", linestyle="None"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="gray",
               markeredgecolor="#1a6ea8", markersize=9, label="RA active", linestyle="None"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="gray",
               markeredgecolor="#333", markersize=9, label="Top TRACE candidate", linestyle="None"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8.5,
              framealpha=0.85, labelcolor=TEXT,
              facecolor="white", edgecolor=SPINE)

    # ── Labels ────────────────────────────────────────────────────────────────
    ax.set_xlabel("UMAP 1", color=TEXT, fontsize=10)
    ax.set_ylabel("UMAP 2", color=TEXT, fontsize=10)
    ax.tick_params(colors=TEXT, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(SPINE)
    ax.set_title(
        "L1000 Drug Landscape — 978 Landmark Genes\n"
        "Color: IPF reversal score   ·   1,768 drug signatures",
        color=TEXT, fontsize=11, pad=12,
    )

    plt.tight_layout(pad=1.5)
    for ext in ("png", "svg"):
        out = FIG_DIR / f"umap_l1000_ipf.{ext}"
        plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"  Saved → {out.relative_to(ROOT)}")
    plt.close()


def main() -> None:
    print("Loading L1000 drug matrix ...")
    drug_df, mat = load_matrix()
    drug_names   = list(drug_df.columns)
    print(f"  {mat.shape[1]} genes × {mat.shape[0]} drugs")

    print("Loading reversal scores ...")
    scores = load_scores(drug_names)

    print("Loading known actives ...")
    ipf_actives, ra_actives = load_actives()

    print("Computing UMAP ...")
    embedding = compute_umap(mat)

    print("Rendering figure ...")
    make_figure(embedding, drug_names, scores, ipf_actives, ra_actives)
    print("Done.")


if __name__ == "__main__":
    main()
