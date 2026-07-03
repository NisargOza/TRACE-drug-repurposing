
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

ROOT    = Path(__file__).resolve().parents[2]
L1K     = ROOT / "results/l1000"
FIG_DIR = ROOT / "results/figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

N_GENES = 60
N_DRUGS = 40


def main() -> None:
    print("Loading L1000 matrix ...")
    df  = pd.read_csv(L1K / "drug_signatures_landmark.csv.gz", index_col=0)
    mat = df.values.astype(np.float32)

    mean = mat.mean(axis=1, keepdims=True)
    std  = mat.std(axis=1, keepdims=True) + 1e-8
    mat_z = (mat - mean) / std

    rng      = np.random.default_rng(42)
    var_rank = np.argsort(mat_z.var(axis=1))[::-1]
    gene_idx = var_rank[:N_GENES]
    drug_idx = np.sort(rng.choice(mat_z.shape[1], size=N_DRUGS, replace=False))

    slice_mat = mat_z[np.ix_(gene_idx, drug_idx)]

    gene_info_path = ROOT / "data/raw/l1000/GSE70138_Broad_LINCS_gene_info_2017-03-06.txt.gz"
    if gene_info_path.exists():
        gi = pd.read_csv(gene_info_path, sep="\t", usecols=["pr_gene_id", "pr_gene_symbol"])
        sym_map = dict(zip(gi["pr_gene_id"].astype(str), gi["pr_gene_symbol"]))
        gene_labels = [sym_map.get(str(g), str(g)) for g in df.index[gene_idx]]
    else:
        gene_labels = df.index[gene_idx].tolist()

    print(f"  Slice: {N_GENES} genes × {N_DRUGS} drugs")

    fig, ax = plt.subplots(figsize=(4.2, 5.8), dpi=150)
    fig.patch.set_facecolor("white")

    im = ax.imshow(
        slice_mat,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-2.5, vmax=2.5,
        interpolation="nearest",
    )

    ax.set_yticks(range(N_GENES))
    ax.set_yticklabels(gene_labels, fontsize=4.2, color="#222")
    ax.yaxis.tick_left()

    ax.set_xticks([])
    ax.set_xlabel(f"{N_DRUGS} drug perturbations (sample)", fontsize=7, color="#444", labelpad=4)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03, shrink=0.6)
    cbar.set_label("expression z-score", fontsize=6.5, color="#444")
    cbar.ax.tick_params(labelsize=6, colors="#444")
    cbar.outline.set_edgecolor("#ccc")

    ax.set_title(
        f"L1000 landmark-gene signatures\n{df.shape[0]} genes × {df.shape[1]:,} drugs",
        fontsize=7.5, color="#222", pad=6,
    )

    for spine in ax.spines.values():
        spine.set_edgecolor("#ccc")

    plt.tight_layout(pad=0.8)
    for ext in ("png", "svg"):
        out = FIG_DIR / f"l1000_input_heatmap.{ext}"
        plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"  Saved → {out.relative_to(ROOT)}")
    plt.close()
    print("Done.")


if __name__ == "__main__":
    main()
