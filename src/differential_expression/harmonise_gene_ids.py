
from pathlib import Path
import pandas as pd

DE_DIR = Path("results/de")


def harmonise_with_map(acc: str, de_csv: Path, map_csv: Path,
                       query_col: str, entrez_col: str) -> None:
    out = DE_DIR / f"{acc}_de_entrez.csv"
    if out.exists():
        print(f"  [skip] {out.name}")
        return

    df = pd.read_csv(de_csv, index_col=0)
    df.index = df.index.astype(str).str.strip()

    id_map_df = pd.read_csv(map_csv)
    id_map = dict(zip(
        id_map_df[query_col].astype(str),
        id_map_df[entrez_col].astype(str)
    ))

    df["entrez_id"] = df.index.map(id_map)
    df = df.dropna(subset=["entrez_id"])
    df = df.sort_values("padj").drop_duplicates(subset="entrez_id")
    df = df.set_index("entrez_id")
    df.to_csv(out)
    print(f"  {acc}: {len(df):,} genes mapped to Entrez")


def harmonise_entrez(acc: str, de_csv: Path) -> None:
    out = DE_DIR / f"{acc}_de_entrez.csv"
    if out.exists():
        print(f"  [skip] {out.name}")
        return
    df = pd.read_csv(de_csv, index_col=0)
    df.index = df.index.astype(str).str.strip()
    df.index.name = "entrez_id"
    df.to_csv(out)
    print(f"  {acc}: {len(df):,} genes (already Entrez, copied)")


def main() -> None:
    harmonise_with_map(
        "GSE213001",
        DE_DIR / "GSE213001_de_results.csv",
        DE_DIR / "GSE213001_ensembl2entrez.csv",
        query_col="ENSEMBL", entrez_col="ENTREZID",
    )
    harmonise_with_map(
        "GSE150910",
        DE_DIR / "GSE150910_de_results.csv",
        DE_DIR / "GSE150910_symbol2entrez.csv",
        query_col="SYMBOL", entrez_col="ENTREZID",
    )
    harmonise_entrez("GSE38958", DE_DIR / "GSE38958_de_results.csv")
    harmonise_entrez("GSE53845", DE_DIR / "GSE53845_de_results.csv")

    print("\nDone. Run 06_meta_analysis.py next.")


if __name__ == "__main__":
    main()
