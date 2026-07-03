
import json
import re
import time
import uuid
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

import numpy as np
import pandas as pd

ROOT  = Path(__file__).resolve().parents[2]
META  = ROOT / "results/meta"
OUT   = ROOT / "results/crispr"
SCRNA = ROOT / "results/scrna"
OUT.mkdir(parents=True, exist_ok=True)

ENRICHR_URL   = "https://maayanlab.cloud/Enrichr"
KO_UP_LIB     = "LINCS_L1000_CRISPR_KO_Consensus_Sigs"
OT_GQL_URL    = "https://api.platform.opentargets.org/api/v4/graphql"
N_TOP         = 150


def build_query_genes() -> tuple[list[str], list[str]]:
    cons = pd.read_csv(META / "consensus_signature.csv", index_col=0)
    cons = cons[cons["meta_padj"] < 0.05].copy()
    l1k_gi = ROOT / "data/raw/l1000/GSE70138_Broad_LINCS_gene_info_2017-03-06.txt.gz"
    if l1k_gi.exists():
        gi = pd.read_csv(l1k_gi, sep="\t", usecols=["pr_gene_id", "pr_gene_symbol"])
        sym_map = dict(zip(gi["pr_gene_id"].astype(str), gi["pr_gene_symbol"]))
        mapped = cons.index.astype(str).map(sym_map)
        cons = cons[mapped.notna()].copy(); cons.index = mapped[mapped.notna()]
    else:
        cons.index = cons.index.astype(str)
    cons = cons.sort_values("meta_log2FC", ascending=False)
    up_genes = cons.index[:N_TOP].tolist()
    dn_genes  = cons.index[-N_TOP:].tolist()
    print(f"  {len(up_genes)} UP, {len(dn_genes)} DOWN gene symbols")
    return up_genes, dn_genes


def _multipart_body(fields: dict) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    lines = []
    for name, value in fields.items():
        lines += [f"--{boundary}".encode(), f'Content-Disposition: form-data; name="{name}"'.encode(),
                  b"", (value.encode() if isinstance(value, str) else value)]
    lines.append(f"--{boundary}--".encode())
    return b"\r\n".join(lines), f"multipart/form-data; boundary={boundary}"


def enrichr_submit(genes: list, description: str) -> str | None:
    body, ctype = _multipart_body({"list": "\n".join(genes), "description": description})
    for attempt in range(5):
        try:
            req = Request(f"{ENRICHR_URL}/addList", data=body, headers={"Content-Type": ctype})
            with urlopen(req, timeout=60) as r:
                uid = str(json.loads(r.read()).get("userListId", ""))
                if uid: return uid
        except Exception as e:
            wait = 2 ** attempt; print(f"  attempt {attempt+1} failed ({e}); retry {wait}s ..."); time.sleep(wait)
    return None


def enrichr_enrich(uid: str, library: str) -> list:
    for attempt in range(5):
        try:
            url = f"{ENRICHR_URL}/enrich?userListId={uid}&backgroundType={library}"
            with urlopen(url, timeout=120) as r:
                data = json.loads(r.read())
                return data if isinstance(data, list) else data.get(library, [])
        except Exception as e:
            wait = 2 ** attempt; print(f"  enrich attempt {attempt+1} failed ({e}); retry {wait}s ..."); time.sleep(wait)
    return []


def parse_ko_term(term: str) -> str:
    return term.split()[0].strip().upper() if term else term


def aggregate_ko_scores(dn_results: list, up_results: list) -> pd.DataFrame:
    from scipy.stats import combine_pvalues
    records_dn, records_up = [], []
    for row in dn_results:
        if len(row) >= 5:
            gene = parse_ko_term(str(row[1]))
            records_dn.append({"gene": gene, "score_dn": float(row[4]), "pval_dn": float(row[2]), "adj_p_dn": float(row[6])})
    for row in up_results:
        if len(row) >= 5:
            gene = parse_ko_term(str(row[1]))
            records_up.append({"gene": gene, "score_up": float(row[4]), "pval_up": float(row[2]), "adj_p_up": float(row[6])})
    df_dn = pd.DataFrame(records_dn).groupby("gene").agg(score_dn=("score_dn","max"), adj_p_dn=("adj_p_dn","min")).reset_index()
    df_up = pd.DataFrame(records_up).groupby("gene").agg(score_up=("score_up","max"), adj_p_up=("adj_p_up","min")).reset_index()
    merged = pd.merge(df_dn, df_up, on="gene", how="outer").fillna(0)
    merged["reversal_score"] = (merged["score_dn"].clip(lower=0) * merged["score_up"].clip(lower=0)) ** 0.5
    merged = merged.sort_values("reversal_score", ascending=False).reset_index(drop=True)
    return merged


def query_open_targets(gene_symbols: list) -> dict:
    gql = """
    query tractability($sym: String!) {
      targetByApprovedSymbol(symbol: $sym) {
        id approvedSymbol
        tractability { label modality value }
      }
    }"""
    results = {}
    for sym in gene_symbols[:30]:
        try:
            payload = json.dumps({"query": gql, "variables": {"sym": sym}}).encode()
            req = Request(OT_GQL_URL, data=payload, headers={"Content-Type": "application/json"})
            with urlopen(req, timeout=30) as r:
                data = json.loads(r.read())
            target = (data.get("data") or {}).get("targetByApprovedSymbol")
            if target:
                results[sym] = {"ensembl_id": target.get("id",""), "tractability": target.get("tractability", [])}
            time.sleep(0.2)
        except Exception as e:
            print(f"    OT query failed for {sym}: {e}")
    return results


def is_druggable(tractability: list) -> bool:
    for item in tractability:
        if (item.get("modality", "").lower() in ("small_molecule", "smallmolecule")
                and item.get("value") is True):
            return True
    return False


def main() -> None:
    dn_cache = OUT / "enrichr_ko_dn.json"
    up_cache = OUT / "enrichr_ko_up.json"

    print("Step 1: Building query genes ...")
    up_genes, dn_genes = build_query_genes()

    if not dn_cache.exists():
        print(f"\nStep 2a: DOWN genes → Enrichr {KO_UP_LIB} ...")
        uid = enrichr_submit(dn_genes, "IPF_DN_for_KO")
        if uid:
            dn_results = enrichr_enrich(uid, KO_UP_LIB)
            dn_cache.write_text(json.dumps(dn_results))
            print(f"  {len(dn_results)} results")
        else:
            dn_cache.write_text("[]"); print("  Submission failed")
        time.sleep(1)
    else:
        print(f"[cached] {dn_cache.name}")

    if not up_cache.exists():
        print(f"\nStep 2b: UP genes → Enrichr {KO_UP_LIB} ...")
        uid = enrichr_submit(up_genes, "IPF_UP_for_KO")
        if uid:
            up_results = enrichr_enrich(uid, KO_UP_LIB)
            up_cache.write_text(json.dumps(up_results))
            print(f"  {len(up_results)} results")
        else:
            up_cache.write_text("[]"); print("  Submission failed")
    else:
        print(f"[cached] {up_cache.name}")

    dn_raw = json.loads(dn_cache.read_text())
    up_raw = json.loads(up_cache.read_text())
    dn_results = dn_raw if isinstance(dn_raw, list) else []
    up_results = up_raw if isinstance(up_raw, list) else []

    if not dn_results and not up_results:
        print("No CRISPR KO enrichment data available.")
        for f in ["crispr_reversal_scores.csv","crispr_druggable_targets.csv","crispr_priority_targets.csv"]:
            (OUT / f).write_text("gene,reversal_score\n")
        (OUT / "crispr_summary.txt").write_text("No CRISPR KO data.\n"); return

    print("\nStep 3: Aggregating KO reversal scores ...")
    crispr_df = aggregate_ko_scores(dn_results, up_results)
    crispr_df.to_csv(OUT / "crispr_reversal_scores.csv", index=False)
    print(f"  {len(crispr_df)} target genes scored")
    print("\nTop-20 CRISPR reversal targets:")
    for _, row in crispr_df.head(20).iterrows():
        print(f"  {row['gene']:<15} score={row['reversal_score']:.2f}  dn={row.get('score_dn',0):.1f}  up={row.get('score_up',0):.1f}")

    print("\nStep 3: Querying Open Targets tractability ...")
    top_genes = crispr_df.head(50)["gene"].tolist()
    ot_cache  = OUT / "crispr_open_targets.json"
    if not ot_cache.exists():
        ot_data = query_open_targets(top_genes)
        ot_cache.write_text(json.dumps(ot_data, indent=2))
    else:
        print(f"  [cached] {ot_cache.name}")
        ot_data = json.loads(ot_cache.read_text())

    crispr_df["druggable"] = crispr_df["gene"].apply(
        lambda g: is_druggable(ot_data.get(g, {}).get("tractability", [])))
    druggable = crispr_df[crispr_df["druggable"]].copy()
    druggable.to_csv(OUT / "crispr_druggable_targets.csv", index=False)
    print(f"\n  Druggable targets (SM tractability): {len(druggable)}/{len(crispr_df)}")

    cons = pd.read_csv(META / "consensus_signature.csv", index_col=0)
    gene_info = ROOT / "data/raw/gene_info_human.tsv"
    if gene_info.exists():
        gi = pd.read_csv(gene_info, sep="\t", dtype=str, usecols=["GeneID", "Symbol"])
        gi = gi.set_index("GeneID")["Symbol"]
        cons_syms = cons.index.astype(str).map(gi).dropna()
    else:
        cons_syms = pd.Series(dtype=str)

    in_consensus = crispr_df["gene"].isin(set(cons_syms.values))
    crispr_df["in_ipf_consensus"] = in_consensus

    scrna_sig = SCRNA / "at2_at1_transition_signature.csv"
    if scrna_sig.exists() and gene_info.exists():
        sc = pd.read_csv(scrna_sig)
        sc_up = set(sc[sc["log2FC"] > 0.5]["gene_symbol"].dropna())
        crispr_df["in_scrna_AT1_up"] = crispr_df["gene"].isin(sc_up)
    else:
        crispr_df["in_scrna_AT1_up"] = False

    crispr_df["priority_score"] = (
        crispr_df["druggable"].astype(int) * 3
        + crispr_df["in_ipf_consensus"].astype(int) * 2
        + crispr_df["in_scrna_AT1_up"].astype(int) * 1
        + crispr_df["reversal_score"].clip(0) / (crispr_df["reversal_score"].max() + 1e-9)
    )
    priority = crispr_df.sort_values("priority_score", ascending=False).head(20)
    priority.to_csv(OUT / "crispr_priority_targets.csv", index=False)

    lines = ["CRISPR KO reversal targets — priority candidates:",
             f"  Total KO genes scored: {len(crispr_df)}",
             f"  Druggable (small molecule): {len(druggable)}",
             "", "Top-10 priority targets:"]
    for _, row in priority.head(10).iterrows():
        flags = []
        if row["druggable"]:        flags.append("druggable")
        if row["in_ipf_consensus"]: flags.append("in_IPF_DE")
        if row["in_scrna_AT1_up"]:  flags.append("AT2→AT1_up")
        lines.append(f"  {row['gene']:<15} score={row['reversal_score']:.1f}  [{', '.join(flags)}]")
    txt = "\n".join(lines)
    (OUT / "crispr_summary.txt").write_text(txt)
    print("\n" + txt)
    print(f"\nC3 complete. Outputs → {OUT}")


if __name__ == "__main__":
    main()
