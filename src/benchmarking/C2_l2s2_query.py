"""
Enrichr / L2S2 expanded drug universe query (C2).

Queries the MaayanLab Enrichr API with the IPF consensus up/down gene sets
against LINCS L1000 drug perturbation libraries, scoring reversal across a
much larger drug space than the 1,768-drug Phase II L1000 matrix.

Primary API: Enrichr (maayanlab.cloud/Enrichr) — mature, well-maintained
  Libraries used:
    LINCS_L1000_Chemical_Perturbations_2020  (drugs, reversed query)
    LINCS_L1000_CRISPR_KO_2023              (used for C3, not here)
  Strategy: query DOWN-regulated IPF genes → find drug signatures in library
  that OVERLAP with DOWN genes (those drugs induce what IPF suppresses → reversal)
  Query UP-regulated IPF genes → find signatures that suppress what IPF induces.

Fallback: L2S2 GraphQL term search (https://l2s2.maayanlab.cloud/graphql)

Outputs:
  results/l2s2/enrichr_up_results.json
  results/l2s2/enrichr_dn_results.json
  results/l2s2/l2s2_consensus_scores.csv
  results/l2s2/l2s2_novel_candidates.csv
  results/l2s2/l2s2_overlap_validation.csv
  results/l2s2/l2s2_priority_candidate_ranks.txt
  results/l2s2/l2s2_correlation_report.txt

Usage:
  python src/benchmarking/C2_l2s2_query.py
"""

import io
import json
import mimetypes
import re
import time
import uuid
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT    = Path(__file__).resolve().parents[2]
META    = ROOT / "results/meta"
REV     = ROOT / "results/reversal"
OUT     = ROOT / "results/l2s2"
OUT.mkdir(parents=True, exist_ok=True)

ENRICHR_URL  = "https://maayanlab.cloud/Enrichr"
CHEM_UP_LIB  = "LINCS_L1000_Chem_Pert_up"    # drugs that UP-regulate query genes
CHEM_DN_LIB  = "LINCS_L1000_Chem_Pert_down"  # drugs that DOWN-regulate query genes
N_TOP        = 150

PRIORITY_DRUGS = ["romidepsin", "JNJ-26481585", "dasatinib",
                  "atorvastatin", "pitavastatin", "trametinib", "nintedanib"]


def build_query_genes(sig_col: str = "meta_log2FC") -> tuple[list[str], list[str]]:
    """Return top-150 UP and DOWN gene SYMBOLS from IPF consensus."""
    cons = pd.read_csv(META / "consensus_signature.csv", index_col=0)
    cons = cons[cons["meta_padj"] < 0.05].copy()

    # Map Entrez IDs → gene symbols using L1000 gene info (most reliable local source)
    l1k_gi = ROOT / "data/raw/l1000/GSE70138_Broad_LINCS_gene_info_2017-03-06.txt.gz"
    str_gi  = ROOT / "data/raw/string_human_info.txt.gz"
    sym_map: dict[str, str] = {}
    if l1k_gi.exists():
        gi = pd.read_csv(l1k_gi, sep="\t", usecols=["pr_gene_id", "pr_gene_symbol"])
        sym_map.update(dict(zip(gi["pr_gene_id"].astype(str), gi["pr_gene_symbol"])))
    if str_gi.exists():
        si = pd.read_csv(str_gi, sep="\t", compression="gzip",
                         usecols=["#string_protein_id", "preferred_name"]
                        ).rename(columns={"#string_protein_id": "sid", "preferred_name": "sym"})
        # STRING uses ENSP IDs, not Entrez — skip unless we have the Entrez map
    # If still missing many, try NCBI gene2accession approach (slow, skip for now)
    mapped = cons.index.astype(str).map(sym_map)
    cons = cons[mapped.notna()].copy()
    cons.index = mapped[mapped.notna()]
    cons = cons.sort_values(sig_col, ascending=False)
    up_genes = cons.index[:N_TOP].tolist()
    dn_genes  = cons.index[-N_TOP:].tolist()
    print(f"  {len(up_genes)} UP gene symbols, {len(dn_genes)} DOWN gene symbols (from L1K gene info)")
    return up_genes, dn_genes


def _multipart_body(fields: dict) -> tuple[bytes, str]:
    """Build a multipart/form-data body from a dict of text fields."""
    boundary = uuid.uuid4().hex
    lines = []
    for name, value in fields.items():
        lines += [
            f"--{boundary}".encode(),
            f'Content-Disposition: form-data; name="{name}"'.encode(),
            b"",
            value.encode() if isinstance(value, str) else value,
        ]
    lines.append(f"--{boundary}--".encode())
    body = b"\r\n".join(lines)
    return body, f"multipart/form-data; boundary={boundary}"


def enrichr_submit_list(genes: list, description: str) -> str | None:
    """Submit a gene list to Enrichr (multipart/form-data), return userListId or None."""
    body, ctype = _multipart_body({"list": "\n".join(genes), "description": description})
    for attempt in range(5):
        try:
            req = Request(f"{ENRICHR_URL}/addList", data=body,
                          headers={"Content-Type": ctype})
            with urlopen(req, timeout=60) as r:
                data = json.loads(r.read())
                uid = str(data.get("userListId", ""))
                if uid:
                    return uid
        except Exception as e:
            wait = 2 ** attempt
            print(f"  Submit attempt {attempt+1} failed ({e}); retry in {wait}s ...")
            time.sleep(wait)
    return None


def enrichr_enrich(user_list_id: str, library: str) -> list | None:
    """Run Enrichr enrichment for a submitted gene list against a library."""
    url = f"{ENRICHR_URL}/enrich?userListId={user_list_id}&backgroundType={library}"
    for attempt in range(5):
        try:
            with urlopen(url, timeout=120) as r:
                data = json.loads(r.read())
                return data.get(library, [])
        except Exception as e:
            wait = 2 ** attempt
            print(f"  Enrich attempt {attempt+1} failed ({e}); retry in {wait}s ...")
            time.sleep(wait)
    return None


def parse_enrichr_drug_term(term: str) -> str:
    """Extract drug name from Enrichr LINCS term.

    Format: 'LJP006 A549 24H-dasatinib-10' or 'LJP005 MCF10A 24H-NVP-AEW541-10'
    Strategy: take last whitespace-delimited token, strip the 'Xh-' prefix and
    trailing '-concentration' suffix.
    """
    parts = term.split()
    if not parts:
        return term.lower()
    last = parts[-1]  # e.g. '24H-dasatinib-10' or '3H-NVP-BEZ235-0.37'
    # Strip hour prefix: e.g. '24H-' or '3H-'
    last = re.sub(r'^\d+h-', '', last, flags=re.IGNORECASE)
    # Strip concentration suffix: trailing '-number' (possibly decimal)
    last = re.sub(r'-\d+(\.\d+)?$', '', last)
    return last.lower().strip()


def aggregate_enrichr_scores(results: list, direction: str) -> pd.DataFrame:
    """
    Parse Enrichr enrichment results.
    direction='dn': we submitted DOWN genes → drugs that induce them (positive reversal)
    direction='up': we submitted UP genes → drugs that suppress them (positive reversal)

    Enrichr result format: [rank, term, p-value, z-score, combined_score, overlapping_genes,
                             adjusted_p-value, old_p-value, old_adjusted_p-value]
    A negative z-score in Enrichr combined score means gene set is 'down' in drug signature.
    """
    records = []
    for row in results:
        if len(row) < 7:
            continue
        term        = str(row[1])
        p_val       = float(row[2])
        adj_p       = float(row[6])
        z           = float(row[3]) if len(row) > 3 else 0.0
        combined    = float(row[4]) if len(row) > 4 else 0.0
        overlap     = row[5] if len(row) > 5 else []
        drug        = parse_enrichr_drug_term(term)
        records.append({
            "drug": drug, "term": term, "pvalue": p_val, "adj_pvalue": adj_p,
            "zscore": z, "combined_score": combined,
            "n_overlap": len(overlap), "direction": direction,
        })
    return pd.DataFrame(records)


def main() -> None:
    up_cache  = OUT / "enrichr_up_results.json"
    dn_cache  = OUT / "enrichr_dn_results.json"

    # ── Step 1: Build query genes ─────────────────────────────────────────────
    print("Step 1: Building IPF consensus query genes ...")
    up_genes, dn_genes = build_query_genes()

    # Reversal strategy:
    # - Submit IPF DOWN genes → LINCS_L1000_Chem_Pert_UP lib → drugs that UP-regulate
    #   what IPF suppresses → these are reversal candidates
    # - Submit IPF UP genes → LINCS_L1000_Chem_Pert_DOWN lib → drugs that DOWN-regulate
    #   what IPF induces → also reversal candidates

    if not dn_cache.exists():
        print(f"\nStep 2a: DOWN genes → Enrichr {CHEM_UP_LIB} ...")
        uid_dn = enrichr_submit_list(dn_genes, "IPF_consensus_DOWN")
        if uid_dn:
            print(f"  UserListId (DN→UP_lib): {uid_dn}")
            dn_results = enrichr_enrich(uid_dn, CHEM_UP_LIB)
            if dn_results is not None and len(dn_results) > 0:
                dn_cache.write_text(json.dumps(dn_results))
                print(f"  {len(dn_results)} results cached")
            else:
                dn_cache.write_text("[]")
                print("  No results")
        else:
            dn_cache.write_text("[]"); print("  Submission failed")
        time.sleep(1)
    else:
        print(f"[cached] {dn_cache.name}")

    if not up_cache.exists():
        print(f"\nStep 2b: UP genes → Enrichr {CHEM_DN_LIB} ...")
        uid_up = enrichr_submit_list(up_genes, "IPF_consensus_UP")
        if uid_up:
            print(f"  UserListId (UP→DN_lib): {uid_up}")
            up_results = enrichr_enrich(uid_up, CHEM_DN_LIB)
            if up_results is not None and len(up_results) > 0:
                up_cache.write_text(json.dumps(up_results))
                print(f"  {len(up_results)} results cached")
            else:
                up_cache.write_text("[]"); print("  No results")
        else:
            up_cache.write_text("[]"); print("  Submission failed")
    else:
        print(f"[cached] {up_cache.name}")

    dn_raw = json.loads(dn_cache.read_text())
    up_raw = json.loads(up_cache.read_text())
    # Enrichr wraps results under the library name key when returned via enrich endpoint
    dn_results = dn_raw if isinstance(dn_raw, list) else dn_raw.get(CHEM_UP_LIB, [])
    up_results = up_raw if isinstance(up_raw, list) else up_raw.get(CHEM_DN_LIB, [])

    if not dn_results and not up_results:
        print("\nERROR: No Enrichr results available. Check API access.")
        for f in ["l2s2_consensus_scores.csv", "l2s2_novel_candidates.csv",
                  "l2s2_overlap_validation.csv"]:
            (OUT / f).write_text("drug,reversal_score\n")
        (OUT / "l2s2_priority_candidate_ranks.txt").write_text("API unavailable.\n")
        (OUT / "l2s2_correlation_report.txt").write_text("API unavailable.\n")
        return

    # ── Step 4: Parse and combine ─────────────────────────────────────────────
    print("\nStep 3: Parsing enrichment results ...")
    df_dn = aggregate_enrichr_scores(dn_results, "dn")
    df_up = aggregate_enrichr_scores(up_results, "up")
    all_df = pd.concat([df_dn, df_up], ignore_index=True)
    print(f"  DN results: {len(df_dn)} terms, UP results: {len(df_up)} terms")

    # Compute a reversal score per drug:
    # For DN query: high combined_score → drug signature overlaps suppressed genes → induction
    # For UP query: high combined_score → drug signature overlaps activated genes → suppression
    # Both directions contribute to reversal; we average their combined scores
    def reversal_score(grp):
        dn_score = grp[grp["direction"] == "dn"]["combined_score"].max() if (grp["direction"] == "dn").any() else 0
        up_score = grp[grp["direction"] == "up"]["combined_score"].max() if (grp["direction"] == "up").any() else 0
        # Geometric-mean-like composite
        return float(np.sqrt(max(dn_score, 0) * max(up_score, 0)) if (dn_score > 0 and up_score > 0)
                     else max(dn_score, up_score))

    drug_scores = (all_df.groupby("drug")
                   .apply(reversal_score, include_groups=False)
                   .reset_index()
                   .rename(columns={0: "reversal_score"})
                   .sort_values("reversal_score", ascending=False))

    # Also add best p-values
    pvals = all_df.groupby("drug")["adj_pvalue"].min().reset_index().rename(columns={"adj_pvalue": "best_adj_pvalue"})
    drug_scores = drug_scores.merge(pvals, on="drug", how="left")
    drug_scores["l2s2_rank"] = range(1, len(drug_scores) + 1)

    drug_scores.to_csv(OUT / "l2s2_consensus_scores.csv", index=False)
    print(f"\nTop-20 Enrichr/L2S2 reversal candidates:")
    for _, row in drug_scores.head(20).iterrows():
        print(f"  {row['drug']:<30} score={row['reversal_score']:.2f}  adj_p={row['best_adj_pvalue']:.3e}")

    # ── Step 5: Cross-reference with TRACE ────────────────────────────────────
    print("\nStep 4: Cross-referencing with TRACE FDR results ...")
    trace_fdr_path = REV / "extended_fdr_results.csv"
    if trace_fdr_path.exists():
        trace_fdr  = pd.read_csv(trace_fdr_path).set_index("drug")
        overlap    = drug_scores[drug_scores["drug"].isin(trace_fdr.index.str.lower())]
        l2s2_nt    = drug_scores[~drug_scores["drug"].isin(
            pd.read_csv(REV / "trace_scores.csv")["drug"].str.lower())].head(100)
        overlap.to_csv(OUT / "l2s2_overlap_validation.csv", index=False)
        l2s2_nt.to_csv(OUT / "l2s2_novel_candidates.csv", index=False)
        print(f"  In both TRACE+L2S2: {len(overlap)}, novel L2S2 top-100: {len(l2s2_nt)}")
    else:
        drug_scores.head(100).to_csv(OUT / "l2s2_novel_candidates.csv", index=False)
        drug_scores.to_csv(OUT / "l2s2_overlap_validation.csv", index=False)

    # ── Step 6: Priority candidate ranks ─────────────────────────────────────
    name_map = dict(zip(drug_scores["drug"], drug_scores["l2s2_rank"]))
    score_map = dict(zip(drug_scores["drug"], drug_scores["reversal_score"]))
    lines = [f"Priority candidate ranks in Enrichr/L2S2 (out of {len(drug_scores):,} drugs):"]
    for d in PRIORITY_DRUGS:
        rank  = name_map.get(d.lower(), "not found")
        score = score_map.get(d.lower(), float("nan"))
        lines.append(f"  {d:<30} rank={rank}  score={score:.2f}" if isinstance(rank, int)
                     else f"  {d:<30} {rank}")
    rank_txt = "\n".join(lines)
    (OUT / "l2s2_priority_candidate_ranks.txt").write_text(rank_txt)
    print("\n" + rank_txt)

    # ── Step 7: Spearman vs Net-TRACE ─────────────────────────────────────────
    trace_all = pd.read_csv(REV / "trace_scores.csv")
    trace_all["drug_lower"] = trace_all["drug"].str.lower()
    common = set(drug_scores["drug"]) & set(trace_all["drug_lower"])
    if len(common) > 10:
        l2s2_r  = drug_scores.set_index("drug").loc[list(common), "l2s2_rank"].values
        trace_r = trace_all.set_index("drug_lower").loc[list(common), "trace_rank"].values
        rho, pval = spearmanr(l2s2_r, trace_r)
        corr_txt = (f"Spearman (Enrichr/L2S2 rank vs Net-TRACE rank)\n"
                    f"  n_common: {len(common)}\n  rho: {rho:.4f}\n  p: {pval:.3e}\n")
    else:
        corr_txt = f"Too few drugs in common ({len(common)}) for correlation.\n"
    (OUT / "l2s2_correlation_report.txt").write_text(corr_txt)
    print("\n" + corr_txt)
    print(f"C2 complete. Outputs → {OUT}")


if __name__ == "__main__":
    main()
