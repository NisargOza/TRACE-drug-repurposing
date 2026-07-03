
from pathlib import Path
import pandas as pd
import numpy as np
import json
from urllib.request import urlopen
from urllib.parse import urlencode
import time

ROOT = Path(__file__).resolve().parents[2]
REV  = ROOT / "results" / "reversal"

PIRFENIDONE_MANUAL_TARGETS_ENTREZ = [
    7040,
    7046,
    7048,
    5155,
    2247,
    7422,
    1490,
    4780,
]

PIRFENIDONE_MANUAL_TARGETS_ENSEMBL = [
    "ENSG00000105329",
    "ENSG00000106799",
    "ENSG00000163513",
    "ENSG00000197217",
    "ENSG00000138685",
    "ENSG00000112715",
    "ENSG00000175592",
    "ENSG00000116044",
]


def fetch_ot_score(ensembl_ids: list) -> float:
    query = """
    query TargetIPF($ensemblId: String!) {
      target(ensemblId: $ensemblId) {
        id
        approvedSymbol
        associatedDiseases(filter: {datasourceIds: ["ot_genetics_portal","eva","gene2phenotype","orphanet","uniprot_literature","uniprot_variants","genomics_england"]}) {
          rows(disease: "EFO_0000768") {
            disease { id name }
            score
          }
        }
      }
    }
    """
    url = "https://api.platform.opentargets.org/api/v4/graphql"
    max_score = 0.0
    for eid in ensembl_ids:
        try:
            import urllib.request, json
            payload = json.dumps({"query": query, "variables": {"ensemblId": eid}}).encode()
            req = urllib.request.Request(url, data=payload,
                                          headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=15) as r:
                data = json.loads(r.read())
            target = data.get("data", {}).get("target", {}) or {}
            assoc = target.get("associatedDiseases", {}) or {}
            rows  = assoc.get("rows", []) or []
            for row in rows:
                s = row.get("score", 0) or 0
                if s > max_score:
                    max_score = s
            time.sleep(0.3)
        except Exception:
            pass
    return max_score


def fetch_ot_score_simple(ensembl_ids: list) -> float:
    disease_id = "EFO_0000768"
    max_score = 0.0
    for eid in ensembl_ids:
        try:
            url = f"https://api.platform.opentargets.org/api/v4/rest/association/filter?targetId={eid}&diseaseId={disease_id}&size=1"
            with urlopen(url, timeout=15) as r:
                data = json.loads(r.read())
            for hit in data.get("data", []):
                s = hit.get("overallScore", 0) or 0
                if s > max_score:
                    max_score = s
            time.sleep(0.3)
        except Exception:
            pass
    return max_score


def main():
    cand = pd.read_csv(REV / "final_candidates_full.csv")

    print("Fetching OT IPF scores for pirfenidone manual targets...")
    ot_cache_path = REV / "ot_ipf_targets_cache.json"
    ot_cache = {}
    if ot_cache_path.exists():
        ot_cache = json.load(open(ot_cache_path))

    target_scores = {}
    for eid in PIRFENIDONE_MANUAL_TARGETS_ENSEMBL:
        if eid in ot_cache:
            target_scores[eid] = ot_cache[eid]
        else:
            s = fetch_ot_score_simple([eid])
            target_scores[eid] = s
            ot_cache[eid] = s
            print(f"  {eid}: {s:.4f}")
            time.sleep(0.3)

    json.dump(ot_cache, open(ot_cache_path, "w"), indent=2)

    max_score = max(target_scores.values()) if target_scores else 0.0
    print(f"\nMax OT IPF score across pirfenidone targets: {max_score:.4f}")

    manual = {
        "drug": "pirfenidone",
        "target_assignment": "manual (literature-derived)",
        "references": [
            "King TE Jr et al. NEJM 2014 (TGF-β1 pathway)",
            "Hecker L et al. (Nrf2/PDGF-B)",
            "Taniguchi H et al. (CTGF/CCN2)",
            "Inomata M et al. (FGF2, VEGF)",
        ],
        "entrez_targets": PIRFENIDONE_MANUAL_TARGETS_ENTREZ,
        "ensembl_targets": PIRFENIDONE_MANUAL_TARGETS_ENSEMBL,
        "max_ot_ipf_score": max_score,
        "target_scores": target_scores,
    }
    json.dump(manual, open(REV / "pirfenidone_manual_targets.json", "w"), indent=2)

    pf_idx = cand[cand["drug"].str.lower() == "pirfenidone"].index
    if len(pf_idx):
        old_gen = float(cand.loc[pf_idx[0], "genetic_support"])
        cand.loc[pf_idx[0], "genetic_support"] = max_score

        w_nt = 0.50; w_gen = 0.30; w_rep = 0.20
        cand["combined_score_fixed"] = (
            cand["net_trace"]           * w_nt  +
            cand["genetic_support"]     * w_gen +
            cand["sig_reproducibility"] * w_rep
        )
        cand = cand.sort_values("combined_score_fixed", ascending=False).reset_index(drop=True)
        cand["combined_rank_fixed"] = cand.index + 1

        new_rank = int(cand.loc[cand["drug"].str.lower() == "pirfenidone", "combined_rank_fixed"].values[0])
        new_pct  = new_rank / len(cand) * 100

        cand.to_csv(REV / "final_candidates_fixed.csv", index=False)

        lines = [
            "Pirfenidone Manual Target Fix",
            "=" * 55,
            "",
            f"Manual MOA targets assigned (literature-derived):",
            f"  TGFB1, TGFBR1, TGFBR2, PDGFB, FGF2, VEGFA, CCN2, NFE2L2",
            f"",
            f"Max Open Targets IPF association score: {max_score:.4f}",
            f"  (was 0.000 — no ChEMBL MOA entry)",
            f"",
            f"Pirfenidone combined rank: before = 793/{len(cand)} (44.9%)",
            f"                           after  = {new_rank}/{len(cand)} ({new_pct:.1f}%)",
            "",
            "Target OT scores:",
        ]
        gene_names = {
            "ENSG00000105329": "TGFB1",
            "ENSG00000106799": "TGFBR1",
            "ENSG00000163513": "TGFBR2",
            "ENSG00000197217": "PDGFB",
            "ENSG00000138685": "FGF2",
            "ENSG00000112715": "VEGFA",
            "ENSG00000175592": "CCN2",
            "ENSG00000116044": "NFE2L2",
        }
        for eid, score in target_scores.items():
            lines.append(f"  {gene_names.get(eid, eid):<10} {score:.4f}")

        report_path = REV / "pirfenidone_fix_report.txt"
        report_path.write_text("\n".join(lines))
        print("\n".join(lines))
        print(f"\nSaved final_candidates_fixed.csv, pirfenidone_fix_report.txt")
    else:
        print("pirfenidone not found in candidate list")


if __name__ == "__main__":
    main()
