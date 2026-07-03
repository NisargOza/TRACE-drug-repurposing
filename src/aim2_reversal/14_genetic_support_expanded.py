
from pathlib import Path
import json
import time
import pandas as pd
import numpy as np
from urllib.request import urlopen, Request
from urllib.parse import urlencode
from urllib.error import URLError, HTTPError

ROOT = Path(__file__).resolve().parents[2]
REV  = ROOT / "results" / "reversal"

CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"
OT_REST     = "https://api.platform.opentargets.org/api/v4/rest"

BATCH_SIZE   = 50
OT_DISEASE   = "EFO_0000768"


def fetch_json(url: str, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            with urlopen(url, timeout=20) as r:
                return json.loads(r.read())
        except HTTPError as e:
            if e.code == 429:
                time.sleep(2 ** attempt)
            else:
                return {}
        except (URLError, Exception):
            time.sleep(1)
    return {}


def chembl_name_to_id(drug_name: str) -> str | None:
    q = urlencode({"q": drug_name, "format": "json", "limit": 1})
    url = f"{CHEMBL_BASE}/molecule/search?{q}"
    data = fetch_json(url)
    mols = data.get("molecules", [])
    if mols:
        return mols[0].get("molecule_chembl_id")
    return None


def chembl_targets_for_drug(chembl_id: str) -> list[str]:
    url = f"{CHEMBL_BASE}/mechanism?molecule_chembl_id={chembl_id}&format=json&limit=100"
    data = fetch_json(url)
    mechs = data.get("mechanisms", [])
    target_ids = set()
    for m in mechs:
        tc = m.get("target_chembl_id")
        if tc:
            target_ids.add(tc)

    url2 = f"{CHEMBL_BASE}/activity?molecule_chembl_id={chembl_id}&format=json&limit=200&assay_type=B"
    data2 = fetch_json(url2)
    acts = data2.get("activities", [])
    for a in acts:
        tc = a.get("target_chembl_id")
        if tc:
            target_ids.add(tc)

    uniprot_ids = []
    for tc in list(target_ids)[:20]:
        url3 = f"{CHEMBL_BASE}/target/{tc}?format=json"
        tdata = fetch_json(url3)
        comps = tdata.get("target_components", [])
        for comp in comps:
            for xref in comp.get("target_component_xrefs", []):
                if xref.get("xref_src_db") in ("UniProt", "SwissProt"):
                    uniprot_ids.append(xref["xref_id"])
        time.sleep(0.1)

    return list(set(uniprot_ids))


def uniprot_to_ensembl(uniprot_ids: list[str]) -> list[str]:
    ensembl_ids = []
    for uid in uniprot_ids[:10]:
        url = f"https://rest.uniprot.org/uniprotkb/{uid}?format=json"
        try:
            data = fetch_json(url)
            for db_ref in data.get("uniProtKBCrossReferences", []):
                if db_ref.get("database") == "Ensembl":
                    for prop in db_ref.get("properties", []):
                        if prop.get("key") == "GeneId":
                            ensembl_ids.append(prop["value"].split(".")[0])
            time.sleep(0.15)
        except Exception:
            pass
    return list(set(ensembl_ids))


def ot_ipf_score(ensembl_ids: list[str]) -> float:
    max_score = 0.0
    for eid in ensembl_ids[:10]:
        url = f"{OT_REST}/association/filter?targetId={eid}&diseaseId={OT_DISEASE}&size=1"
        data = fetch_json(url)
        for hit in data.get("data", []):
            s = hit.get("overallScore", 0) or 0
            max_score = max(max_score, s)
        time.sleep(0.15)
    return max_score


def main():
    cand = pd.read_csv(REV / "final_candidates_full.csv")
    existing_cache = {}
    cache_path = REV / "expanded_drug_targets.json"
    if cache_path.exists():
        existing_cache = json.load(open(cache_path))

    print(f"Drugs to process: {len(cand)}")
    print(f"Already cached: {len(existing_cache)}")

    no_support = cand[cand["genetic_support"] == 0]["drug"].tolist()
    print(f"Drugs with zero genetic support: {len(no_support)}")

    MAX_NEW = 200
    to_process = [d for d in no_support if d not in existing_cache][:MAX_NEW]
    print(f"Processing {len(to_process)} new drugs this run...")

    for i, drug in enumerate(to_process):
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(to_process)} ...")

        chembl_id = chembl_name_to_id(drug)
        if not chembl_id:
            existing_cache[drug] = {"chembl_id": None, "uniprot": [], "ensembl": [], "ot_score": 0.0}
            time.sleep(0.2)
            continue

        uniprot = chembl_targets_for_drug(chembl_id)
        time.sleep(0.2)

        ensembl = uniprot_to_ensembl(uniprot)
        time.sleep(0.2)

        score = ot_ipf_score(ensembl) if ensembl else 0.0
        existing_cache[drug] = {
            "chembl_id": chembl_id,
            "uniprot": uniprot[:10],
            "ensembl": ensembl[:10],
            "ot_score": score,
        }
        time.sleep(0.3)

    json.dump(existing_cache, open(cache_path, "w"), indent=2)

    def get_score(drug: str, existing_gen: float) -> float:
        if existing_gen > 0:
            return existing_gen
        entry = existing_cache.get(drug, {})
        return float(entry.get("ot_score", 0.0))

    cand["genetic_support_expanded"] = [
        get_score(r["drug"], r["genetic_support"])
        for _, r in cand.iterrows()
    ]

    n_orig = (cand["genetic_support"] > 0).sum()
    n_new  = (cand["genetic_support_expanded"] > 0).sum()
    print(f"\nGenetic support coverage: {n_orig} → {n_new} drugs ({n_new/len(cand)*100:.1f}%)")

    w_nt = 0.50; w_gen = 0.30; w_rep = 0.20
    cand["combined_score_expanded"] = (
        cand["net_trace"]                  * w_nt +
        cand["genetic_support_expanded"]   * w_gen +
        cand["sig_reproducibility"]        * w_rep
    )
    cand = cand.sort_values("combined_score_expanded", ascending=False).reset_index(drop=True)
    cand["combined_rank_expanded"] = cand.index + 1

    cand.to_csv(REV / "genetic_support_expanded.csv", index=False)

    print("\nRank changes after expanded genetic support:")
    print(f"{'Drug':<20} {'Old rank':>10} {'New rank':>10} {'Change':>8}")
    for drug in ["nintedanib", "pirfenidone", "cediranib", "romidepsin",
                 "dasatinib", "atorvastatin", "baricitinib"]:
        old_r = cand[cand["drug"].str.lower() == drug]["combined_rank"].values
        new_r = cand[cand["drug"].str.lower() == drug]["combined_rank_expanded"].values
        if len(old_r) and len(new_r):
            delta = int(old_r[0]) - int(new_r[0])
            print(f"  {drug:<18} {int(old_r[0]):>10} {int(new_r[0]):>10} {delta:>+8}")

    print(f"\nSaved genetic_support_expanded.csv, expanded_drug_targets.json")


if __name__ == "__main__":
    main()
