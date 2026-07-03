
import csv
import os
import time
from pathlib import Path

import requests

_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))

NCBI_API_KEY = os.environ.get("NCBI_API_KEY", "")
EMAIL = "nisargo09@outlook.com"
TOOL = "TRACE-IPF"
BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
SLEEP = 0.11 if NCBI_API_KEY else 0.34

QUERIES = [
    (
        "array",
        '("idiopathic pulmonary fibrosis"[All Fields]) '
        'AND "expression profiling by array"[Filter] '
        'AND "Homo sapiens"[Organism]',
    ),
    (
        "rnaseq",
        '"pulmonary fibrosis" '
        'AND "expression profiling by high throughput sequencing"[Filter] '
        'AND "Homo sapiens"[Organism]',
    ),
    (
        "broad",
        '("idiopathic pulmonary fibrosis" OR IPF) AND lung '
        'AND (control OR normal OR donor) AND "Homo sapiens"[Organism]',
    ),
]

RETMAX = 200


def _base_params() -> dict:
    p = {"email": EMAIL, "tool": TOOL}
    if NCBI_API_KEY:
        p["api_key"] = NCBI_API_KEY
    return p


def esearch(term: str) -> list[str]:
    params = {**_base_params(), "db": "gds", "term": term,
               "retmax": RETMAX, "retmode": "json"}
    r = requests.get(f"{BASE}/esearch.fcgi", params=params, timeout=30)
    r.raise_for_status()
    time.sleep(SLEEP)
    return r.json()["esearchresult"]["idlist"]


def esummary_batch(uids: list[str]) -> list[dict]:
    records = []
    for i in range(0, len(uids), 50):
        batch = uids[i : i + 50]
        params = {**_base_params(), "db": "gds",
                  "id": ",".join(batch), "retmode": "json"}
        r = requests.get(f"{BASE}/esummary.fcgi", params=params, timeout=30)
        r.raise_for_status()
        result = r.json().get("result", {})
        for uid in batch:
            rec = result.get(uid)
            if rec and isinstance(rec, dict):
                records.append(rec)
        time.sleep(SLEEP)
    return records


def parse_record(rec: dict) -> dict | None:
    if rec.get("entrytype", "") != "GSE":
        return None
    return {
        "accession": rec.get("accession", ""),
        "title": rec.get("title", "")[:120],
        "organism": rec.get("taxon", ""),
        "platform_type": rec.get("ptype", ""),
        "gpl": rec.get("gpl", ""),
        "n_samples_total": rec.get("n_samples", ""),
        "n_cases": "",
        "n_controls": "",
        "tissue": "",
        "has_healthy_ctrl": "",
        "notes": "",
        "summary_excerpt": rec.get("summary", "")[:200],
    }


def main() -> None:
    if NCBI_API_KEY:
        print(f"Using NCBI API key (10 req/s limit)")
    else:
        print("No NCBI_API_KEY found — using 3 req/s limit")

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "geo_dataset_inventory.csv"

    all_uids: dict[str, str] = {}

    for label, term in QUERIES:
        print(f"\n[{label}] Searching GEO...")
        uids = esearch(term)
        print(f"  {len(uids)} UIDs returned")
        for uid in uids:
            if uid not in all_uids:
                all_uids[uid] = label

    print(f"\nTotal unique UIDs across all queries: {len(all_uids)}")
    print("Fetching summaries...")
    records = esummary_batch(list(all_uids.keys()))

    rows = []
    for rec in records:
        parsed = parse_record(rec)
        if parsed is not None:
            parsed["query_source"] = all_uids.get(rec.get("uid", ""), "")
            rows.append(parsed)

    rows.sort(key=lambda r: r["accession"])

    fieldnames = [
        "accession", "title", "organism", "platform_type", "gpl",
        "n_samples_total", "n_cases", "n_controls",
        "tissue", "has_healthy_ctrl", "notes", "query_source", "summary_excerpt",
    ]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} GSE records → {out_path}")
    print("Next: open the CSV, fill in n_cases / n_controls / tissue / has_healthy_ctrl,")
    print("and select >= 3 independent series with genuine lung-tissue controls.")


if __name__ == "__main__":
    main()
