
import csv
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.request import urlopen
from urllib.parse import urlencode
from urllib.error import URLError

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "aim3"
RESULTS.mkdir(parents=True, exist_ok=True)

CANDIDATES = [
    ("cediranib",    "VEGFR/PDGFR inhibitor"),
    ("nintedanib",   "VEGFR/PDGFR/FGFR inhibitor — POSITIVE CONTROL"),
    ("romidepsin",   "HDAC inhibitor"),
    ("osimertinib",  "EGFR inhibitor — ADVERSE SIGNAL IN FAERS"),
    ("dacomitinib",  "pan-EGFR inhibitor"),
    ("JNJ-26481585", "HDAC inhibitor"),
    ("baricitinib",  "JAK1/2 inhibitor"),
    ("vorinostat",   "HDAC inhibitor"),
    ("dasatinib",    "BCR-ABL/Src inhibitor"),
    ("afatinib",     "pan-HER EGFR inhibitor — ADVERSE SIGNAL IN FAERS"),
    ("atorvastatin", "HMG-CoA reductase inhibitor (statin)"),
    ("pitavastatin", "HMG-CoA reductase inhibitor (statin)"),
    ("pirfenidone",  "anti-fibrotic — POSITIVE CONTROL"),
    ("riluzole",     "sodium channel blocker"),
    ("rufinamide",   "sodium channel modulator"),
    ("neratinib",    "pan-HER inhibitor"),
    ("AEE-788",      "EGFR/VEGFR dual inhibitor"),
    ("canertinib",   "pan-HER inhibitor"),
    ("BIBX-1382",    "EGFR inhibitor"),
    ("efatutazone",  "PPARγ agonist"),
]

BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"


def esearch(term: str, retmax: int = 10) -> list[str]:
    params = urlencode({
        "db": "pubmed",
        "term": term,
        "retmax": retmax,
        "retmode": "json",
        "usehistory": "n",
    })
    url = f"{BASE_URL}esearch.fcgi?{params}"
    try:
        with urlopen(url, timeout=15) as r:
            data = json.loads(r.read())
        return data["esearchresult"]["idlist"]
    except (URLError, KeyError, json.JSONDecodeError):
        return []


def efetch_abstracts(pmids: list[str]) -> list[dict]:
    if not pmids:
        return []
    params = urlencode({
        "db": "pubmed",
        "id": ",".join(pmids),
        "rettype": "abstract",
        "retmode": "xml",
    })
    url = f"{BASE_URL}efetch.fcgi?{params}"
    try:
        with urlopen(url, timeout=20) as r:
            xml_data = r.read()
    except URLError:
        return []

    records = []
    try:
        root = ET.fromstring(xml_data)
    except ET.ParseError:
        return []

    for article in root.findall(".//PubmedArticle"):
        pmid_el = article.find(".//PMID")
        pmid = pmid_el.text if pmid_el is not None else ""

        title_el = article.find(".//ArticleTitle")
        title = "".join(title_el.itertext()) if title_el is not None else ""

        abstract_el = article.find(".//AbstractText")
        abstract = "".join(abstract_el.itertext()) if abstract_el is not None else ""

        journal_el = article.find(".//Journal/Title")
        journal = journal_el.text if journal_el is not None else ""

        year_el = article.find(".//PubDate/Year")
        year = year_el.text if year_el is not None else ""

        pub_types = [
            pt.text
            for pt in article.findall(".//PublicationTypeList/PublicationType")
            if pt.text
        ]

        records.append({
            "pmid": pmid,
            "title": title,
            "abstract": abstract[:500],
            "journal": journal,
            "year": year,
            "pub_types": "; ".join(pub_types),
        })
    return records


def classify_study(pub_types: str, title: str, abstract: str) -> tuple[str, str]:
    t = (pub_types + " " + title + " " + abstract).lower()

    if any(k in t for k in ["randomized controlled trial", "clinical trial, phase iii",
                              "clinical trial, phase iv", "meta-analysis", "systematic review"]):
        return "RCT/meta-analysis", "high"
    if any(k in t for k in ["clinical trial", "phase i", "phase ii", "phase 1", "phase 2",
                              "open-label", "pilot study"]):
        return "clinical trial (early phase)", "moderate"
    if any(k in t for k in ["cohort", "case-control", "observational", "retrospective",
                              "prospective", "epidemiolog", "pharmacoepidemi", "registry"]):
        return "observational study", "moderate"
    if any(k in t for k in ["case report", "case series"]):
        return "case report/series", "low"
    if any(k in t for k in ["mouse", "murine", "rat ", "animal model", "in vivo"]):
        return "animal/preclinical in vivo", "low"
    if any(k in t for k in ["in vitro", "cell line", "fibroblast", "epithelial cell"]):
        return "in vitro/preclinical", "low"
    if any(k in t for k in ["review", "editorial", "commentary"]):
        return "review/commentary", "low"
    return "other", "low"


def summarize_finding(title: str, abstract: str) -> str:
    for sent in abstract.replace(". ", ".\n").split("\n"):
        s = sent.strip()
        if len(s) > 40 and any(k in s.lower() for k in [
            "associat", "reduc", "increas", "decreas", "improve", "protect",
            "inhibit", "fibrosis", "survival", "progression", "treatment", "effect",
        ]):
            return s[:200]
    return title[:150] if title else "—"


def main():
    rows = []

    for drug, mechanism in CANDIDATES:
        query = f'("{drug}"[tiab]) AND ("idiopathic pulmonary fibrosis"[tiab] OR "IPF"[tiab] OR "pulmonary fibrosis"[tiab])'
        print(f"  Searching: {drug} ...", end=" ", flush=True)
        pmids = esearch(query, retmax=8)
        time.sleep(0.4)

        articles = efetch_abstracts(pmids)
        time.sleep(0.4)

        if not articles:
            rows.append({
                "drug": drug,
                "mechanism": mechanism,
                "n_hits": 0,
                "pmid": "",
                "year": "",
                "journal": "",
                "study_type": "no hits",
                "evidence_strength": "none",
                "pub_types": "",
                "title": "",
                "key_finding": "",
            })
            print("0 hits")
            continue

        print(f"{len(articles)} hits")
        for art in articles:
            stype, strength = classify_study(art["pub_types"], art["title"], art["abstract"])
            finding = summarize_finding(art["title"], art["abstract"])
            rows.append({
                "drug": drug,
                "mechanism": mechanism,
                "n_hits": len(articles),
                "pmid": art["pmid"],
                "year": art["year"],
                "journal": art["journal"],
                "study_type": stype,
                "evidence_strength": strength,
                "pub_types": art["pub_types"],
                "title": art["title"],
                "key_finding": finding,
            })

    fields = ["drug", "mechanism", "n_hits", "pmid", "year", "journal",
              "study_type", "evidence_strength", "pub_types", "title", "key_finding"]
    out_csv = RESULTS / "literature_results.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {len(rows)} records → {out_csv}")

    by_drug: dict[str, list] = {}
    for r in rows:
        by_drug.setdefault(r["drug"], []).append(r)

    strength_order = {"high": 0, "moderate": 1, "low": 2, "none": 3}

    lines = [
        "PubMed Literature Corroboration — Top TRACE Candidates",
        "=" * 60,
        "",
    ]
    for drug, mechanism in CANDIDATES:
        hits = by_drug.get(drug, [])
        lines.append(f"{drug.upper()}  [{mechanism}]")
        lines.append("-" * 50)
        if not hits or hits[0]["n_hits"] == 0:
            lines.append("  No PubMed hits found.")
            lines.append("")
            continue

        n = hits[0]["n_hits"]
        lines.append(f"  PubMed hits returned: {n}")
        sorted_hits = sorted(hits, key=lambda x: (strength_order.get(x["evidence_strength"], 9),
                                                   -(int(x["year"]) if x["year"].isdigit() else 0)))
        for h in sorted_hits:
            lines.append(
                f"  [{h['evidence_strength'].upper():8s}] {h['year']} | {h['study_type']}"
            )
            lines.append(f"    PMID {h['pmid']}: {h['title'][:90]}")
            lines.append(f"    Finding: {h['key_finding'][:160]}")
        lines.append("")

    report_path = RESULTS / "literature_report.txt"
    report_path.write_text("\n".join(lines))
    print(f"Report  → {report_path}")


if __name__ == "__main__":
    main()
