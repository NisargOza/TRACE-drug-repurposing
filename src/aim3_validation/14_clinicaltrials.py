
import time
from pathlib import Path

import pandas as pd
import requests

REV_DIR  = Path("results/reversal")
AIM3_DIR = Path("results/aim3")
AIM3_DIR.mkdir(parents=True, exist_ok=True)

CT_API = "https://clinicaltrials.gov/api/v2/studies"

POSITIVE_CONTROLS = ["pirfenidone", "nintedanib"]

IPF_TERMS = [
    "idiopathic pulmonary fibrosis",
    "IPF",
    "pulmonary fibrosis",
    "interstitial lung disease",
    "ILD",
]


def search_trials(drug: str) -> list[dict]:
    all_studies = []
    for condition in ["idiopathic pulmonary fibrosis", "pulmonary fibrosis"]:
        params = {
            "query.intr":  drug,
            "query.cond":  condition,
            "fields":      "NCTId,BriefTitle,OverallStatus,Phase,StartDate,CompletionDate,EnrollmentCount",
            "pageSize":    10,
            "format":      "json",
        }
        try:
            r = requests.get(CT_API, params=params, timeout=15)
            r.raise_for_status()
            studies = r.json().get("studies", [])
            for s in studies:
                proto = s.get("protocolSection") or s
                id_mod     = proto.get("identificationModule", {})
                status_mod = proto.get("statusModule", {})
                design_mod = proto.get("designModule", {})
                nct  = id_mod.get("nctId", "")
                if any(s["protocolSection"]["identificationModule"].get("nctId") == nct
                       for s in all_studies):
                    continue
                all_studies.append({
                    "nct_id":          nct,
                    "title":           id_mod.get("briefTitle", "")[:100],
                    "status":          status_mod.get("overallStatus", ""),
                    "phase":           "|".join(design_mod.get("phases", [])),
                    "start_date":      status_mod.get("startDateStruct", {}).get("date", ""),
                    "completion_date": status_mod.get("completionDateStruct", {}).get("date", ""),
                    "enrollment":      design_mod.get("enrollmentInfo", {}).get("count", ""),
                    "condition_query": condition,
                })
            time.sleep(0.3)
        except Exception as e:
            print(f"    [WARN] {drug} / {condition}: {e}")
    return all_studies


def classify_trial(studies: list[dict]) -> dict:
    if not studies:
        return {
            "n_trials": 0,
            "has_completed": False,
            "has_active": False,
            "phases": "",
            "best_status": "None",
            "trial_ids": "",
            "evidence_level": "none",
        }

    statuses   = [s["status"] for s in studies]
    phases     = sorted(set(p for s in studies for p in s["phase"].split("|") if p))
    completed  = any("COMPLETED" in st.upper() for st in statuses)
    active     = any(st.upper() in ("RECRUITING", "ACTIVE_NOT_RECRUITING", "ENROLLING_BY_INVITATION")
                     for st in statuses)
    trial_ids  = "|".join(s["nct_id"] for s in studies[:5])

    if completed and any("3" in p or "4" in p for p in phases):
        level = "strong"
    elif completed:
        level = "moderate"
    elif active:
        level = "active"
    elif studies:
        level = "weak"
    else:
        level = "none"

    best_priority = {"COMPLETED": 0, "ACTIVE_NOT_RECRUITING": 1,
                     "RECRUITING": 2, "TERMINATED": 3}
    best_status = min(statuses, key=lambda s: best_priority.get(s.upper(), 9))

    return {
        "n_trials":      len(studies),
        "has_completed": completed,
        "has_active":    active,
        "phases":        "|".join(phases),
        "best_status":   best_status,
        "trial_ids":     trial_ids,
        "evidence_level": level,
    }


def main() -> None:
    final  = pd.read_csv(REV_DIR / "final_candidates.csv")
    top_drugs = final["drug"].head(20).tolist()
    all_drugs = POSITIVE_CONTROLS + [d for d in top_drugs if d not in POSITIVE_CONTROLS]

    print(f"Searching ClinicalTrials.gov for {len(all_drugs)} drugs...\n")

    rows = []
    for drug in all_drugs:
        print(f"  {drug}...", end=" ", flush=True)
        studies = search_trials(drug)
        record  = classify_trial(studies)
        record["drug"] = drug
        record["is_positive_control"] = drug in POSITIVE_CONTROLS
        rows.append(record)

        if record["n_trials"] > 0:
            print(f"{record['n_trials']} trials  [{record['evidence_level'].upper()}]  "
                  f"phases={record['phases']}  status={record['best_status']}")
        else:
            print("no trials found")

    results = pd.DataFrame(rows)
    results.to_csv(AIM3_DIR / "clinicaltrials_results.csv", index=False)

    lines = [
        "ClinicalTrials.gov evidence for top TRACE candidates",
        "=" * 60,
        f"Query date: 2026-06-02",
        f"Drugs queried: {len(all_drugs)}",
        "",
        "EVIDENCE LEVELS:",
        "  strong   — Phase 3/4 completed trial in IPF",
        "  moderate — Phase 1/2 completed trial in IPF",
        "  active   — currently recruiting in IPF",
        "  weak     — terminated/withdrawn trial in IPF",
        "  none     — no trials found",
        "",
        "=" * 60,
    ]

    for level in ["strong", "moderate", "active", "weak", "none"]:
        subset = results[results["evidence_level"] == level]
        if subset.empty:
            continue
        lines.append(f"\n[{level.upper()}] ({len(subset)} drugs)")
        for _, r in subset.iterrows():
            pc_tag = " *POSITIVE CONTROL*" if r["is_positive_control"] else ""
            lines.append(f"  {r['drug']:30}{pc_tag}")
            if r["n_trials"] > 0:
                lines.append(f"    Trials: {r['n_trials']}  Phases: {r['phases']}  "
                             f"IDs: {r['trial_ids']}")

    report = "\n".join(lines)
    (AIM3_DIR / "trial_evidence_report.txt").write_text(report)
    print(f"\n{report}")
    print(f"\n  → results/aim3/clinicaltrials_results.csv")
    print(f"  → results/aim3/trial_evidence_report.txt")


if __name__ == "__main__":
    main()
