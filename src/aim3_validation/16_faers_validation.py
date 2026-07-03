
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

REV_DIR  = Path("results/reversal")
AIM3_DIR = Path("results/aim3")
AIM3_DIR.mkdir(parents=True, exist_ok=True)

OPENFDA_API = "https://api.fda.gov/drug/event.json"
POSITIVE_CONTROLS = ["pirfenidone", "nintedanib"]

IPF_PT_TERMS = [
    "idiopathic pulmonary fibrosis",
    "pulmonary fibrosis",
    "interstitial lung disease",
]

def _load_env():
    from pathlib import Path as P
    env = P(".env")
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, _, v = line.partition("=")
                import os; os.environ.setdefault(k.strip(), v.strip().strip("'\""))
_load_env()


def count_reports(search_query: str) -> int:
    try:
        r = requests.get(
            OPENFDA_API,
            params={"search": search_query, "limit": 1},
            timeout=15,
        )
        data = r.json()
        if "error" in data:
            return 0
        return data.get("meta", {}).get("results", {}).get("total", 0)
    except Exception:
        return 0


def get_ipf_reports(drug: str) -> int:
    total = 0
    for term in IPF_PT_TERMS:
        params = {
            "search": (f'patient.drug.medicinalproduct:"{drug}" '
                       f'AND patient.reaction.reactionmeddrapt:"{term}"'),
            "limit": 1,
        }
        try:
            r = requests.get(OPENFDA_API, params=params, timeout=15)
            data = r.json()
            if "error" not in data:
                total += data.get("meta", {}).get("results", {}).get("total", 0)
            time.sleep(0.2)
        except Exception:
            pass
    return total


def compute_ror(a: int, b: int, c: int, d: int) -> tuple[float, float, float]:
    if a == 0 or b == 0 or c == 0 or d == 0:
        return np.nan, np.nan, np.nan
    ror = (a * d) / (b * c)
    log_ror = np.log(ror)
    se = np.sqrt(1/a + 1/b + 1/c + 1/d)
    ci_lo = np.exp(log_ror - 1.96 * se)
    ci_hi = np.exp(log_ror + 1.96 * se)
    return ror, ci_lo, ci_hi


def main() -> None:
    final = pd.read_csv(REV_DIR / "final_candidates.csv")
    top_drugs = final["drug"].head(15).tolist()
    all_drugs = POSITIVE_CONTROLS + [d for d in top_drugs if d not in POSITIVE_CONTROLS]

    print("Querying openFDA FAERS for disproportionality analysis...")
    print(f"Drugs: {len(all_drugs)}\n")

    print("Fetching total IPF reports in FAERS...")
    total_ipf = sum(
        count_reports(f'patient.reaction.reactionmeddrapt:"{term}"')
        for term in IPF_PT_TERMS
    )
    print(f"  Total FAERS IPF reports: {total_ipf:,}")

    total_all = count_reports("_exists_:safetyreportid")
    print(f"  Total FAERS reports:     {total_all:,}\n")

    rows = []
    for drug in all_drugs:
        print(f"  {drug}...", end=" ", flush=True)

        a = get_ipf_reports(drug)
        drug_total = count_reports(f'patient.drug.medicinalproduct:"{drug}"')
        b = max(drug_total - a, 0)
        c = max(total_ipf - a, 0)
        d = max(total_all - drug_total - c, 0)

        ror, ci_lo, ci_hi = compute_ror(a, b, c, d)

        rows.append({
            "drug":          drug,
            "n_drug_ipf":    a,
            "n_drug_total":  drug_total,
            "ror":           ror,
            "ror_ci_lo":     ci_lo,
            "ror_ci_hi":     ci_hi,
            "is_protective": (ci_hi < 1.0) if not np.isnan(ror) else False,
            "is_pc":         drug in POSITIVE_CONTROLS,
        })

        status = ""
        if not np.isnan(ror):
            direction = "PROTECTIVE" if ror < 1 else "RISK"
            sig = "*" if (ci_hi < 1 or ci_lo > 1) else ""
            status = f"n={a}  ROR={ror:.2f} [{ci_lo:.2f}-{ci_hi:.2f}] {direction}{sig}"
        else:
            status = f"n={a}  insufficient data for ROR"
        print(status)
        time.sleep(0.3)

    results = pd.DataFrame(rows)
    results.to_csv(AIM3_DIR / "faers_results.csv", index=False)

    lines = [
        "FAERS Disproportionality Analysis (Reporting Odds Ratio)",
        "=" * 60,
        f"Total FAERS reports: {total_all:,}",
        f"Total IPF reports in FAERS: {total_ipf:,}",
        "",
        "CAVEAT: FAERS is an adverse-event database. Protective signals",
        "(ROR < 1) must be interpreted cautiously due to under-reporting,",
        "channeling bias, and confounding by indication.",
        "",
        "ROR < 1: fewer IPF reports than expected (potential protective signal)",
        "ROR > 1: more IPF reports than expected (adverse/risk signal)",
        "* = CI excludes 1.0 (nominally significant)",
        "",
        "=" * 60,
    ]

    for _, r in results.sort_values("ror").iterrows():
        pc_tag = " [POSITIVE CONTROL]" if r["is_pc"] else ""
        if np.isnan(r["ror"]):
            lines.append(f"  {r['drug']:25} n={int(r['n_drug_ipf'])}  ROR=N/A{pc_tag}")
        else:
            sig = "*" if (r["ror_ci_hi"] < 1 or r["ror_ci_lo"] > 1) else ""
            direction = "PROTECTIVE" if r["ror"] < 1 else "risk"
            lines.append(
                f"  {r['drug']:25} n={int(r['n_drug_ipf']):4}  "
                f"ROR={r['ror']:.2f} [{r['ror_ci_lo']:.2f}-{r['ror_ci_hi']:.2f}] "
                f"{direction}{sig}{pc_tag}"
            )

    protective = results[results["is_protective"]]
    if not protective.empty:
        lines += ["", f"Drugs with nominally protective signal (ROR CI entirely < 1.0):"]
        for _, r in protective.iterrows():
            lines.append(f"  {r['drug']}  ROR={r['ror']:.2f} [{r['ror_ci_lo']:.2f}-{r['ror_ci_hi']:.2f}]")

    report = "\n".join(lines)
    (AIM3_DIR / "faers_report.txt").write_text(report)
    print(f"\n{report}")
    print(f"\n  → results/aim3/faers_results.csv")
    print(f"  → results/aim3/faers_report.txt")


if __name__ == "__main__":
    main()
