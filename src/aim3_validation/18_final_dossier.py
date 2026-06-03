"""
Aim 3 — Final candidate evidence dossier.

Reads all Aim 2/3 outputs and assembles a per-drug triangulated evidence
summary for the top 10 novel TRACE candidates.  Writes:
  results/aim3/final_dossier.csv
  results/aim3/final_dossier_report.txt
"""

import csv
import json
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]
REV  = ROOT / "results" / "reversal"
AIM3 = ROOT / "results" / "aim3"

# ── Load combined scores ───────────────────────────────────────────────────────
def load_combined() -> dict:
    data = {}
    with open(REV / "final_candidates_full.csv") as f:
        for row in csv.DictReader(f):
            data[row["drug"].lower()] = row
    return data

# ── Load clinical trials ───────────────────────────────────────────────────────
def load_trials() -> dict:
    data = {}
    with open(AIM3 / "clinicaltrials_results.csv") as f:
        for row in csv.DictReader(f):
            data[row["drug"].lower()] = row
    return data

# ── Load FAERS ─────────────────────────────────────────────────────────────────
def load_faers() -> dict:
    data = {}
    with open(AIM3 / "faers_results.csv") as f:
        for row in csv.DictReader(f):
            data[row["drug"].lower()] = row
    return data

# ── Load literature ────────────────────────────────────────────────────────────
def load_literature() -> dict[str, list]:
    data: dict[str, list] = defaultdict(list)
    path = AIM3 / "literature_results.csv"
    if not path.exists():
        return data
    with open(path) as f:
        for row in csv.DictReader(f):
            if row["n_hits"] != "0":
                data[row["drug"].lower()].append(row)
    return data

# ── Top candidates to feature (excluding the two positive controls) ────────────
TOP_NOVEL = [
    ("cediranib",    "VEGFR/PDGFR kinase inhibitor"),
    ("romidepsin",   "HDAC inhibitor"),
    ("dasatinib",    "BCR-ABL/Src kinase inhibitor (senolytic)"),
    ("dacomitinib",  "pan-EGFR inhibitor"),
    ("JNJ-26481585", "HDAC inhibitor"),
    ("baricitinib",  "JAK1/2 inhibitor"),
    ("vorinostat",   "HDAC inhibitor"),
    ("atorvastatin", "HMG-CoA reductase inhibitor (statin)"),
    ("pitavastatin", "HMG-CoA reductase inhibitor (statin)"),
    ("osimertinib",  "EGFR inhibitor — ⚠ ADVERSE SIGNAL"),
    ("afatinib",     "pan-HER inhibitor — ⚠ ADVERSE SIGNAL"),
]
POSITIVE_CONTROLS = ["nintedanib", "pirfenidone"]


def best_lit_strength(hits: list) -> str:
    order = {"high": 0, "moderate": 1, "low": 2, "none": 3}
    if not hits:
        return "none"
    return min(hits, key=lambda h: order.get(h["evidence_strength"], 9))["evidence_strength"]


def best_lit_study(hits: list) -> str:
    """Return description of the strongest study."""
    order = {"high": 0, "moderate": 1, "low": 2, "none": 3}
    if not hits:
        return "—"
    best = min(hits, key=lambda h: order.get(h["evidence_strength"], 9))
    return f"PMID {best['pmid']} ({best['year']}): {best['title'][:80]}"


def faers_summary(frow: dict | None) -> str:
    if frow is None:
        return "no data"
    n = frow.get("n_drug_ipf", "0")
    ror = frow.get("ror", "")
    lo  = frow.get("ror_ci_lo", "")
    hi  = frow.get("ror_ci_hi", "")
    if not ror:
        return f"n={n}, ROR=N/A (insufficient reports)"
    try:
        r = float(ror)
        direction = "PROTECTIVE" if r < 1 else "risk signal"
    except ValueError:
        direction = "?"
    return f"n={n}, ROR={float(ror):.2f} [{float(lo):.2f}–{float(hi):.2f}] ({direction})"


def trials_summary(trow: dict | None) -> str:
    if trow is None:
        return "no data"
    lvl = trow.get("evidence_level", "none")
    ids = trow.get("trial_ids", "")
    phases = trow.get("phases", "")
    if lvl == "none":
        return "no trials found"
    return f"{lvl.upper()} — {phases} | {ids}"


def overall_verdict(drug: str, scores: dict | None, gen: float,
                    trial_lvl: str, faers_ror: float | None,
                    lit_strength: str) -> str:
    """One-sentence triangulated verdict."""
    flags = []

    # Reversal signal
    if scores:
        nt = float(scores.get("net_trace", 0))
        rank = int(scores.get("combined_rank", 9999))
        if rank <= 10:
            flags.append("strong reversal (top 10 combined)")
        elif rank <= 25:
            flags.append("moderate reversal (top 25 combined)")

    # Genetic support
    if gen >= 0.5:
        flags.append("high genetic support (Open Targets)")
    elif gen >= 0.05:
        flags.append("modest genetic support")

    # Trials
    if trial_lvl in ("strong", "moderate"):
        flags.append(f"clinical trial evidence ({trial_lvl})")

    # FAERS adverse
    if faers_ror is not None and faers_ror >= 2.0:
        flags.append(f"⚠ FAERS adverse signal (ROR={faers_ror:.1f})")

    # Literature
    if lit_strength == "high":
        flags.append("high-quality literature support")
    elif lit_strength == "moderate":
        flags.append("moderate literature support")
    elif lit_strength == "low" and any(k in drug for k in ["statin", "vorinostat", "romidepsin",
                                                            "baricitinib", "dasatinib",
                                                            "atorvastatin", "pitavastatin"]):
        flags.append("preclinical literature support")

    return "; ".join(flags) if flags else "limited evidence"


def main():
    combined  = load_combined()
    trials    = load_trials()
    faers     = load_faers()
    literature = load_literature()

    rows = []
    report_lines = [
        "TRACE — Final Candidate Evidence Dossier",
        "=" * 70,
        "Triangulated evidence: reversal score + genetic + trials + FAERS + literature",
        "",
    ]

    all_drugs = POSITIVE_CONTROLS + [d for d, _ in TOP_NOVEL]

    for drug in all_drugs:
        drug_key = drug.lower()
        mech = next((m for d, m in TOP_NOVEL if d.lower() == drug_key),
                    "POSITIVE CONTROL" if drug in POSITIVE_CONTROLS else "—")

        sc   = combined.get(drug_key)
        tr   = trials.get(drug_key)
        fa   = faers.get(drug_key)
        lit  = literature.get(drug_key, [])

        gen  = float(sc["genetic_support"]) if sc else 0.0
        crank = int(sc["combined_rank"]) if sc else 9999
        nt   = float(sc["net_trace"]) if sc else 0.0
        vae  = float(sc["vae_score"]) if sc else 0.0

        trial_lvl = tr.get("evidence_level", "none") if tr else "none"
        try:
            fror = float(fa["ror"]) if fa and fa.get("ror") else None
        except ValueError:
            fror = None

        lit_strength = best_lit_strength(lit)
        n_lit = len(lit)

        verdict = overall_verdict(drug_key, sc, gen, trial_lvl, fror, lit_strength)

        rows.append({
            "drug": drug,
            "mechanism": mech,
            "combined_rank": crank,
            "net_trace": f"{nt:.4f}",
            "vae_trace": f"{vae:.4f}",
            "genetic_support": f"{gen:.3f}",
            "trial_evidence": trial_lvl,
            "trial_ids": tr.get("trial_ids", "") if tr else "",
            "faers_ror": f"{fror:.2f}" if fror else "N/A",
            "faers_n_ipf": fa.get("n_drug_ipf", "0") if fa else "0",
            "lit_n_hits": n_lit,
            "lit_best_strength": lit_strength,
            "lit_best_study": best_lit_study(lit),
            "verdict": verdict,
        })

        # ── Report block ───────────────────────────────────────────────────────
        tag = " [POSITIVE CONTROL]" if drug in POSITIVE_CONTROLS else ""
        report_lines.append(f"{drug.upper()}{tag}")
        report_lines.append(f"  Mechanism   : {mech}")
        report_lines.append(f"  Combined rank: {crank}/1768  |  Net-TRACE={nt:.4f}  |  VAE={vae:.4f}")
        report_lines.append(f"  Genetic (OT): {gen:.3f}")
        report_lines.append(f"  Trials      : {trials_summary(tr)}")
        report_lines.append(f"  FAERS       : {faers_summary(fa)}")
        report_lines.append(f"  Literature  : {n_lit} PubMed hits  |  best strength = {lit_strength}")
        if lit:
            report_lines.append(f"    Best study: {best_lit_study(lit)}")
        report_lines.append(f"  VERDICT     : {verdict}")
        report_lines.append("")

    # ── Save CSV ───────────────────────────────────────────────────────────────
    fields = ["drug", "mechanism", "combined_rank", "net_trace", "vae_trace",
              "genetic_support", "trial_evidence", "trial_ids",
              "faers_ror", "faers_n_ipf", "lit_n_hits", "lit_best_strength",
              "lit_best_study", "verdict"]
    out_csv = AIM3 / "final_dossier.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # ── Save report ────────────────────────────────────────────────────────────
    out_txt = AIM3 / "final_dossier_report.txt"
    out_txt.write_text("\n".join(report_lines))

    print(f"Dossier CSV    → {out_csv}")
    print(f"Dossier report → {out_txt}")
    print()
    print("\n".join(report_lines))


if __name__ == "__main__":
    main()
