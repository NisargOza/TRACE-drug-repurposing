# Pre-registered Analysis Plan — All of Us Pharmacoepidemiology
## Statin initiation and incident IPF in the All of Us Researcher Workbench

**Status:** Pre-registered. This plan was committed BEFORE any All of Us data were accessed.
Any deviation from this plan must be documented as post-hoc and clearly labeled.

**Corresponding TRACE prediction:** atorvastatin is FDR-significant (emp. p = 0.0006,
BH-FDR = 0.012) by Net-TRACE reversal of the IPF consensus signature. This analysis
provides the patient-level test of that prediction.

---

## 1. Study question

Primary: Among adults without prevalent IPF, is new initiation of a statin
(atorvastatin primary; any-statin secondary) associated with lower incidence of IPF?

Secondary: Among patients with prevalent IPF, is statin use at diagnosis associated
with lower all-cause mortality / slower progression (FVC decline proxy)?

---

## 2. Design

New-user (incident-user), active-comparator cohort design.

**Rationale for design choices:**
- New-user: no statin in the 365-day baseline. Avoids prevalent-user bias and
  immortal-time bias (the two most common pharmacoepi traps in drug-disease studies).
- Active comparator (primary): new initiators of a cardiovascular-risk medication
  prescribed to a similar population (antihypertensive initiator; or non-statin
  lipid-lowering agent if counts allow). Approximates "patients under similar
  cardiometabolic management who did not start a statin" — far stronger control for
  confounding by indication and healthy-user bias than untreated comparators.
- Non-initiator comparator (sensitivity): propensity-matched non-statin initiators
  with randomly assigned index date drawn from the exposure group's distribution
  (to prevent immortal time in the control arm).

---

## 3. Cohort definition

**Source population:**
- All of Us Registered Tier EHR participants
- Age ≥ 40 at index date
- ≥ 365 days of EHR history before index date
- ≥ 1 prior clinical visit documented

**Exposure group:**
- First-ever statin prescription (RxNorm statin class concept set)
- No statin in prior 365 days (new-user criterion)
- Index date = date of first statin fill

**Primary comparator (active):**
- New initiators of antihypertensive medication (ACE inhibitor, ARB, calcium-channel
  blocker, or thiazide diuretic) with no statin in prior 365 days
- Same eligibility criteria as exposure group
- Index date = date of first antihypertensive fill

**Secondary comparator (non-initiator sensitivity):**
- Participants not initiating a statin during the study period
- Propensity-score matched 1:1 to the exposure group
- Index date randomly sampled from the exposure group's index-date distribution

**Exclusion criteria (applied before index):**
- Prevalent IPF or any pulmonary fibrosis (ICD-10 J84.1x or SNOMED equivalent)
  at any point before or on the index date
- < 365 days of baseline EHR history
- Age < 40 at index
- No documented clinical visits in the baseline period

---

## 4. Outcome definition

**Primary outcome:** Incident IPF using a validated phenotype:
- ICD-10 J84.112 on ≥ 2 separate dates ≥ 30 days apart, OR
- ICD-10 J84.112 on ≥ 1 date PLUS at least one of:
  - Antifibrotic prescription (pirfenidone or nintedanib)
  - Pulmonology outpatient visit within ±180 days of diagnosis code
  - HRCT/chest CT code within ±180 days of diagnosis code

**Sensitivity outcome definitions:**
- Strict: ≥ 2 IPF codes AND (antifibrotic OR pulmonology visit)
- Loose: ≥ 1 IPF code

**Induction/lag period:** Outcomes are counted only after a 12-month lag from index
date. IPF diagnosed within the 12-month lag window is excluded (addresses reverse
causation and protopathic bias — early subclinical IPF symptoms could influence
prescribing decisions). Sensitivity: 6-month lag.

---

## 5. Time and censoring

- **Time zero:** index date (aligns eligibility, exposure start, and follow-up start)
- **Follow-up:** from index date + lag period until the earliest of:
  - Incident IPF (outcome event)
  - Death
  - Last EHR contact (disenrollment / data cutoff proxy)
  - Study end date
- **Exposure handling (primary):** intention-to-treat — first statin fill carried
  forward regardless of subsequent adherence. Avoids informative censoring.
- **Sensitivity:** as-treated with a 90-day permissible gap (grace period). If
  counts allow, add a cumulative-exposure dose-response analysis.

---

## 6. Covariates (measured in 365-day baseline window)

| Covariate | Source |
|---|---|
| Age at index | Demographic |
| Sex | Demographic |
| Race/ethnicity | Demographic |
| Smoking status (current/former/never) | EHR condition/observation |
| BMI | Measurement |
| GERD / gastroesophageal reflux | Condition |
| Diabetes mellitus | Condition |
| Hypertension | Condition |
| Established cardiovascular disease (MI, HF, stroke) | Condition |
| COPD / obstructive lung disease | Condition |
| Baseline LDL/lipids (if available) | Measurement |
| Healthcare-utilization intensity (visit count) | Visit |
| Calendar year of cohort entry | Derived |

---

## 7. Analysis

### 7.1 Propensity score estimation
- Logistic regression: statin initiation ~ all §6 covariates
- Robustness check: gradient-boosted trees (optional)
- Assess positivity/overlap by plotting PS distributions for exposed and comparator

### 7.2 Matching
- 1:1 nearest-neighbor PS matching with caliper = 0.2 × SD(logit PS)
- Confirm balance: standardized mean differences (SMD) < 0.1 for all covariates
  post-match; report Love plot

### 7.3 Primary model
- Cox proportional hazards for time-to-incident-IPF in the matched cohort
- Report hazard ratio with 95% CI (95% CI via robust/sandwich variance)
- Test proportional-hazards assumption (Schoenfeld residuals)
- If PH violated: report time-stratified HRs

### 7.4 Secondary estimator
- Inverse-probability-of-treatment weighting (IPTW) as sensitivity; should agree
  with matching estimate

---

## 8. Bias control and falsification

### 8.1 Negative-control outcomes (pre-specified)
- Appendicitis (ICD-10 K37) — acute, unrelated to statins or IPF etiology
- Traumatic fracture (ICD-10 S-range) — acute, confounder-independent
- If statins appear "protective" against these outcomes, this flags residual
  confounding and must be reported as such

### 8.2 Negative-control exposure
- Proton pump inhibitor (PPI) initiation as a negative-control drug — same
  cardiometabolic-care setting, no plausible IPF-protective mechanism
- Run through identical pipeline; expect null HR for IPF

### 8.3 E-value
- Compute E-value (VanderWeele & Ding 2017) for the point estimate and lower CI
  bound — quantifies minimum unmeasured-confounder strength to explain the result

### 8.4 Sensitivity battery (pre-specified, all labeled as sensitivity)
1. Strict IPF outcome definition (≥ 2 codes + antifibrotic/pulmonology)
2. Loose IPF outcome definition (≥ 1 code)
3. Alternate lag: 6 months (vs primary 12 months)
4. As-treated exposure (90-day grace period)
5. IPTW estimator (vs primary matching)
6. Non-initiator comparator (vs primary active comparator)
7. Any-statin (vs primary atorvastatin-only)
8. With/without the dose-response cumulative-exposure analysis (if counts allow)

---

## 9. Power and pre-specified fallback

**Before modeling, estimate expected event counts:**
- N matched pairs × background IPF incidence (~3–5/10,000 person-years in age ≥ 40)
  × mean follow-up years

**Decision rule (pre-specified, documented here before seeing data):**
- If expected events ≥ 20 per arm: proceed with incidence primary design
- If expected events < 20 per arm: pivot to secondary design — among prevalent IPF
  patients, statin use at diagnosis → time-to-death (Cox PH). This mirrors Kreuter
  2017 and may be adequately powered even in a small IPF subgroup.
- If neither design yields ≥ 10 events: report as a pre-specified, underpowered
  exploratory analysis with descriptive statistics only. This is a legitimate result.

**Document the event count and design choice before running any outcome model.**

---

## 10. Reporting checklist

Follow STROBE and RECORD-PE reporting standards. Produce:

- [ ] Cohort-attrition (CONSORT-style) diagram with exclusion counts at each step
- [ ] Love plot confirming post-match SMD < 0.1 for all covariates
- [ ] Cox HR table: primary estimate + all 8 pre-specified sensitivity analyses
- [ ] Negative-control outcome results (appendicitis, fracture)
- [ ] Negative-control exposure result (PPI)
- [ ] E-value for primary estimate
- [ ] Proportional-hazards test output (Schoenfeld residuals)
- [ ] Event count and design-choice documentation (per §9)

**Framing:** present result — protective, null, or underpowered — as convergent or
divergent with (a) the TRACE prediction (atorvastatin FDR-significant, emp. p=0.0006)
and (b) the published observational literature (Kreuter 2017, Korean NHIS 2024,
meta-analysis non-significant in random-effects model). Do not selectively present
confirmatory evidence.

---

## Data access note

All analysis runs inside the All of Us Researcher Workbench (Registered Tier).
No row-level participant data are exported. Only aggregate results (HRs, CIs,
counts ≥ 20 per cell per AoU policy) are reported outside the platform.
