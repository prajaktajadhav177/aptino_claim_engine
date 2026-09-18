"""
Policy reasoning logic — no LLM call, just Python functions that check
each policy rule against the case facts.

Every check does three things: decides if the rule even applies to this
case, pulls the retrieved evidence chunk that backs it up, and returns a
Finding (text + citation + whether it blocks the claim, needs review, or
is just informational).

Numbers and exclusions here are the actual ones from the supplied policy,
not something an LLM guessed at. If an LLM gets plugged in later (see
src/llm.py) it should only be used to phrase things, not decide them —
the decisions stay here, and ValidationAgent checks anything an LLM adds
still has a real citation behind it.
"""

from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional

from src.retrieval import HybridRetriever, RetrievedChunk

SPECIFIC_DISEASE_1YR_LIST = [
    "cataract",
    "benign prostatic hypertrophy",
    "myomectomy",
    "hysterectomy",
    "hernia",
    "hydrocele",
    "fistula",
    "piles",
    "arthritis",
    "gout",
    "rheumatism",
    "joint replacement",
    "sinusitis",
    "stone",
    "dilatation and curettage",
    "polyp",
    "adenoid",
    "hemorrhoid",
    "dialysis",
    "tonsil",
    "gastric",
    "duodenal ulcer",
]

ALTERNATIVE_TREATMENT_KEYWORDS = [
    "ayurved",
    "ayurveda",
    "panchakarma",
    "homeopath",
    "unani",
    "siddha",
    "naturopathy",
]

PREGNANCY_KEYWORDS = [
    "pregnan",
    "childbirth",
    "miscarriage",
    "abortion",
    "caesarean",
    "cesarean",
    "infertility",
    "sub fertility",
    "sub-fertility",
    "assisted conception",
]

# Item 20 of "WHAT WE EXCLUDE" — these specific diseases aren't payable.
# Keyword match only, so it can't tell "treatment FOR diabetes" apart from
# "treatment of a complication from diabetes" (e.g. a diabetic foot ulcer
# needing surgery) — that needs someone to actually read the discharge
# summary. Flagging NEEDS_REVIEW here instead of guessing.
EXCLUDED_ILLNESS_KEYWORDS = [
    "asthma",
    "bronchitis",
    "chronic nephritis",
    "nephritic syndrome",
    "diarrhoea",
    "diarrhea",
    "dysentery",
    "gastro-enteritis",
    "gastroenteritis",
    "diabetes",
    "epilepsy",
    "hypertension",
    "influenza",
    "psychiatric",
    "psychosomatic",
    "pyrexia",
    "tonsillitis",
    "upper respiratory tract",
    "laryngitis",
    "pharingitis",
    "pharyngitis",
]

DAY_CARE_RECOGNISED = [
    "dialysis",
    "chemotherapy",
    "radiotherapy",
    "cataract",
    "eye surgery",
    "lithotripsy",
    "tonsillectomy",
    "d&c",
]


@dataclass
class Finding:
    text: str
    evidence: Optional[RetrievedChunk]
    kind: str  # "finding" | "limit" | "missing_evidence"
    severity: str = "info"  # "block" (=> NOT_ADMISSIBLE), "review" (=> NEEDS_REVIEW), "info"


def _best(results: List[RetrievedChunk]) -> Optional[RetrievedChunk]:
    return results[0] if results else None


def _days_between(start: str, end: str) -> int:
    try:
        y1, m1, d1 = [int(x) for x in start.split("-")]
        y2, m2, d2 = [int(x) for x in end.split("-")]
        return (date(y2, m2, d2) - date(y1, m1, d1)).days
    except Exception:
        return 0


def evaluate(case, retriever: HybridRetriever) -> Dict:
    """Runs every relevant policy dimension for the case and returns a dict
    with findings/limits/missing_evidence/decision/confidence/dimensions."""

    findings: List[Finding] = []
    dimensions_checked: List[str] = []

    evidence_pool: List[Dict] = []

    def do_search(query: str, top_k: int = 2):
        results = retriever.search(query, top_k=top_k)
        for r in results:
            evidence_pool.append({
                "chunk_id": r.chunk.chunk_id,
                "section": r.chunk.section,
                "page": r.chunk.page,
                "rerank_score": round(r.rerank_score, 3),
            })
        return results

    treatment = case.treatment
    diag_procedure = f"{treatment.diagnosis or ''} {treatment.procedure or ''}".lower()
    days_since_inception = _days_between(case.policy_start_date, case.claim_date)
    continuous_months = case.continuous_coverage_months or 0
    prior_years = case.prior_insurer_continuous_years or 0

    # ---- 1. Hospital definition -----------------------------------------
    dimensions_checked.append("hospital_definition")
    hosp_results = do_search("Hospital definition registered qualified nursing staff beds", top_k=2)
    hosp_evidence_gap = False
    ec = case.evidence_context or {}
    if ec.get("hospital_registered") is None and ec.get("hospital_minimum_criteria_documented") is False:
        hosp_evidence_gap = True
    elif ec.get("hospital_registered") is False and not ec.get("hospital_minimum_criteria_documented"):
        hosp_evidence_gap = True

    if hosp_evidence_gap:
        findings.append(
            Finding(
                text=(
                    "The supplied evidence does not establish that the treating facility is "
                    "registered under the Clinical Establishments Act or meets the policy's "
                    "minimum-criteria definition of a 'Hospital' (qualified nursing staff, bed count, "
                    "resident medical practitioner, own operation theatre, patient records)."
                ),
                evidence=_best(hosp_results),
                kind="finding",
                severity="review",
            )
        )
        findings.append(
            Finding(
                text="Registration status / minimum-criteria documentation for the treating facility.",
                evidence=None,
                kind="missing_evidence",
            )
        )
    else:
        findings.append(
            Finding(
                text=(
                    "The hospital's network/cashless status affects how the claim is paid "
                    "(cashless vs. reimbursement) but is not itself a condition for admissibility "
                    "under the policy's Hospital definition."
                ),
                evidence=_best(hosp_results),
                kind="finding",
            )
        )

    # ---- 2. Generic evidence_context gaps (medical necessity etc.) ------
    if case.evidence_context and "medical_necessity_confirmed" in case.evidence_context and ec.get("medical_necessity_confirmed") is None:
        dimensions_checked.append("medical_necessity")
        mn_results = do_search("Medically Necessary treatment prescribed medical practitioner", top_k=2)
        findings.append(
            Finding(
                text=(
                    "Medical necessity of the treatment/stay has not been confirmed in the "
                    "supplied evidence, which the policy requires ('Medically Necessary')."
                ),
                evidence=_best(mn_results),
                kind="finding",
                severity="review",
            )
        )
        findings.append(
            Finding(
                text="Confirmation that the treatment/hospitalization was medically necessary.",
                evidence=None,
                kind="missing_evidence",
            )
        )

    # ---- 3. 30-day initial waiting period --------------------------------
    dimensions_checked.append("initial_waiting_period")
    wp_results = do_search("30 days waiting period applies to all claims", top_k=2)
    waiting_period_cleared = (
        continuous_months >= 1 or days_since_inception >= 30 or prior_years >= 1
    )
    if not waiting_period_cleared:
        findings.append(
            Finding(
                text=(
                    f"The claim falls within the policy's 30-day initial waiting period "
                    f"(days since inception: {days_since_inception}, continuous coverage months: "
                    f"{continuous_months}), and no prior-insurer continuity is on record to waive it."
                ),
                evidence=_best(wp_results),
                kind="finding",
                severity="block",
            )
        )
    else:
        findings.append(
            Finding(
                text=(
                    f"The 30-day initial waiting period does not bar this claim (days since "
                    f"inception: {days_since_inception}, continuous coverage months: "
                    f"{continuous_months}, prior-insurer continuity years: {prior_years})."
                ),
                evidence=_best(wp_results),
                kind="finding",
            )
        )

    # ---- 4. Pre-existing disease 48-month waiting period -----------------
    if treatment.pre_existing:
        dimensions_checked.append("pre_existing_disease")
        ped_results = do_search("Pre-existing diseases 48 months waiting period", top_k=2)
        effective_months = continuous_months + prior_years * 12
        if effective_months >= 48:
            findings.append(
                Finding(
                    text=(
                        f"The condition is pre-existing, but {effective_months:.0f} effective months "
                        "of continuous coverage (including any recognised prior-insurer credit) have "
                        "elapsed, clearing the 48-month Pre-Existing Disease waiting period."
                    ),
                    evidence=_best(ped_results),
                    kind="finding",
                )
            )
        else:
            findings.append(
                Finding(
                    text=(
                        f"The condition is pre-existing and only {effective_months:.0f} effective "
                        "months of continuous coverage have elapsed, which is short of the policy's "
                        "48-month Pre-Existing Disease waiting period."
                    ),
                    evidence=_best(ped_results),
                    kind="finding",
                    severity="block",
                )
            )

    # ---- 5. Specific-disease 1-year waiting period -----------------------
    matched_disease = next((d for d in SPECIFIC_DISEASE_1YR_LIST if d in diag_procedure), None)
    if matched_disease:
        dimensions_checked.append("specific_disease_first_year")
        sd_results = do_search(
            "first year hospitalization cataract hernia waiting period specific diseases", top_k=2
        )
        waived = continuous_months >= 12 or prior_years >= 1
        if waived:
            findings.append(
                Finding(
                    text=(
                        f"'{matched_disease.title()}' is on the policy's first-year specific-disease "
                        "waiting-period list, but continuous prior coverage of at least one year "
                        "waives that waiting period for this claim."
                    ),
                    evidence=_best(sd_results),
                    kind="finding",
                )
            )
        else:
            findings.append(
                Finding(
                    text=(
                        f"'{matched_disease.title()}' is on the policy's first-year specific-disease "
                        "waiting-period list, and continuous coverage is under 1 year, so this "
                        "hospitalization expense is excluded in the first policy year."
                    ),
                    evidence=_best(sd_results),
                    kind="finding",
                    severity="block",
                )
            )

    # ---- 6. Cosmetic / plastic surgery exclusion -------------------------
    if "cosmetic" in diag_procedure or "aesthetic" in diag_procedure or "plastic" in diag_procedure:
        dimensions_checked.append("cosmetic_exclusion")
        cos_results = do_search("cosmetic aesthetic treatment plastic surgery excluded injury disease", top_k=2)
        findings.append(
            Finding(
                text=(
                    "The treatment is described as cosmetic, which the policy excludes unless it "
                    "relates to treatment of an Injury or Disease; no such link is stated in the "
                    "case facts."
                ),
                evidence=_best(cos_results),
                kind="finding",
                severity="block",
            )
        )

    # ---- 6b. Alternative / non-allopathic treatment exclusion ------------
    if any(k in diag_procedure for k in ALTERNATIVE_TREATMENT_KEYWORDS) or any(
        k in (case.hospital.name or "").lower() for k in ALTERNATIVE_TREATMENT_KEYWORDS
    ):
        dimensions_checked.append("alternative_treatment_exclusion")
        alt_results = do_search("Naturopathy non-allopathic treatment not approved Indian Medical Council", top_k=2)
        findings.append(
            Finding(
                text=(
                    "The treatment is a form of Alternative Treatment (Ayurveda/Homeopathy/Unani/"
                    "Siddha/Naturopathy), which the policy excludes as it only covers Allopathy/"
                    "modern medicine."
                ),
                evidence=_best(alt_results),
                kind="finding",
                severity="block",
            )
        )

    # ---- 6c. Pregnancy / infertility / assisted conception exclusion -----
    if any(k in diag_procedure for k in PREGNANCY_KEYWORDS):
        dimensions_checked.append("pregnancy_exclusion")
        preg_results = do_search("pregnancy childbirth miscarriage abortion infertility assisted conception excluded", top_k=2)
        findings.append(
            Finding(
                text=(
                    "The claim arises from pregnancy, childbirth, or infertility/assisted-"
                    "conception treatment, which the policy excludes."
                ),
                evidence=_best(preg_results),
                kind="finding",
                severity="block",
            )
        )

    # ---- 6d. Specifically-excluded illness list (item 20) -----------------
    matched_excluded_illness = next((d for d in EXCLUDED_ILLNESS_KEYWORDS if d in diag_procedure), None)
    if matched_excluded_illness:
        dimensions_checked.append("excluded_illness_list")
        excl_results = do_search("treatment of Asthma Bronchitis Diabetes Mellitus Hypertension excluded diseases", top_k=2)
        findings.append(
            Finding(
                text=(
                    f"The diagnosis/procedure text mentions '{matched_excluded_illness}', which is on the "
                    "policy's specifically-excluded illness list. However, this claim may be for a "
                    "complication requiring hospitalization rather than routine treatment of the "
                    "listed condition itself, and the supplied case facts do not clearly distinguish "
                    "the two, so this needs a human/clinical review rather than an automatic decision."
                ),
                evidence=_best(excl_results),
                kind="finding",
                severity="review",
            )
        )
        findings.append(
            Finding(
                text=(
                    f"Clinical confirmation of whether the hospitalization is for '{matched_excluded_illness}' "
                    "itself (excluded) or for a distinct complication arising from it (potentially payable)."
                ),
                evidence=None,
                kind="missing_evidence",
            )
        )

    # ---- 7. Experimental / unproven treatment exclusion ------------------
    if treatment.experimental:
        dimensions_checked.append("experimental_treatment")
        exp_results = do_search("Unproven Experimental Treatment not established medical practice", top_k=2)
        findings.append(
            Finding(
                text=(
                    "The treatment is flagged as experimental. The policy defines "
                    "'Unproven/Experimental Treatment' as treatment not based on established "
                    "medical practice in India, which this claim falls under, so it is not payable."
                ),
                evidence=_best(exp_results),
                kind="finding",
                severity="block",
            )
        )

    # ---- 8. Domiciliary treatment conditions + sub-limit -----------------
    if (treatment.type or "").lower() == "domiciliary":
        dimensions_checked.append("domiciliary_treatment")
        dom_results = do_search("Domiciliary Treatment confined at home cannot be removed room unavailable", top_k=2)
        eligible = bool(treatment.patient_cannot_be_moved) or bool(treatment.hospital_room_unavailable)
        dom_cap = 0.20 * case.sum_insured_inr
        dom_claim = (
            case.expenses_inr.room
            + case.expenses_inr.doctor_fees
            + case.expenses_inr.medicines_diagnostics
        )
        if not eligible:
            findings.append(
                Finding(
                    text=(
                        "Domiciliary treatment is claimed, but neither condition required by the "
                        "policy (patient not fit to be moved, or no hospital room available) is "
                        "confirmed in the case facts."
                    ),
                    evidence=_best(dom_results),
                    kind="finding",
                    severity="review",
                )
            )
        else:
            findings.append(
                Finding(
                    text="The domiciliary treatment conditions (room unavailable or patient not movable) are satisfied.",
                    evidence=_best(dom_results),
                    kind="finding",
                )
            )
            findings.append(
                Finding(
                    text=(
                        f"Domiciliary Hospitalization expenses are capped at 20% of Sum Insured "
                        f"(INR {dom_cap:,.0f}). Claimed: INR {dom_claim:,.0f} -> "
                        f"{'within limit' if dom_claim <= dom_cap else f'capped, payable INR {dom_cap:,.0f}'}."
                    ),
                    evidence=_best(dom_results),
                    kind="limit",
                )
            )

    # ---- 9. Day-care / <24hr treatment ------------------------------------
    treatment_type = (treatment.type or "").lower()
    if treatment_type != "domiciliary" and (
        treatment_type == "day_care"
        or (treatment.admission_hours is not None and treatment.admission_hours < 24)
    ):
        dimensions_checked.append("day_care_treatment")
        dc_results = do_search("Day Care Treatment less than 24 hours technological advancement", top_k=2)
        recognised = next((p for p in DAY_CARE_RECOGNISED if p in diag_procedure), None)
        if recognised:
            findings.append(
                Finding(
                    text=(
                        f"The procedure ('{recognised}') is one the policy recognises as qualifying "
                        "for less-than-24-hour Day Care Treatment because of technological "
                        "advancement, so the usual 24-hour minimum stay is waived."
                    ),
                    evidence=_best(dc_results),
                    kind="finding",
                )
            )
        else:
            findings.append(
                Finding(
                    text=(
                        "The treatment lasted less than 24 hours and is not clearly one of the "
                        "policy's recognised Day Care procedures, so it cannot be confirmed that the "
                        "24-hour minimum stay requirement is validly waived."
                    ),
                    evidence=_best(dc_results),
                    kind="finding",
                    severity="review",
                )
            )
            findings.append(
                Finding(
                    text="Confirmation that this specific procedure is on the policy's recognised Day Care Treatment list.",
                    evidence=None,
                    kind="missing_evidence",
                )
            )

    # ---- 10. Category sub-limits (doctor fees / other expenses / package) ----
    dimensions_checked.append("professional_fee_limit")
    doc_results = do_search("Medical Practitioner Surgeon Anesthetist fees 25% Sum Insured", top_k=2)
    doc_cap = 0.25 * case.sum_insured_inr
    doc_claim = case.expenses_inr.doctor_fees
    findings.append(
        Finding(
            text=(
                f"Medical Practitioner/Surgeon/Anesthetist fees are capped at 25% of Sum Insured "
                f"(INR {doc_cap:,.0f}). Claimed: INR {doc_claim:,.0f} -> "
                f"{'within limit' if doc_claim <= doc_cap else f'capped, payable INR {doc_cap:,.0f}'}."
            ),
            evidence=_best(doc_results),
            kind="limit",
        )
    )

    dimensions_checked.append("other_expense_limit")
    other_results = do_search("Anesthesia Blood Oxygen Medicines Diagnostic 40% Sum Insured", top_k=2)
    other_cap = 0.40 * case.sum_insured_inr
    other_claim = case.expenses_inr.medicines_diagnostics
    findings.append(
        Finding(
            text=(
                f"Anesthesia/medicines/diagnostics and similar expenses are capped at 40% of Sum "
                f"Insured (INR {other_cap:,.0f}). Claimed: INR {other_claim:,.0f} -> "
                f"{'within limit' if other_claim <= other_cap else f'capped, payable INR {other_cap:,.0f}'}."
            ),
            evidence=_best(other_results),
            kind="limit",
        )
    )

    dimensions_checked.append("room_sub_limit")
    room_results = do_search("Room Boarding Nursing expenses 1% Sum Insured per day ICU 2%", top_k=2)
    room_cap_per_day = 0.01 * case.sum_insured_inr
    findings.append(
        Finding(
            text=(
                f"Room/boarding/nursing expenses are capped at 1% of Sum Insured per day (INR "
                f"{room_cap_per_day:,.0f}/day); ICU expenses at 2%/day. Length of stay was not "
                "supplied, so the exact rupee cap for the room component cannot be finalised."
            ),
            evidence=_best(room_results),
            kind="limit",
        )
    )
    findings.append(
        Finding(
            text="Length of hospital stay (days) to finalise the per-day room-rent sub-limit.",
            evidence=None,
            kind="missing_evidence",
        )
    )

    dimensions_checked.append("package_cap")
    pkg_results = do_search("Any One Illness agreed package charges 75% Sum Insured", top_k=2)
    pkg_cap = 0.75 * case.sum_insured_inr
    total_claim = (
        case.expenses_inr.room
        + case.expenses_inr.doctor_fees
        + case.expenses_inr.medicines_diagnostics
        + case.expenses_inr.ambulance
    )
    if total_claim > pkg_cap:
        findings.append(
            Finding(
                text=(
                    f"Total hospitalization expenses for this illness (INR {total_claim:,.0f}) exceed "
                    f"the policy's 75% of Sum Insured package cap (INR {pkg_cap:,.0f}); the payable "
                    "amount is restricted to the cap."
                ),
                evidence=_best(pkg_results),
                kind="limit",
                severity="info",
            )
        )

    # ---- 11. Pre/Post hospitalization windows -----------------------------
    if case.expense_timing:
        dimensions_checked.append("pre_post_hospitalization_window")
        pp_results = do_search("Pre-Hospitalization 30 days Post-Hospitalization 60 days", top_k=2)
        pre_days = case.expense_timing.get("pre_hospitalization_days_before_admission", 0) or 0
        post_days = case.expense_timing.get("post_hospitalization_days_after_discharge", 0) or 0
        same_condition = case.expense_timing.get("same_condition_confirmed", False)
        pre_ok = pre_days <= 30 and same_condition
        post_ok = post_days <= 60 and same_condition
        findings.append(
            Finding(
                text=(
                    f"Pre-hospitalization expenses ({pre_days} days before admission) are "
                    f"{'within' if pre_ok else 'outside'} the policy's 30-day pre-hospitalization "
                    f"window; post-hospitalization expenses ({post_days} days after discharge) are "
                    f"{'within' if post_ok else 'outside'} the 60-day post-hospitalization window."
                ),
                evidence=_best(pp_results),
                kind="finding",
            )
        )
        if not pre_ok:
            findings.append(
                Finding(
                    text="Pre-hospitalization expenses claimed outside the 30-day window are not payable.",
                    evidence=_best(pp_results),
                    kind="limit",
                )
            )
        if not post_ok:
            findings.append(
                Finding(
                    text="Post-hospitalization expenses claimed outside the 60-day window are not payable.",
                    evidence=_best(pp_results),
                    kind="limit",
                )
            )

    # ---- 12. Portability credit note (informational) ----------------------
    if case.prior_policy:
        dimensions_checked.append("portability")
        port_results = do_search("Portability credit gained pre-existing conditions previous insurer", top_k=2)
        findings.append(
            Finding(
                text=(
                    "Continuous prior coverage with another Indian insurer is on record and, since "
                    "the database/claim history was received, is used above to reduce the "
                    "applicable waiting periods."
                ),
                evidence=_best(port_results),
                kind="finding",
            )
        )

    result = _aggregate(findings, dimensions_checked)
    result["evidence_pool"] = evidence_pool
    return result


def _aggregate(findings: List[Finding], dimensions_checked: List[str]) -> Dict:
    blocking = [f for f in findings if f.kind == "finding" and f.severity == "block"]
    review = [f for f in findings if f.kind == "finding" and f.severity == "review"]
    limits = [f for f in findings if f.kind == "limit"]
    missing_evidence = [f for f in findings if f.kind == "missing_evidence"]
    plain_findings = [f for f in findings if f.kind == "finding" and f.severity == "info"]

    if blocking:
        decision = "NOT_ADMISSIBLE"
        confidence = 0.85
    elif review:
        decision = "NEEDS_REVIEW"
        confidence = 0.55
    elif limits and any(("payable INR" in f.text) or ("exceed" in f.text) for f in limits):
        decision = "ADMISSIBLE_WITH_LIMITS"
        confidence = 0.8
    else:
        decision = "ADMISSIBLE"
        confidence = 0.85

    return {
        "decision": decision,
        "confidence": confidence,
        "findings": plain_findings + blocking + review,
        "limits": limits,
        "missing_evidence": missing_evidence,
        "dimensions_checked": dimensions_checked,
    }
