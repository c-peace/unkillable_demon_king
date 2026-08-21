from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from app.clinical_state import AssertionState, ClinicalState


class TaskKind(StrEnum):
    TRIAGE = "triage"
    MEDICATION_SAFETY = "medication_safety"
    DOSING = "dosing"
    GENERAL_EDUCATION = "general_education"
    TEXT_TRANSFORMATION = "text_transformation"
    GUIDELINE_LOOKUP = "guideline_lookup"
    CODING = "coding"
    COVERAGE = "coverage"
    LAW = "law"
    RESEARCH_SUMMARY = "research_summary"


class ClinicalRisk(StrEnum):
    ROUTINE = "routine"
    ELEVATED = "elevated"
    URGENT = "urgent"
    EMERGENCY = "emergency"


class InteractionMode(StrEnum):
    ANSWER = "answer"
    ANSWER_AND_CLARIFY = "answer_and_clarify"
    CLARIFY_FIRST = "clarify_first"
    TRANSFORM_TEXT = "transform_text"


class EvidenceNeedType(StrEnum):
    GUIDELINE = "guideline"
    DRUG_LABEL = "drug_label"
    REGULATORY = "regulatory"
    COVERAGE = "coverage"
    CODING = "coding"
    LAW = "law"
    RESEARCH = "research"
    SAFETY_SIGNAL = "safety_signal"


@dataclass(frozen=True, slots=True)
class MissingFact:
    id: str
    reason: str
    criticality: str
    blocks_retrieval: bool
    response_mode: InteractionMode


@dataclass(frozen=True, slots=True)
class EvidenceNeed:
    id: str
    claim: str
    need_type: EvidenceNeedType
    source_family: str
    jurisdiction: str = ""
    time_sensitive: bool = False
    clinical_constraints: str = ""
    criticality: str = "critical"


@dataclass(frozen=True, slots=True)
class ResponseContract:
    kind: TaskKind
    required_checks: tuple[str, ...]
    forbidden_expansions: tuple[str, ...]
    reviewer_categories: tuple[str, ...]
    prompt: str


_DOSING = re.compile(r"용량|복용량|몇\s*(?:mg|정)|dose|dosage|how much", re.IGNORECASE)
_DRUG_SAFETY = re.compile(
    r"상호작용|병용|금기|부작용|경고|안전|같이\s*(?:먹|복용)|"
    r"먹어도|복용해도|계속\s*(?:먹|복용)|"
    r"interaction|contraindicat(?:ion|ed)|adverse|side effects?|warning|safe|take\s+.+\s+with|"
    r"keep taking|continue taking",
    re.IGNORECASE,
)
_GUIDELINE = re.compile(
    r"가이드라인|진료지침|권고|목표(?:\s*(?:치|값|수치))?|기준치|"
    r"guideline|recommendation|treatment target|target value|\btarget\b|\bgoal\b|threshold",
    re.IGNORECASE,
)
_CODING = re.compile(r"kcd|질병코드|상병코드|진단코드|disease code|billing code", re.IGNORECASE)
_COVERAGE = re.compile(r"급여|보험|약가|심평원|hira|reimbursement|coverage|ceiling price", re.IGNORECASE)
_LAW = re.compile(
    r"의료법|약사법|의료기기법|개인정보\s*보호법|법령|법률|조문|제\s*\d+\s*조|"
    r"law|statute|regulation",
    re.IGNORECASE,
)
_RESEARCH = re.compile(r"논문|연구|임상시험|pubmed|study|trial|meta-analysis|systematic review", re.IGNORECASE)
_REGULATORY = re.compile(r"허가|승인|적응증|approval|approved|indication|mfds", re.IGNORECASE)
_MECHANISM = re.compile(r"왜|기전|원리|원인|how does|why does|mechanism", re.IGNORECASE)
_PEDIATRIC = re.compile(r"소아|영아|신생아|아이|아기|child|kid|infant|newborn|pediatric", re.IGNORECASE)
_PERSONAL = re.compile(r"\b(?:i|my|me)\b|저는|제가|나는|내가|제\s|우리\s*아이", re.IGNORECASE)
_HIGH_RISK_DRUG_RELATION = re.compile(
    r"상호작용|병용|금기|같이\s*(?:먹|복용)|임신|수유|"
    r"interaction|contraindicat(?:ion|ed)|take\s+.+\s+with|pregnan|breastfeed",
    re.IGNORECASE,
)
_VERIFY_TEXT = re.compile(r"검증|확인|사실인지|의학적으로|fact[- ]?check|verify|medically accurate", re.IGNORECASE)


def classify_task(latest_user_text: str, clinical: ClinicalState) -> TaskKind:
    text = latest_user_text.strip()
    if clinical.interaction.requested_format == "text_operation":
        return TaskKind.TEXT_TRANSFORMATION
    if _CODING.search(text):
        return TaskKind.CODING
    if _COVERAGE.search(text):
        return TaskKind.COVERAGE
    if _LAW.search(text):
        return TaskKind.LAW
    if _RESEARCH.search(text):
        return TaskKind.RESEARCH_SUMMARY
    if _GUIDELINE.search(text):
        return TaskKind.GUIDELINE_LOOKUP
    if _DOSING.search(text):
        return TaskKind.DOSING
    if _DRUG_SAFETY.search(text) and not (
        _MECHANISM.search(text)
        and re.search(r"부작용|adverse effect|side effect", text, re.I)
        and not re.search(r"상호작용|금기|안전|interaction|contraindicat|safe", text, re.I)
    ):
        return TaskKind.MEDICATION_SAFETY
    if clinical.hard_risk_signals or clinical.active_symptom_names:
        return TaskKind.TRIAGE
    return TaskKind.GENERAL_EDUCATION


def classify_risk(task: TaskKind, latest_user_text: str, clinical: ClinicalState) -> ClinicalRisk:
    if clinical.hard_risk_signals:
        return ClinicalRisk.EMERGENCY
    if task == TaskKind.TRIAGE and clinical.active_symptom_names:
        return ClinicalRisk.URGENT
    if task == TaskKind.DOSING:
        return ClinicalRisk.ELEVATED
    if task == TaskKind.MEDICATION_SAFETY:
        active_reproductive_fact = any(
            fact is not None
            and fact.assertion not in {AssertionState.NEGATED, AssertionState.HISTORICAL}
            for fact in (
                clinical.latest_fact("pregnancy"),
                clinical.latest_fact("breastfeeding"),
            )
        )
        if (
            _PERSONAL.search(latest_user_text)
            or _HIGH_RISK_DRUG_RELATION.search(latest_user_text)
            or active_reproductive_fact
        ):
            return ClinicalRisk.ELEVATED
    return ClinicalRisk.ROUTINE


def decision_missing_facts(
    task: TaskKind,
    latest_user_text: str,
    clinical: ClinicalState,
    legacy_missing: tuple[str, ...] = (),
) -> tuple[MissingFact, ...]:
    missing: list[MissingFact] = []
    existing = set(legacy_missing)
    if task == TaskKind.DOSING and not clinical.named_medications:
        existing.add("medication_name")
    if "medication_name" in existing:
        missing.append(
            MissingFact(
                id="medication_name",
                reason="A dosing lookup and personal dose are not meaningful without the medicine.",
                criticality="critical",
                blocks_retrieval=True,
                response_mode=InteractionMode.CLARIFY_FIRST,
            )
        )
    if task == TaskKind.DOSING and _PEDIATRIC.search(latest_user_text):
        for identifier, kind in (("child_age", "age"), ("child_weight", "weight")):
            if not clinical.facts_of_kind(kind):
                missing.append(
                    MissingFact(
                        id=identifier,
                        reason=f"Pediatric dosing depends on {kind}.",
                        criticality="critical",
                        blocks_retrieval=False,
                        response_mode=InteractionMode.CLARIFY_FIRST,
                    )
                )
    if clinical.hard_risk_signals:
        missing.append(
            MissingFact(
                id="emergency_context",
                reason="Additional context may refine advice but must not delay immediate action.",
                criticality="supporting",
                blocks_retrieval=False,
                response_mode=InteractionMode.ANSWER_AND_CLARIFY,
            )
        )
    for identifier in existing:
        if any(item.id == identifier for item in missing):
            continue
        missing.append(
            MissingFact(
                id=identifier,
                reason="The missing fact can materially change the answer.",
                criticality="critical",
                blocks_retrieval=identifier == "reference_target",
                response_mode=InteractionMode.CLARIFY_FIRST,
            )
        )
    return tuple(missing)


def _constraints(clinical: ClinicalState) -> str:
    parts: list[str] = []
    if clinical.interaction.jurisdiction:
        parts.append(f"jurisdiction={clinical.interaction.jurisdiction}")
    for kind in ("pregnancy", "breastfeeding", "age", "weight", "allergy"):
        fact = clinical.latest_fact(kind)
        if fact is not None and fact.assertion not in {
            AssertionState.NEGATED,
            AssertionState.HISTORICAL,
        }:
            parts.append(f"{kind}={fact.value}")
    return ";".join(parts)[:500]


def plan_evidence_needs(
    task: TaskKind,
    latest_user_text: str,
    clinical: ClinicalState,
) -> tuple[EvidenceNeed, ...]:
    mapping: dict[TaskKind, tuple[EvidenceNeedType, str, bool]] = {
        TaskKind.GUIDELINE_LOOKUP: (EvidenceNeedType.GUIDELINE, "guideline", True),
        TaskKind.MEDICATION_SAFETY: (EvidenceNeedType.DRUG_LABEL, "drug_safety", True),
        TaskKind.CODING: (EvidenceNeedType.CODING, "coding", True),
        TaskKind.COVERAGE: (EvidenceNeedType.COVERAGE, "hira", True),
        TaskKind.LAW: (EvidenceNeedType.LAW, "law", True),
        TaskKind.RESEARCH_SUMMARY: (EvidenceNeedType.RESEARCH, "research", True),
    }
    if task == TaskKind.DOSING and clinical.named_medications:
        mapping_value = (EvidenceNeedType.DRUG_LABEL, "drug", True)
    elif task == TaskKind.TEXT_TRANSFORMATION and _VERIFY_TEXT.search(latest_user_text):
        if _DRUG_SAFETY.search(latest_user_text) or _DOSING.search(latest_user_text):
            mapping_value = (EvidenceNeedType.DRUG_LABEL, "drug_safety", True)
        elif _CODING.search(latest_user_text):
            mapping_value = (EvidenceNeedType.CODING, "coding", True)
        elif _LAW.search(latest_user_text):
            mapping_value = (EvidenceNeedType.LAW, "law", True)
        else:
            mapping_value = (EvidenceNeedType.GUIDELINE, "guideline", False)
    elif task == TaskKind.GENERAL_EDUCATION and _REGULATORY.search(latest_user_text):
        mapping_value = (EvidenceNeedType.REGULATORY, "drug", True)
    else:
        mapping_value = mapping.get(task)
    if mapping_value is None:
        return ()
    need_type, source_family, time_sensitive = mapping_value
    jurisdiction = clinical.interaction.jurisdiction
    return (
        EvidenceNeed(
            id="evidence-1",
            claim=latest_user_text.strip()[:1_000],
            need_type=need_type,
            source_family=source_family,
            jurisdiction=jurisdiction,
            time_sensitive=time_sensitive,
            clinical_constraints=_constraints(clinical),
        ),
    )


_CONTRACTS: dict[TaskKind, ResponseContract] = {
    TaskKind.TRIAGE: ResponseContract(
        TaskKind.TRIAGE,
        ("direct_action", "urgency", "material_red_flags", "safe_interim_action"),
        ("definitive_diagnosis", "encyclopedic_differential"),
        ("missing_action", "wrong_urgency", "unsafe_action", "conversation_conflict"),
        "For a triage request, lead with what to do now and the proportional urgency. Include only the red flags and interim actions that can change the decision; do not claim a definitive diagnosis.",
    ),
    TaskKind.MEDICATION_SAFETY: ResponseContract(
        TaskKind.MEDICATION_SAFETY,
        ("direct_safety_answer", "patient_modifiers", "change_or_stop_caution", "evidence_applicability"),
        ("unrelated_condition_overview",),
        ("unsupported_claim", "missing_modifier", "unsafe_action", "citation_mismatch"),
        "For medication safety, give the direct answer, apply the known medicine and patient modifiers, and distinguish label evidence from individualized prescribing. Do not add an unrelated disease overview.",
    ),
    TaskKind.DOSING: ResponseContract(
        TaskKind.DOSING,
        ("medicine_identity", "decision_changing_inputs", "safe_limits", "clarification"),
        ("unconditional_personal_dose",),
        ("unsafe_dose", "missing_modifier", "unsupported_claim"),
        "For dosing, never invent a personal dose when a decision-changing medicine, age, weight, route, or indication is missing. Give safe general boundaries and the exact clarification the contract requests.",
    ),
    TaskKind.GENERAL_EDUCATION: ResponseContract(
        TaskKind.GENERAL_EDUCATION,
        ("direct_explanation", "material_exceptions"),
        ("automatic_emergency_referral", "unrequested_management_plan"),
        ("factual_error", "instruction_miss"),
        "For general education, answer the explanation at the requested depth. Do not add automatic emergency, referral, or management boilerplate unless the conversation itself warrants it.",
    ),
    TaskKind.TEXT_TRANSFORMATION: ResponseContract(
        TaskKind.TEXT_TRANSFORMATION,
        ("finished_artifact", "meaning_preserved", "format_followed"),
        ("treat_quoted_text_as_patient_state",),
        ("unfinished_artifact", "meaning_changed", "format_miss"),
        "Perform the requested text operation and return a complete usable artifact. Treat clinical content inside the source as quoted material unless the user explicitly asks for medical verification.",
    ),
    TaskKind.GUIDELINE_LOOKUP: ResponseContract(
        TaskKind.GUIDELINE_LOOKUP,
        ("recommendation", "population", "source_scope", "uncertainty"),
        ("unrelated_triage",),
        ("unsupported_claim", "applicability_error", "citation_mismatch"),
        "Answer the requested guideline question with its population and scope. Do not add unrelated triage or personal treatment instructions.",
    ),
    TaskKind.CODING: ResponseContract(
        TaskKind.CODING,
        ("code", "official_name", "version", "limitations"),
        ("triage", "dosing", "differential_diagnosis"),
        ("wrong_identifier", "version_miss", "citation_mismatch"),
        "For coding, provide the identifier, official name and version or scope. Do not expand into triage, dosing, or a differential diagnosis.",
    ),
    TaskKind.COVERAGE: ResponseContract(
        TaskKind.COVERAGE,
        ("coverage_answer", "jurisdiction", "effective_scope"),
        ("triage", "dosing", "differential_diagnosis"),
        ("applicability_error", "outdated_source", "citation_mismatch"),
        "For coverage, answer the reimbursement question for the stated jurisdiction and effective scope. Do not add unrelated clinical management.",
    ),
    TaskKind.LAW: ResponseContract(
        TaskKind.LAW,
        ("legal_answer", "jurisdiction", "article_or_scope", "effective_date"),
        ("triage", "dosing", "differential_diagnosis"),
        ("wrong_provision", "applicability_error", "outdated_source", "citation_mismatch"),
        "For law, answer the provision and jurisdiction requested, with effective scope when known. Do not add unrelated medical management.",
    ),
    TaskKind.RESEARCH_SUMMARY: ResponseContract(
        TaskKind.RESEARCH_SUMMARY,
        ("research_scope", "main_result", "limitations", "research_vs_recommendation"),
        ("personal_treatment_directive",),
        ("overclaim", "missing_limitation", "citation_mismatch"),
        "For research, separate study findings from clinical recommendations and preserve important limitations. Do not turn a study summary into a personal treatment directive.",
    ),
}


def response_contract(task: TaskKind) -> ResponseContract:
    return _CONTRACTS[task]
