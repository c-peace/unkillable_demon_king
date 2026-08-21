from __future__ import annotations

from dataclasses import dataclass

from app.admission import Admission, admit
from app.config import Settings
from app.conversation import CompiledConversation
from app.coverage import AnswerContract
from app.evidence.models import EvidenceRequirement
from app.evidence.routing import SourceRouter
from app.response_contracts import (
    ClinicalRisk,
    EvidenceNeed,
    InteractionMode,
    MissingFact,
    ResponseContract,
    TaskKind,
    classify_risk,
    classify_task,
    decision_missing_facts,
    plan_evidence_needs,
    response_contract as typed_response_contract,
)


_STATE_DEPENDENT = frozenset(
    {
        "drug",
        "drug_safety",
        "guideline",
        "hira",
        "coding",
        "law",
        "research",
        "faers",
    }
)

_SOURCE_SIGNAL_TO_DOMAIN = {
    "drug_safety": "drug_safety",
    "guideline": "guideline",
    "regulatory": "drug",
    "coverage": "hira",
    "coding": "coding",
    "law": "law",
    "research": "research",
}

RESPONSE_ANSWER_ONLY = "answer_only"
RESPONSE_ANSWER_WITH_QUESTION = "answer_with_question"


@dataclass(frozen=True, slots=True)
class HarnessPlan:
    lane: str
    lane_reasons: tuple[str, ...]
    retrieval_allowed: bool
    retrieval_domains: tuple[str, ...]
    evidence_requirements: tuple[EvidenceRequirement, ...]
    clarification_required: bool
    clarification_reason: str
    response_contract: str
    review_reasons: tuple[str, ...]
    recovery_policy: str
    admission: Admission
    clinical_risk: ClinicalRisk
    task_kind: TaskKind
    interaction_mode: InteractionMode
    contract_kind: TaskKind
    missing_facts: tuple[MissingFact, ...]
    evidence_needs: tuple[EvidenceNeed, ...]
    contract_spec: ResponseContract


def _unique(values: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return tuple(ordered)


def _state_domains(compiled: CompiledConversation) -> tuple[str, ...]:
    router = SourceRouter("route")
    routed = [
        domain
        for domain in router.routes(compiled.state.retrieval_context())
        if domain in _STATE_DEPENDENT
    ]
    fallback = [
        _SOURCE_SIGNAL_TO_DOMAIN[signal]
        for signal in compiled.state.source_sensitive_signals
        if signal in _SOURCE_SIGNAL_TO_DOMAIN
    ]
    return _unique([*routed, *fallback])


def _planned_evidence_requirements(
    compiled: CompiledConversation,
    contract: AnswerContract,
    domains: tuple[str, ...],
) -> tuple[EvidenceRequirement, ...]:
    if not domains:
        return ()
    router = SourceRouter("route")
    evidence_claims: list[tuple[str, str]] = []
    for claim in contract.requirements:
        claim_domains = tuple(domain for domain in router.routes(claim) if domain in domains)
        if claim_domains:
            evidence_claims.append((claim, claim_domains[0]))
    if not evidence_claims:
        evidence_claims.append((compiled.state.retrieval_context(), domains[0]))
    return tuple(
        EvidenceRequirement(
            id=f"plan-{index}",
            claim_or_question=claim,
            source_preference=domain,
        )
        for index, (claim, domain) in enumerate(evidence_claims[:6], start=1)
        if claim.strip()
    )


def _claim_evidence_requirements(
    compiled: CompiledConversation,
    contract: AnswerContract,
    planned_needs: tuple[EvidenceNeed, ...],
) -> tuple[EvidenceRequirement, ...]:
    requirements: list[EvidenceRequirement] = []
    claims = contract.requirements if contract.is_multipart else ()
    for claim in claims:
        task = classify_task(claim, compiled.state.clinical)
        for need in plan_evidence_needs(task, claim, compiled.state.clinical):
            requirements.append(
                EvidenceRequirement(
                    id=f"plan-{len(requirements) + 1}",
                    claim_or_question=claim,
                    source_preference=need.source_family,
                    clinical_constraints=need.clinical_constraints,
                    jurisdiction=need.jurisdiction,
                    time_sensitive=need.time_sensitive,
                )
            )
            break
    if requirements:
        return tuple(requirements[:6])
    return tuple(
        EvidenceRequirement(
            id=f"plan-{index}",
            claim_or_question=need.claim,
            source_preference=need.source_family,
            clinical_constraints=need.clinical_constraints,
            jurisdiction=need.jurisdiction,
            time_sensitive=need.time_sensitive,
        )
        for index, need in enumerate(planned_needs[:6], start=1)
    )


def build_harness_plan(
    compiled: CompiledConversation,
    settings: Settings,
    contract: AnswerContract | None = None,
) -> HarnessPlan:
    answer_contract = contract or compiled.state.answer_contract
    task_text = (
        compiled.state.retrieval_context()
        if compiled.state.has_follow_up_reference
        else compiled.latest_user_text
    )
    task_kind = classify_task(task_text, compiled.state.clinical)
    clinical_risk = classify_risk(task_kind, task_text, compiled.state.clinical)
    contract_spec = typed_response_contract(task_kind)
    missing_facts = decision_missing_facts(
        task_kind,
        compiled.latest_user_text,
        compiled.state.clinical,
        compiled.state.decision_critical_missing_facts,
    )
    evidence_needs = plan_evidence_needs(
        task_kind,
        compiled.latest_user_text,
        compiled.state.clinical,
    )
    active_risk = clinical_risk != ClinicalRisk.ROUTINE
    if (
        settings.planner_policy == "legacy"
        or getattr(settings, "clinical_state_policy", "structured") == "legacy"
    ):
        admission = admit(compiled.latest_user_text)
        domains = admission.domains
        retrieval_allowed = admission.admitted and not compiled.state.is_text_operation
        lane = "HIGH_RISK" if active_risk else "GROUNDED" if retrieval_allowed else "DIRECT"
        return HarnessPlan(
            lane=lane,
            lane_reasons=("legacy_policy",),
            retrieval_allowed=retrieval_allowed,
            retrieval_domains=domains,
            evidence_requirements=(
                _planned_evidence_requirements(compiled, answer_contract, domains)
                if retrieval_allowed
                else ()
            ),
            clarification_required=False,
            clarification_reason="",
            response_contract=RESPONSE_ANSWER_ONLY,
            review_reasons=("active_high_risk",) if active_risk else (),
            recovery_policy=settings.empty_recovery_policy,
            admission=admission,
            clinical_risk=clinical_risk,
            task_kind=task_kind,
            interaction_mode=InteractionMode.ANSWER,
            contract_kind=task_kind,
            missing_facts=missing_facts,
            evidence_needs=evidence_needs,
            contract_spec=contract_spec,
        )

    admission_policy = getattr(settings, "admission_policy", "claim")
    if admission_policy == "claim":
        admission = Admission(_unique([need.source_family for need in evidence_needs]))
    else:
        admission = Admission(_state_domains(compiled))
    domains = admission.domains
    retrieval_blocked = any(item.blocks_retrieval for item in missing_facts)
    retrieval_allowed = admission.admitted and not retrieval_blocked
    lane_reasons: list[str] = []
    if retrieval_allowed:
        lane_reasons.append("source_sensitive")
        lane_reasons.extend(f"domain:{domain}" for domain in domains)
    if active_risk:
        lane_reasons.extend(f"risk:{signal}" for signal in compiled.state.active_risk_signals)
    if compiled.state.has_follow_up_reference:
        lane_reasons.append("follow_up_reference")

    clarification_required = False
    clarification_reason = ""
    interaction_mode = (
        InteractionMode.TRANSFORM_TEXT
        if task_kind == TaskKind.TEXT_TRANSFORMATION
        else InteractionMode.ANSWER
    )
    if settings.clarification_policy == "always" and task_kind != TaskKind.TEXT_TRANSFORMATION:
        clarification_required = True
        clarification_reason = "policy_always"
        interaction_mode = InteractionMode.ANSWER_AND_CLARIFY
    elif settings.clarification_policy == "conditional":
        if missing_facts and task_kind != TaskKind.TEXT_TRANSFORMATION:
            clarification_required = True
            clarification_reason = ",".join(item.id for item in missing_facts)
            interaction_mode = (
                InteractionMode.ANSWER_AND_CLARIFY
                if any(item.response_mode == InteractionMode.ANSWER_AND_CLARIFY for item in missing_facts)
                else InteractionMode.CLARIFY_FIRST
            )

    if active_risk:
        lane = "HIGH_RISK"
    elif retrieval_allowed:
        lane = "GROUNDED"
    elif clarification_required:
        lane = "CLARIFY"
    else:
        lane = "DIRECT"

    review_reasons: list[str] = []
    if active_risk:
        review_reasons.append("active_high_risk")
    if answer_contract.is_multipart and compiled.state.strict_output_requested:
        review_reasons.append("strict_multipart_output")
    return HarnessPlan(
        lane=lane,
        lane_reasons=tuple(lane_reasons),
        retrieval_allowed=retrieval_allowed,
        retrieval_domains=domains,
        evidence_requirements=(
            _claim_evidence_requirements(compiled, answer_contract, evidence_needs)
            if retrieval_allowed and admission_policy == "claim"
            else _planned_evidence_requirements(compiled, answer_contract, domains)
            if retrieval_allowed
            else ()
        ),
        clarification_required=clarification_required,
        clarification_reason=clarification_reason,
        response_contract=(
            RESPONSE_ANSWER_WITH_QUESTION
            if clarification_required
            else RESPONSE_ANSWER_ONLY
        ),
        review_reasons=tuple(review_reasons),
        recovery_policy=settings.empty_recovery_policy,
        admission=admission,
        clinical_risk=clinical_risk,
        task_kind=task_kind,
        interaction_mode=interaction_mode,
        contract_kind=task_kind,
        missing_facts=missing_facts,
        evidence_needs=evidence_needs,
        contract_spec=contract_spec,
    )


def build_plan(
    settings: Settings,
    compiled: CompiledConversation,
    contract: object | None = None,
) -> HarnessPlan:
    answer_contract = contract if isinstance(contract, AnswerContract) else None
    return build_harness_plan(compiled, settings, answer_contract)
