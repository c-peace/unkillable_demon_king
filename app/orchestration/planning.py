from __future__ import annotations

from dataclasses import dataclass

from app.admission import Admission, admit
from app.config import Settings
from app.conversation import CompiledConversation
from app.coverage import AnswerContract
from app.evidence.models import EvidenceRequirement
from app.evidence.routing import SourceRouter


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


def build_harness_plan(
    compiled: CompiledConversation,
    settings: Settings,
    contract: AnswerContract | None = None,
) -> HarnessPlan:
    answer_contract = contract or compiled.state.answer_contract
    active_risk = bool(compiled.state.active_risk_signals) and not compiled.state.is_text_operation
    if settings.planner_policy == "legacy":
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
        )

    admission = Admission(_state_domains(compiled))
    domains = admission.domains
    retrieval_allowed = admission.admitted and not compiled.state.is_text_operation
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
    if settings.clarification_policy == "always" and not compiled.state.is_text_operation:
        clarification_required = True
        clarification_reason = "policy_always"
    elif settings.clarification_policy == "conditional":
        if compiled.state.decision_critical_missing_facts and not compiled.state.is_text_operation:
            clarification_required = True
            clarification_reason = ",".join(compiled.state.decision_critical_missing_facts)

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
            _planned_evidence_requirements(compiled, answer_contract, domains)
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
    )


def build_plan(
    settings: Settings,
    compiled: CompiledConversation,
    contract: object | None = None,
) -> HarnessPlan:
    answer_contract = contract if isinstance(contract, AnswerContract) else None
    return build_harness_plan(compiled, settings, answer_contract)
