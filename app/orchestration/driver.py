from __future__ import annotations

import json
import logging
import inspect
import time
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from app.clients.l2 import L2Client, L2Response, parse_tool_arguments
from app.commit import compose, response_commit_tool, unpack
from app.config import Settings
from app.contracts import ChatCompletionRequest
from app.conversation import CompiledConversation, compile_conversation
from app.coverage import extract_contract
from app.deadline import Deadline
from app.errors import AppError, UpstreamError
from app.evidence.models import EvidenceRequirement, RetrievalOutcome
from app.evidence.session import RetrievalSession
from app.evidence.verification import EvidenceVerifier
from app.orchestration.planning import (
    RESPONSE_ANSWER_WITH_QUESTION,
    HarnessPlan,
    build_plan,
)
from app.orchestration.retrieval import RetrievalEngine
from app.orchestration.review import ConditionalReviewer, ReviewReason
from app.orchestration.state_enrichment import ClinicalStateEnricher
from app.observability import set_request_id
from app.prompts import generation_after_retrieval_prompt, generation_system_prompt


LOGGER = logging.getLogger("lunit_driver")


RETRIEVE_RELEVANT_CONTENT_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "retrieve_relevant_content",
        "description": (
            "Retrieve authoritative evidence for claims that depend on current or official "
            "external state. Submit only the evidence-dependent claims, not a plan for the "
            "whole answer."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "One self-contained evidence question with patient, population, "
                        "jurisdiction, and time constraints resolved from the conversation."
                    ),
                },
                "requirements": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 8,
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "claim_or_question": {"type": "string"},
                            "criticality": {
                                "type": "string",
                                "enum": ["critical", "supporting"],
                            },
                            "preferred_domain": {"type": "string"},
                        },
                        "required": ["id", "claim_or_question", "criticality"],
                        "additionalProperties": False,
                    },
                    "description": (
                        "Every atomic evidence-dependent claim in the query. Split independent "
                        "claims so each can be closed separately."
                    ),
                },
            },
            "required": ["query", "requirements"],
            "additionalProperties": False,
        },
    },
}


@dataclass(frozen=True, slots=True)
class DriverResult:
    content: str
    usage: dict[str, int]
    trace: dict[str, Any]


def _sum_usage(total: dict[str, int], usage: Mapping[str, int]) -> None:
    for key, value in usage.items():
        if isinstance(value, int) and value >= 0:
            total[key] = total.get(key, 0) + value


def _tool_error(call_id: str, code: str, detail: str) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "content": json.dumps({"error": code, "detail": detail}, ensure_ascii=False),
    }


def _commit_content(response: L2Response) -> tuple[str, int]:
    for call in response.tool_calls or ():
        if call.name != "commit_response":
            continue
        answer, question = unpack(call.arguments)
        content = compose(answer, question)
        if content:
            return content, 1 if question else 2
        return "", 0
    if response.content:
        return response.content, 3
    return "", 0


def _retrieval_requirements(
    arguments: Mapping[str, Any],
    query: str,
    planned: tuple[EvidenceRequirement, ...] = (),
) -> tuple[EvidenceRequirement, ...]:
    raw_requirements = arguments.get("requirements")
    requirements: list[EvidenceRequirement] = []
    seen: set[str] = set()
    candidates = raw_requirements if isinstance(raw_requirements, list) else []
    for index, raw in enumerate(candidates[:8], start=1):
        if not isinstance(raw, Mapping):
            continue
        raw_id = raw.get("id")
        raw_claim = raw.get("claim_or_question")
        identifier = raw_id.strip() if isinstance(raw_id, str) else f"req-{index}"
        claim = raw_claim.strip() if isinstance(raw_claim, str) else ""
        if not identifier or identifier in seen or not claim:
            continue
        seen.add(identifier)
        criticality = raw.get("criticality")
        if criticality not in {"critical", "supporting"}:
            criticality = "critical"
        preferred_domain = raw.get("preferred_domain")
        planned_fallback = planned[min(index - 1, len(planned) - 1)] if planned else None
        requirements.append(
            EvidenceRequirement(
                id=identifier[:80],
                claim_or_question=claim[:1_000],
                source_preference=(
                    preferred_domain.strip()[:80]
                    if isinstance(preferred_domain, str) and preferred_domain.strip()
                    else planned_fallback.source_preference
                    if planned_fallback is not None
                    else ""
                ),
                clinical_constraints=(
                    planned_fallback.clinical_constraints if planned_fallback is not None else ""
                ),
                jurisdiction=(
                    planned_fallback.jurisdiction if planned_fallback is not None else ""
                ),
                time_sensitive=(
                    planned_fallback.time_sensitive if planned_fallback is not None else False
                ),
                criticality=criticality,
            )
        )
    model_requirements = tuple(requirements)
    if len(planned) >= 2:
        return planned
    if len(model_requirements) >= 2:
        return model_requirements

    query_contract = extract_contract(query)
    if len(query_contract.requirements) >= 2:
        return tuple(
            EvidenceRequirement(
                id=f"query-{index}",
                claim_or_question=claim,
                source_preference=(planned[0].source_preference if planned else ""),
            )
            for index, claim in enumerate(query_contract.requirements[:8], start=1)
        )
    return model_requirements or planned or (
        EvidenceRequirement(id="req-1", claim_or_question=query),
    )


def _retrieval_available(
    retrieval_count: int,
    retrieval_budget: int,
    outcome: RetrievalOutcome | None,
) -> bool:
    return retrieval_count < retrieval_budget and not (
        outcome is not None and outcome.status == "sufficient"
    )


def _unresolved_retrieval_outcome(
    requirements: tuple[EvidenceRequirement, ...],
) -> RetrievalOutcome:
    unresolved = tuple(
        replace(
            requirement,
            status="unresolved",
            gap_reason="evidence_unavailable",
        )
        for requirement in requirements
    )
    return RetrievalOutcome(
        status="no_evidence",
        items=(),
        note=(
            "Authoritative evidence was unavailable. Answer from established medical "
            "knowledge without citations."
        ),
        evidence=(),
        requirements=unresolved,
        model_rounds=0,
        mcp_calls=0,
    )


def _requirement_counts(outcome: RetrievalOutcome | None) -> dict[str, int]:
    counts = {"missing": 0, "supported": 0, "contradicted": 0, "unresolved": 0}
    if outcome is None:
        return counts
    for requirement in outcome.requirements:
        if requirement.status in counts:
            counts[requirement.status] += 1
    return counts


class ConversationDriver:
    def __init__(
        self,
        settings: Settings,
        *,
        l2: L2Client,
        retrieval: RetrievalEngine,
    ) -> None:
        self._settings = settings
        self._l2 = l2
        self._retrieval = retrieval
        self._reviewer = ConditionalReviewer(l2)
        self._evidence_verifier = EvidenceVerifier(settings, l2)
        self._state_enricher = ClinicalStateEnricher(l2)

    def _complete_l2(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        deadline: Deadline,
        tools: Sequence[Mapping[str, Any]] | None = None,
        tool_choice: Any = None,
        recovery_messages: Sequence[Mapping[str, Any]] | None = None,
    ) -> L2Response:
        kwargs: dict[str, Any] = {
            "deadline": deadline,
            "tools": tools,
            "tool_choice": tool_choice,
        }
        # Test doubles and older compatible clients do not know the recovery keyword.
        if isinstance(self._l2, L2Client) and recovery_messages:
            kwargs["recovery_messages"] = recovery_messages
        return self._l2.complete(messages, **kwargs)

    def _run_retrieval(
        self,
        query: str,
        *,
        deadline: Deadline,
        requirements: tuple[EvidenceRequirement, ...],
        session: RetrievalSession | None = None,
    ) -> Any:
        parameters = inspect.signature(self._retrieval.run).parameters
        if "requirements" in parameters:
            kwargs: dict[str, Any] = {
                "deadline": deadline,
                "requirements": requirements,
            }
            if session is not None and "session" in parameters:
                kwargs["session"] = session
            return self._retrieval.run(query, **kwargs)
        return self._retrieval.run(query, deadline=deadline)

    def complete(
        self,
        request: ChatCompletionRequest,
        *,
        request_id: str,
    ) -> DriverResult:
        set_request_id(request_id)
        started = time.monotonic()
        deadline = (
            Deadline.unbounded()
            if self._settings.request_timeout_sec is None
            else Deadline.after(self._settings.request_timeout_sec)
        )
        compiled = compile_conversation(request.messages)
        state_enrichment_calls = 0
        state_enrichment_usage: dict[str, int] = {}
        state_enrichment_status = "not_requested"
        if (
            getattr(self._settings, "clinical_state_policy", "structured") == "hybrid"
            and isinstance(self._l2, L2Client)
            and compiled.state.clinical.needs_semantic_enrichment
            and deadline.can_start(1.0)
        ):
            enrichment = self._state_enricher.run(
                compiled.state.clinical,
                case_packet=compiled.case_packet,
                turn_roles=tuple(str(message.get("role", "unknown")) for message in compiled.messages),
                deadline=deadline.with_timeout_cap(min(self._settings.l2_timeout_sec, 30.0)),
            )
            state_enrichment_calls = enrichment.l2_calls
            state_enrichment_usage = enrichment.usage
            state_enrichment_status = enrichment.status
            if enrichment.state is not compiled.state.clinical:
                enriched_signals = tuple(
                    dict.fromkeys(
                        (*compiled.state.active_risk_signals, *enrichment.state.hard_risk_signals)
                    )
                )
                enriched_state = replace(
                    compiled.state,
                    clinical=enrichment.state,
                    active_risk_signals=enriched_signals,
                )
                compiled = replace(
                    compiled,
                    state=enriched_state,
                    is_high_risk=bool(enriched_signals),
                )
        contract = extract_contract(compiled.latest_user_text)
        plan = build_plan(self._settings, compiled, contract)
        retrieval_session = (
            # Seed on the first validated bridge call. A generation model may refine one
            # coarse planned requirement into several atomic ones; pre-seeding both shapes
            # would leave duplicate critical requirements in the cumulative ledger.
            RetrievalSession.create()
            if getattr(self._settings, "retrieval_session_policy", "cumulative")
            == "cumulative"
            else None
        )
        retrieval_budget = (
            self._settings.max_generation_retrievals if plan.retrieval_allowed else 0
        )
        require_question = plan.response_contract == RESPONSE_ANSWER_WITH_QUESTION
        commit_tool = response_commit_tool(require_question=require_question)

        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": generation_system_prompt(
                    retrieval_offered=plan.retrieval_allowed,
                    contract=(
                        plan.contract_spec
                        if getattr(self._settings, "response_contract_policy", "typed") == "typed"
                        else None
                    ),
                ),
            },
            *compiled.generation_messages(self._settings.conversation_representation),
        ]
        if contract.is_multipart:
            messages.append({"role": "system", "content": contract.as_prompt()})
        grounding_payloads: list[str] = []

        def current_recovery_messages() -> list[dict[str, Any]]:
            system_messages = tuple(
                message for message in messages if message.get("role") == "system"
            )
            return compiled.recovery_messages(
                requirements=contract.requirements,
                lead_system_messages=system_messages,
                grounding_payloads=grounding_payloads,
                plan_record={
                    "lane": plan.lane,
                    "clinical_risk": plan.clinical_risk.value,
                    "task_kind": plan.task_kind.value,
                    "interaction_mode": plan.interaction_mode.value,
                    "contract_kind": plan.contract_kind.value,
                    "missing_facts": [item.id for item in plan.missing_facts],
                    "clarification_reason": plan.clarification_reason,
                },
            )

        usage: dict[str, int] = dict(state_enrichment_usage)
        l2_calls = state_enrichment_calls
        retrieval_count = 0
        retrieval_l2_calls = 0
        mcp_calls = 0
        last_outcome: RetrievalOutcome | None = None
        draft = ""
        committed = 0
        retrieval_closed_prompt_added = False
        generation_seconds = 0.0
        retrieval_seconds = 0.0
        review_seconds = 0.0
        retrieval_diagnostics: dict[str, Any] = {}
        verification_l2_calls = 0
        verification_state = "not_used"
        verification_issues: tuple[str, ...] = ()
        history_compacted = False

        if self._settings.max_generation_retrievals == 0:
            messages.insert(
                1,
                {
                    "role": "system",
                    "content": generation_after_retrieval_prompt(
                        plan.contract_spec
                        if getattr(self._settings, "response_contract_policy", "typed") == "typed"
                        else None
                    ),
                },
            )
            retrieval_closed_prompt_added = True

        max_rounds = self._settings.max_generation_retrievals + 2
        for _ in range(max_rounds):
            available_tools: list[Mapping[str, Any]] = [commit_tool]
            if _retrieval_available(retrieval_count, retrieval_budget, last_outcome):
                available_tools.append(RETRIEVE_RELEVANT_CONTENT_TOOL)

            generation_started = time.monotonic()
            response = self._complete_l2(
                messages,
                deadline=deadline,
                tools=available_tools,
                tool_choice="required",
                recovery_messages=current_recovery_messages(),
            )
            generation_seconds += time.monotonic() - generation_started
            l2_calls += 1
            _sum_usage(usage, response.usage)
            history_compacted = history_compacted or response.recovery_mode != "none"

            draft, committed = _commit_content(response)
            if draft:
                break
            if not response.tool_calls:
                continue

            messages.append(response.assistant_message)
            retrieval_called = False
            for call in response.tool_calls:
                if call.name == "commit_response":
                    messages.append(
                        _tool_error(
                            call.id,
                            "invalid_commit_response",
                            "Submit a non-empty complete answer.",
                        )
                    )
                    continue
                if call.name != "retrieve_relevant_content":
                    messages.append(
                        _tool_error(
                            call.id,
                            "unknown_generation_tool",
                            "Use commit_response or retrieve_relevant_content.",
                        )
                    )
                    continue
                if not _retrieval_available(
                    retrieval_count,
                    retrieval_budget,
                    last_outcome,
                ):
                    messages.append(
                        _tool_error(
                            call.id,
                            "generation_retrieval_budget_exhausted",
                            "Commit the answer with the available conversation and evidence.",
                        )
                    )
                    continue
                try:
                    arguments = parse_tool_arguments(call.arguments)
                    query = arguments.get("query")
                    if not isinstance(query, str) or not query.strip():
                        raise ValueError("query must be a non-empty string")
                    if len(query) > self._settings.max_retrieval_query_chars:
                        raise ValueError("query exceeds the configured length limit")
                except (ValueError, json.JSONDecodeError) as exc:
                    messages.append(
                        _tool_error(call.id, "invalid_retrieval_query", str(exc))
                    )
                    continue

                retrieval_called = True
                retrieval_count += 1
                requirements = (
                    _retrieval_requirements(
                        arguments,
                        query.strip(),
                        plan.evidence_requirements,
                    )
                    if self._settings.retrieval_ledger_policy == "ledger"
                    else (
                        EvidenceRequirement(
                            id="req-1",
                            claim_or_question=query.strip(),
                        ),
                    )
                )
                retrieval_started = time.monotonic()
                try:
                    retrieval_deadline = deadline.with_timeout_cap(
                        self._settings.retrieval_timeout_sec
                    )
                    run = self._run_retrieval(
                        query.strip(),
                        deadline=retrieval_deadline,
                        requirements=requirements,
                        session=retrieval_session,
                    )
                    verification = self._evidence_verifier.run(
                        run.outcome,
                        clinical_risk=plan.clinical_risk.value,
                        deadline=deadline,
                        allow_semantic=(
                            isinstance(self._l2, L2Client)
                            and (
                                run.outcome.status == "sufficient"
                                or retrieval_count >= retrieval_budget
                            )
                        ),
                    )
                    verified_outcome = verification.outcome
                    verification_l2_calls += verification.l2_calls
                    l2_calls += verification.l2_calls
                    _sum_usage(usage, verification.usage)
                    verification_state = verification.state
                    verification_issues = tuple(
                        dict.fromkeys((*verification_issues, *verification.issues))
                    )
                    if retrieval_session is not None:
                        retrieval_session.apply_outcome(
                            verified_outcome,
                            diagnostics=dict(run.diagnostics or {}),
                        )
                        last_outcome = retrieval_session.snapshot(note=verified_outcome.note)
                        retrieval_diagnostics = retrieval_session.diagnostics()
                    else:
                        last_outcome = verified_outcome
                        retrieval_diagnostics = dict(run.diagnostics or {})
                    retrieval_l2_calls += run.l2_calls
                    mcp_calls += run.outcome.mcp_calls
                    _sum_usage(usage, run.usage)
                    # Generation must see the same verified cumulative ledger used by
                    # review and observability. Passing the raw per-run outcome here can
                    # forget earlier support or expose evidence rejected by verification.
                    content = last_outcome.to_generation_payload(
                        max_chars=self._settings.max_evidence_chars
                    )
                except AppError as exc:
                    LOGGER.warning(
                        "bounded retrieval failure",
                        extra={
                            "request_id": request_id,
                            "retrieval_error": exc.code,
                        },
                    )
                    failed_outcome = _unresolved_retrieval_outcome(requirements)
                    if retrieval_session is not None:
                        retrieval_session.apply_outcome(
                            failed_outcome,
                            diagnostics={"stop_reason": "retrieval_failure"},
                        )
                        last_outcome = retrieval_session.snapshot(note=failed_outcome.note)
                        retrieval_diagnostics = retrieval_session.diagnostics()
                    else:
                        last_outcome = failed_outcome
                    content = last_outcome.to_generation_payload(
                        max_chars=self._settings.max_evidence_chars
                    )
                finally:
                    retrieval_seconds += time.monotonic() - retrieval_started
                messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": content}
                )
                grounding_payloads.append(content)

            if (
                retrieval_called
                and not _retrieval_available(
                    retrieval_count,
                    retrieval_budget,
                    last_outcome,
                )
                and not retrieval_closed_prompt_added
            ):
                messages.append(
                    {
                        "role": "system",
                        "content": generation_after_retrieval_prompt(
                            plan.contract_spec
                            if getattr(self._settings, "response_contract_policy", "typed") == "typed"
                            else None
                        ),
                    }
                )
                retrieval_closed_prompt_added = True

        if not draft:
            if not retrieval_closed_prompt_added:
                messages.append(
                    {
                        "role": "system",
                        "content": generation_after_retrieval_prompt(
                            plan.contract_spec
                            if getattr(self._settings, "response_contract_policy", "typed") == "typed"
                            else None
                        ),
                    }
                )
            generation_started = time.monotonic()
            response = self._complete_l2(
                messages,
                deadline=deadline,
                tools=[commit_tool],
                tool_choice="required",
                recovery_messages=current_recovery_messages(),
            )
            generation_seconds += time.monotonic() - generation_started
            l2_calls += 1
            _sum_usage(usage, response.usage)
            history_compacted = history_compacted or response.recovery_mode != "none"
            draft, committed = _commit_content(response)
            if not draft:
                raise UpstreamError(
                    "L2 did not produce final assistant content after orchestration",
                    code="missing_final_l2_content",
                )

        review_reasons = list(plan.review_reasons)
        if history_compacted:
            review_reasons.append(ReviewReason.HISTORY_COMPACTED.value)
        if last_outcome is not None:
            critical = [
                requirement
                for requirement in last_outcome.requirements
                if requirement.criticality == "critical"
            ]
            if any(requirement.status == "contradicted" for requirement in critical):
                review_reasons.append(ReviewReason.CONFLICTING_EVIDENCE.value)
            if any(
                requirement.status in {"missing", "unresolved"}
                for requirement in critical
            ):
                review_reasons.append(ReviewReason.PARTIAL_CRITICAL_EVIDENCE.value)
        if any(
            marker in issue
            for issue in verification_issues
            for marker in ("mismatch", "date_unknown", "inapplicable", "insufficient")
        ):
            review_reasons.append(ReviewReason.UNCERTAIN_APPLICABILITY.value)

        review_reasons = list(dict.fromkeys(review_reasons))
        if self._settings.review_policy == "off":
            review_reasons = []
        elif self._settings.review_policy == "always":
            review_reasons = ["always"]
        elif not self._settings.enable_high_risk_review:
            review_reasons = [
                reason
                for reason in review_reasons
                if reason != ReviewReason.ACTIVE_HIGH_RISK.value
            ]

        reviewed = False
        revised = False
        review_state = "not_requested"
        if review_reasons and deadline.can_start(min(self._settings.l2_timeout_sec, 30.0)):
            review_started = time.monotonic()
            evidence = (
                last_outcome.to_generation_payload(
                    max_chars=self._settings.max_evidence_chars
                )
                if last_outcome
                else json.dumps({"status": "not_used", "evidence": []})
            )
            review_run = self._reviewer.run(
                compiled,
                draft,
                evidence,
                review_reasons,
                deadline=deadline,
                contract=(
                    plan.contract_spec
                    if getattr(self._settings, "response_contract_policy", "typed") == "typed"
                    else None
                ),
            )
            draft = review_run.draft
            l2_calls += review_run.l2_calls
            _sum_usage(usage, review_run.usage)
            reviewed = review_run.reviewed
            revised = review_run.revised
            review_state = review_run.state
            review_seconds += time.monotonic() - review_started
        elif review_reasons:
            review_state = "insufficient_deadline"

        counts = _requirement_counts(last_outcome)
        final_lane = "GROUNDED" if retrieval_count and plan.lane != "HIGH_RISK" else plan.lane
        trace = {
            "request_id": request_id,
            "history_hash": compiled.history_hash,
            "lane": final_lane,
            "lane_reasons": ",".join(plan.lane_reasons),
            "representation": self._settings.conversation_representation,
            "planner_policy": self._settings.planner_policy,
            "clinical_state_policy": getattr(
                self._settings, "clinical_state_policy", "legacy"
            ),
            "admission_policy": getattr(self._settings, "admission_policy", "legacy"),
            "task_kind": plan.task_kind.value,
            "contract_kind": plan.contract_kind.value,
            "clinical_risk": plan.clinical_risk.value,
            "interaction_mode": plan.interaction_mode.value,
            "clinical_fact_count": len(compiled.state.clinical.facts),
            "clinical_symptom_count": len(compiled.state.clinical.symptoms),
            "clinical_medication_count": len(compiled.state.clinical.medications),
            "state_enrichment_status": state_enrichment_status,
            "state_enrichment_l2_calls": state_enrichment_calls,
            "retrieval_ledger_policy": self._settings.retrieval_ledger_policy,
            "retrieval_session_policy": getattr(
                self._settings, "retrieval_session_policy", "legacy"
            ),
            "l2_calls": l2_calls,
            "retrieval_l2_calls": retrieval_l2_calls,
            "retrievals": retrieval_count,
            "mcp_calls": mcp_calls,
            "retrieval_stop_reason": retrieval_diagnostics.get("stop_reason", "not_used"),
            "retrieval_bridge_calls": retrieval_diagnostics.get("bridge_calls", 0),
            "retrieval_duplicates_blocked": retrieval_diagnostics.get(
                "duplicate_blocked", 0
            ),
            "retrieval_invalid_calls": retrieval_diagnostics.get("invalid_calls", 0),
            "retrieval_no_progress": retrieval_diagnostics.get("no_progress", 0),
            "retrieval_status": last_outcome.status if last_outcome else "not_used",
            "retrieval_requirements": sum(counts.values()),
            "requirements_supported": counts["supported"],
            "requirements_contradicted": counts["contradicted"],
            "requirements_unresolved": counts["missing"] + counts["unresolved"],
            "evidence_count": len(last_outcome.evidence) if last_outcome else 0,
            "verification_policy": getattr(
                self._settings, "evidence_verification_policy", "structural"
            ),
            "verification_state": verification_state,
            "verification_l2_calls": verification_l2_calls,
            "verification_issue_count": len(verification_issues),
            "review_reasons": ",".join(review_reasons),
            "review_state": review_state,
            "reviewed": reviewed,
            "revised": revised,
            "answer_requirements": len(contract.requirements),
            "lane_admission": plan.admission.lane,
            "committed": committed,
            "clarification_required": plan.clarification_required,
            "clarification_reason": plan.clarification_reason,
            "history_compacted": history_compacted,
            "admission_domains": ",".join(plan.retrieval_domains),
            "response_chars": len(draft),
            "generation_latency_ms": round(generation_seconds * 1000),
            "retrieval_latency_ms": round(retrieval_seconds * 1000),
            "review_latency_ms": round(review_seconds * 1000),
            "latency_ms": round((time.monotonic() - started) * 1000),
            "state": "completed",
        }
        LOGGER.info(json.dumps(trace, ensure_ascii=False, separators=(",", ":")))
        return DriverResult(content=draft, usage=usage, trace=trace)
