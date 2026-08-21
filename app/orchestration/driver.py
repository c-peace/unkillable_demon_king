from __future__ import annotations

import json
import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from app.clients.l2 import L2Client, parse_tool_arguments
from app.config import Settings
from app.contracts import ChatCompletionRequest
from app.conversation import CompiledConversation, compile_conversation
from app.deadline import Deadline
from app.errors import AppError, UpstreamError
from app.evidence.models import RetrievalOutcome
from app.observability import emit_event
from app.orchestration.planning import StructuredPlanner
from app.orchestration.retrieval import RetrievalEngine
from app.orchestration.review import StructuredReviewer, deterministic_review_issues
from app.prompts import (
    GENERATION_AFTER_RETRIEVAL_PROMPT,
    GENERATION_SYSTEM_PROMPT,
    PLANNED_GENERATION_PROMPT,
    REVIEW_SYSTEM_PROMPT,
    REVISION_SYSTEM_PROMPT,
)

LOGGER = logging.getLogger("lunit_driver")


RETRIEVE_RELEVANT_CONTENT_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "retrieve_relevant_content",
        "description": (
            "Retrieve authoritative evidence to ground the answer. Pass one self-contained query "
            "with relevant patient, population, jurisdiction, and time constraints."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A single, self-contained evidence question.",
                }
            },
            "required": ["query"],
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
        self._planner = StructuredPlanner(l2=l2)
        self._structured_reviewer = StructuredReviewer(settings, l2=l2)

    def complete(
        self,
        request: ChatCompletionRequest,
        *,
        request_id: str,
    ) -> DriverResult:
        if self._settings.enable_structured_planning:
            return self._complete_planned(request, request_id=request_id)
        return self._complete_legacy(request, request_id=request_id)

    def _complete_legacy(
        self,
        request: ChatCompletionRequest,
        *,
        request_id: str,
        planner_mode: str = "legacy",
        fallback_reason: str = "",
        planning_l2_calls: int = 0,
        planning_usage: Mapping[str, int] | None = None,
        started_at: float | None = None,
        deadline_override: Deadline | None = None,
    ) -> DriverResult:
        started = time.monotonic() if started_at is None else started_at
        deadline = deadline_override or (
            Deadline.unbounded()
            if self._settings.request_timeout_sec is None
            else Deadline.after(self._settings.request_timeout_sec)
        )
        compiled = compile_conversation(request.messages)
        emit_event(
            LOGGER,
            "request_started",
            request_id=request_id,
            message_count=len(request.messages),
            representation=self._settings.conversation_representation,
            max_generation_retrievals=self._settings.max_generation_retrievals,
            max_retrieval_model_rounds=self._settings.max_retrieval_model_rounds,
            max_mcp_tool_calls=self._settings.max_mcp_tool_calls,
            planner_mode=planner_mode,
            ledger_mode="legacy",
            contract_mode="legacy",
            review_mode=(
                "legacy_conditional" if self._settings.enable_high_risk_review else "disabled"
            ),
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
            *compiled.generation_messages(self._settings.conversation_representation),
        ]
        usage: dict[str, int] = {}
        if planning_usage:
            _sum_usage(usage, planning_usage)
        l2_calls = planning_l2_calls
        retrieval_count = 0
        retrieval_l2_calls = 0
        mcp_calls = 0
        last_outcome: RetrievalOutcome | None = None
        draft = ""
        retrieval_closed_prompt_added = False
        generation_seconds = 0.0
        retrieval_seconds = 0.0
        review_seconds = 0.0

        if self._settings.max_generation_retrievals == 0:
            messages.insert(
                1,
                {"role": "system", "content": GENERATION_AFTER_RETRIEVAL_PROMPT},
            )
            retrieval_closed_prompt_added = True

        max_rounds = self._settings.max_generation_retrievals + 2
        for generation_round in range(1, max_rounds + 1):
            tools = (
                [RETRIEVE_RELEVANT_CONTENT_TOOL]
                if retrieval_count < self._settings.max_generation_retrievals
                else None
            )
            emit_event(
                LOGGER,
                "generation_round_started",
                request_id=request_id,
                round=generation_round,
                tools_enabled=bool(tools),
                retrievals_used=retrieval_count,
            )
            generation_started = time.monotonic()
            response = self._l2.complete(messages, deadline=deadline, tools=tools)
            generation_elapsed = time.monotonic() - generation_started
            generation_seconds += generation_elapsed
            l2_calls += 1
            _sum_usage(usage, response.usage)
            emit_event(
                LOGGER,
                "generation_round_completed",
                request_id=request_id,
                round=generation_round,
                latency_ms=round(generation_elapsed * 1000),
                tool_call_count=len(response.tool_calls),
                content_chars=len(response.content),
            )

            if not response.tool_calls:
                if response.content:
                    draft = response.content
                    break
                continue

            messages.append(response.assistant_message)
            for call in response.tool_calls:
                emit_event(
                    LOGGER,
                    "generation_tool_received",
                    request_id=request_id,
                    round=generation_round,
                    tool=call.name,
                )
                if call.name != "retrieve_relevant_content":
                    emit_event(
                        LOGGER,
                        "generation_tool_rejected",
                        request_id=request_id,
                        round=generation_round,
                        tool=call.name,
                        reason="unknown_tool",
                    )
                    messages.append(
                        _tool_error(
                            call.id,
                            "unknown_generation_tool",
                            "Only retrieve_relevant_content is available in generation.",
                        )
                    )
                    continue
                if retrieval_count >= self._settings.max_generation_retrievals:
                    emit_event(
                        LOGGER,
                        "generation_tool_rejected",
                        request_id=request_id,
                        round=generation_round,
                        tool=call.name,
                        reason="retrieval_budget_exhausted",
                    )
                    messages.append(
                        _tool_error(
                            call.id,
                            "generation_retrieval_budget_exhausted",
                            "Answer with the available conversation and evidence.",
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
                    emit_event(
                        LOGGER,
                        "generation_tool_rejected",
                        request_id=request_id,
                        round=generation_round,
                        tool=call.name,
                        reason="invalid_arguments",
                    )
                    messages.append(
                        _tool_error(call.id, "invalid_retrieval_query", str(exc))
                    )
                    continue

                retrieval_count += 1
                emit_event(
                    LOGGER,
                    "retrieval_started",
                    request_id=request_id,
                    retrieval=retrieval_count,
                    query_chars=len(query.strip()),
                )
                retrieval_started = time.monotonic()
                try:
                    retrieval_deadline = deadline.with_timeout_cap(
                        self._settings.retrieval_timeout_sec
                    )
                    run = self._retrieval.run(
                        query.strip(),
                        deadline=retrieval_deadline,
                        request_id=request_id,
                    )
                    last_outcome = run.outcome
                    retrieval_l2_calls += run.l2_calls
                    mcp_calls += run.outcome.mcp_calls
                    _sum_usage(usage, run.usage)
                    content = run.outcome.to_generation_payload(
                        max_chars=self._settings.max_evidence_chars
                    )
                    emit_event(
                        LOGGER,
                        "retrieval_completed",
                        request_id=request_id,
                        retrieval=retrieval_count,
                        status=run.outcome.status,
                        model_rounds=run.outcome.model_rounds,
                        mcp_calls=run.outcome.mcp_calls,
                        evidence_count=len(run.outcome.evidence),
                    )
                except AppError as exc:
                    emit_event(
                        LOGGER,
                        "retrieval_failed",
                        level=logging.WARNING,
                        request_id=request_id,
                        retrieval=retrieval_count,
                        error=exc.code,
                    )
                    content = json.dumps(
                        {
                            "status": "no_evidence",
                            "note": (
                                "Authoritative retrieval was unavailable within the bounded "
                                "evidence path. Answer the question from established medical "
                                "knowledge without citations, and do not mention this failure."
                            ),
                            "evidence": [],
                        },
                        ensure_ascii=False,
                    )
                finally:
                    retrieval_seconds += time.monotonic() - retrieval_started
                messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": content}
                )

            if (
                retrieval_count >= self._settings.max_generation_retrievals
                and not retrieval_closed_prompt_added
            ):
                messages.append(
                    {"role": "system", "content": GENERATION_AFTER_RETRIEVAL_PROMPT}
                )
                retrieval_closed_prompt_added = True

        if not draft:
            emit_event(
                LOGGER,
                "final_generation_fallback_started",
                request_id=request_id,
            )
            messages.append(
                {
                    "role": "system",
                    "content": GENERATION_AFTER_RETRIEVAL_PROMPT,
                }
            )
            generation_started = time.monotonic()
            response = self._l2.complete(messages, deadline=deadline, tools=None)
            generation_seconds += time.monotonic() - generation_started
            l2_calls += 1
            _sum_usage(usage, response.usage)
            if response.tool_calls or not response.content:
                raise UpstreamError(
                    "L2 did not produce final assistant content after orchestration",
                    code="missing_final_l2_content",
                )
            draft = response.content

        reviewed = False
        revised = False
        if self._should_review(compiled, last_outcome) and deadline.can_start(1.0):
            review_started = time.monotonic()
            draft, review_l2_calls, reviewed, revised, review_usage = self._review(
                compiled,
                draft,
                last_outcome,
                deadline=deadline,
            )
            l2_calls += review_l2_calls
            _sum_usage(usage, review_usage)
            review_seconds += time.monotonic() - review_started

        lane = (
            "HIGH_RISK"
            if compiled.is_high_risk
            else "GROUNDED"
            if retrieval_count
            else "DIRECT"
        )
        trace = {
            "event": "request_completed",
            "request_id": request_id,
            "history_hash": compiled.history_hash,
            "lane": lane,
            "planned_lane": lane,
            "planner_mode": planner_mode,
            "ledger_mode": "legacy",
            "contract_mode": "legacy",
            "review_mode": (
                "legacy_conditional" if self._settings.enable_high_risk_review else "disabled"
            ),
            "fallback_reason": fallback_reason,
            "representation": self._settings.conversation_representation,
            "l2_calls": l2_calls,
            "retrieval_l2_calls": retrieval_l2_calls,
            "retrievals": retrieval_count,
            "mcp_calls": mcp_calls,
            "retrieval_status": last_outcome.status if last_outcome else "not_used",
            "evidence_count": len(last_outcome.evidence) if last_outcome else 0,
            "reviewed": reviewed,
            "revised": revised,
            "response_chars": len(draft),
            "generation_latency_ms": round(generation_seconds * 1000),
            "retrieval_latency_ms": round(retrieval_seconds * 1000),
            "review_latency_ms": round(review_seconds * 1000),
            "latency_ms": round((time.monotonic() - started) * 1000),
            "usage": usage,
            "state": "completed",
        }
        emit_event(
            LOGGER,
            "request_completed",
            **{key: value for key, value in trace.items() if key != "event"},
        )
        return DriverResult(content=draft, usage=usage, trace=trace)

    def _complete_planned(
        self,
        request: ChatCompletionRequest,
        *,
        request_id: str,
    ) -> DriverResult:
        started = time.monotonic()
        deadline = (
            Deadline.unbounded()
            if self._settings.request_timeout_sec is None
            else Deadline.after(self._settings.request_timeout_sec)
        )
        compiled = compile_conversation(request.messages)
        planning = self._planner.plan(compiled, deadline=deadline)
        if planning.plan is None:
            return self._complete_legacy(
                request,
                request_id=request_id,
                planner_mode=planning.mode,
                fallback_reason=planning.fallback_reason,
                planning_l2_calls=planning.l2_calls,
                planning_usage=planning.usage,
                started_at=started,
                deadline_override=deadline,
            )

        plan = planning.plan
        ledger_mode = "requirements" if self._settings.enable_requirement_ledger else "legacy"
        contract_mode = "dual_track" if self._settings.enable_response_contract else "disabled"
        review_mode = (
            "structured"
            if self._settings.enable_structured_review
            else "legacy_conditional"
            if self._settings.enable_high_risk_review
            else "disabled"
        )
        emit_event(
            LOGGER,
            "request_started",
            request_id=request_id,
            message_count=len(request.messages),
            representation=self._settings.conversation_representation,
            planner_mode=planning.mode,
            ledger_mode=ledger_mode,
            contract_mode=contract_mode,
            review_mode=review_mode,
            planned_lane=plan.lane,
            max_retrieval_model_rounds=self._settings.max_retrieval_model_rounds,
            max_mcp_tool_calls=self._settings.max_mcp_tool_calls,
        )

        usage: dict[str, int] = {}
        if planning.usage:
            _sum_usage(usage, planning.usage)
        l2_calls = planning.l2_calls
        retrieval_l2_calls = 0
        mcp_calls = 0
        retrieval_seconds = 0.0
        generation_seconds = 0.0
        review_seconds = 0.0
        last_outcome: RetrievalOutcome | None = None
        retrieval_status = "not_used"
        fallback_reason = planning.fallback_reason

        evidence_payload = json.dumps(
            {"status": "not_used", "requirements": [], "evidence": []},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        if plan.evidence_requirements:
            emit_event(
                LOGGER,
                "retrieval_started",
                request_id=request_id,
                retrieval=1,
                requirement_count=len(plan.evidence_requirements),
            )
            retrieval_started = time.monotonic()
            try:
                retrieval_input: str | Any
                if self._settings.enable_requirement_ledger:
                    retrieval_input = plan.ledger()
                else:
                    retrieval_input = "; ".join(
                        requirement.claim_or_question
                        for requirement in plan.evidence_requirements
                    )
                run = self._retrieval.run(
                    retrieval_input,
                    deadline=deadline.with_timeout_cap(
                        self._settings.retrieval_timeout_sec
                    ),
                    request_id=request_id,
                )
                last_outcome = run.outcome
                retrieval_l2_calls = run.l2_calls
                mcp_calls = run.outcome.mcp_calls
                _sum_usage(usage, run.usage)
                evidence_payload = run.outcome.to_generation_payload(
                    max_chars=self._settings.max_evidence_chars
                )
                retrieval_status = run.outcome.status
                if run.outcome.ledger is not None:
                    plan = replace(
                        plan,
                        evidence_requirements=run.outcome.ledger.requirements,
                    )
                emit_event(
                    LOGGER,
                    "retrieval_completed",
                    request_id=request_id,
                    retrieval=1,
                    status=retrieval_status,
                    model_rounds=run.outcome.model_rounds,
                    mcp_calls=mcp_calls,
                    evidence_count=len(run.outcome.evidence),
                    unresolved_requirements=(
                        len(run.outcome.ledger.unresolved())
                        if run.outcome.ledger is not None
                        else None
                    ),
                )
            except AppError as exc:
                retrieval_status = "no_evidence"
                fallback_reason = "bounded_retrieval_failure"
                evidence_payload = json.dumps(
                    {
                        "status": "no_evidence",
                        "note": "Authoritative retrieval was unavailable.",
                        "requirements": [],
                        "evidence": [],
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                emit_event(
                    LOGGER,
                    "retrieval_failed",
                    level=logging.WARNING,
                    request_id=request_id,
                    retrieval=1,
                    error=exc.code,
                )
            finally:
                retrieval_seconds += time.monotonic() - retrieval_started

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
            {"role": "system", "content": PLANNED_GENERATION_PROMPT},
            *compiled.generation_messages(self._settings.conversation_representation),
        ]
        if self._settings.enable_response_contract:
            messages.append(
                {
                    "role": "system",
                    "content": f"RESPONSE CONTRACT:\n{plan.to_generation_payload()}",
                }
            )
        if plan.evidence_requirements or retrieval_status != "not_used":
            messages.append(
                {
                    "role": "system",
                    "content": f"ADJUDICATED EVIDENCE REPORT:\n{evidence_payload}",
                }
            )
        if plan.lane == "CLARIFY":
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Ask only the smallest decision-changing clarification identified in "
                        "the response contract. Do not provide a full answer yet."
                    ),
                }
            )

        emit_event(
            LOGGER,
            "generation_round_started",
            request_id=request_id,
            round=1,
            tools_enabled=False,
            planned_lane=plan.lane,
        )
        generation_started = time.monotonic()
        response = self._l2.complete(messages, deadline=deadline, tools=None)
        generation_elapsed = time.monotonic() - generation_started
        generation_seconds += generation_elapsed
        l2_calls += 1
        _sum_usage(usage, response.usage)
        emit_event(
            LOGGER,
            "generation_round_completed",
            request_id=request_id,
            round=1,
            latency_ms=round(generation_elapsed * 1000),
            tool_call_count=len(response.tool_calls),
            content_chars=len(response.content),
        )
        if response.tool_calls or not response.content:
            raise UpstreamError(
                "L2 did not produce planned final assistant content",
                code="missing_final_l2_content",
            )
        draft = response.content

        reviewed = False
        revised = False
        review_status = "not_used"
        review_issue_categories: list[str] = []
        if self._settings.enable_structured_review:
            deterministic_issues = deterministic_review_issues(
                draft=draft,
                plan=plan,
                evidence_payload=evidence_payload,
            )
            if self._structured_reviewer.should_review(
                plan,
                retrieval_status=retrieval_status,
                deterministic_issues=deterministic_issues,
            ) and deadline.can_start(1.0):
                review_started = time.monotonic()
                review = self._structured_reviewer.review(
                    compiled=compiled,
                    plan=plan,
                    draft=draft,
                    evidence_payload=evidence_payload,
                    deadline=deadline,
                    deterministic_issues=deterministic_issues,
                )
                l2_calls += review.l2_calls
                if review.usage:
                    _sum_usage(usage, review.usage)
                reviewed = review.l2_calls > 0
                review_status = review.status
                review_issue_categories = sorted(
                    {issue.category for issue in review.issues}
                )
                if review.material_issues and deadline.can_start(1.0):
                    revised_draft, revision_calls, revision_usage = (
                        self._structured_reviewer.revise(
                            compiled=compiled,
                            plan=plan,
                            draft=draft,
                            evidence_payload=evidence_payload,
                            issues=review.material_issues,
                            deadline=deadline,
                        )
                    )
                    l2_calls += revision_calls
                    _sum_usage(usage, revision_usage)
                    revised = revised_draft != draft
                    draft = revised_draft
                review_seconds += time.monotonic() - review_started
        elif self._should_review(compiled, last_outcome) and deadline.can_start(1.0):
            review_started = time.monotonic()
            draft, review_l2_calls, reviewed, revised, review_usage = self._review(
                compiled,
                draft,
                last_outcome,
                deadline=deadline,
            )
            l2_calls += review_l2_calls
            _sum_usage(usage, review_usage)
            review_status = "legacy_revised" if revised else "legacy_passed"
            review_seconds += time.monotonic() - review_started

        trace = {
            "event": "request_completed",
            "request_id": request_id,
            "history_hash": compiled.history_hash,
            "lane": plan.lane,
            "planned_lane": plan.lane,
            "representation": self._settings.conversation_representation,
            "planner_mode": planning.mode,
            "ledger_mode": ledger_mode,
            "contract_mode": contract_mode,
            "review_mode": review_mode,
            "fallback_reason": fallback_reason,
            "l2_calls": l2_calls,
            "retrieval_l2_calls": retrieval_l2_calls,
            "retrievals": 1 if plan.evidence_requirements else 0,
            "mcp_calls": mcp_calls,
            "retrieval_status": retrieval_status,
            "evidence_count": len(last_outcome.evidence) if last_outcome else 0,
            "requirement_count": len(plan.evidence_requirements),
            "reviewed": reviewed,
            "revised": revised,
            "review_status": review_status,
            "review_issue_categories": review_issue_categories,
            "response_chars": len(draft),
            "generation_latency_ms": round(generation_seconds * 1000),
            "retrieval_latency_ms": round(retrieval_seconds * 1000),
            "review_latency_ms": round(review_seconds * 1000),
            "latency_ms": round((time.monotonic() - started) * 1000),
            "usage": usage,
            "state": "completed",
        }
        emit_event(
            LOGGER,
            "request_completed",
            **{key: value for key, value in trace.items() if key != "event"},
        )
        return DriverResult(content=draft, usage=usage, trace=trace)

    def _should_review(
        self,
        compiled: CompiledConversation,
        outcome: RetrievalOutcome | None,
    ) -> bool:
        if not self._settings.enable_high_risk_review:
            return False
        return compiled.is_high_risk or (
            outcome is not None and outcome.status in {"partial", "no_evidence"}
        )

    def _review(
        self,
        compiled: CompiledConversation,
        draft: str,
        outcome: RetrievalOutcome | None,
        *,
        deadline: Deadline,
    ) -> tuple[str, int, bool, bool, dict[str, int]]:
        usage: dict[str, int] = {}
        evidence = (
            outcome.to_generation_payload(max_chars=self._settings.max_evidence_chars)
            if outcome
            else json.dumps({"status": "not_used", "evidence": []})
        )
        audit_input = (
            f"CONVERSATION:\n{compiled.case_packet}\n\n"
            f"EVIDENCE:\n{evidence}\n\nDRAFT:\n{draft}"
        )
        try:
            review = self._l2.complete(
                [
                    {"role": "system", "content": REVIEW_SYSTEM_PROMPT},
                    {"role": "user", "content": audit_input},
                ],
                deadline=deadline,
            )
            _sum_usage(usage, review.usage)
            decision = review.content.strip()
            if decision.upper() == "PASS" or not decision.upper().startswith("REVISE:"):
                return draft, 1, True, False, usage
            if not deadline.can_start(1.0):
                return draft, 1, True, False, usage
            revision = self._l2.complete(
                [
                    {"role": "system", "content": REVISION_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"CONVERSATION:\n{compiled.case_packet}\n\n"
                            f"EVIDENCE:\n{evidence}\n\nDRAFT:\n{draft}\n\n"
                            f"GROUNDED AUDIT ISSUES:\n{decision}"
                        ),
                    },
                ],
                deadline=deadline,
            )
            _sum_usage(usage, revision.usage)
            return revision.content, 2, True, True, usage
        except AppError:
            return draft, 0, False, False, usage
