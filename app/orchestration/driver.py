from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Mapping

from app.clients.l2 import L2Client, parse_tool_arguments
from app.config import Settings
from app.contracts import ChatCompletionRequest
from app.conversation import CompiledConversation, compile_conversation
from app.deadline import Deadline
from app.errors import AppError, UpstreamError
from app.evidence.models import RetrievalOutcome
from app.orchestration.retrieval import RetrievalEngine
from app.prompts import (
    GENERATION_AFTER_RETRIEVAL_PROMPT,
    GENERATION_SYSTEM_PROMPT,
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

    def complete(
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
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
            *compiled.generation_messages(self._settings.conversation_representation),
        ]
        usage: dict[str, int] = {}
        l2_calls = 0
        retrieval_count = 0
        retrieval_l2_calls = 0
        mcp_calls = 0
        last_outcome: RetrievalOutcome | None = None
        draft = ""
        retrieval_closed_prompt_added = False

        if self._settings.max_generation_retrievals == 0:
            messages.insert(
                1,
                {"role": "system", "content": GENERATION_AFTER_RETRIEVAL_PROMPT},
            )
            retrieval_closed_prompt_added = True

        max_rounds = self._settings.max_generation_retrievals + 2
        for _ in range(max_rounds):
            tools = (
                [RETRIEVE_RELEVANT_CONTENT_TOOL]
                if retrieval_count < self._settings.max_generation_retrievals
                else None
            )
            response = self._l2.complete(messages, deadline=deadline, tools=tools)
            l2_calls += 1
            _sum_usage(usage, response.usage)

            if not response.tool_calls:
                if response.content:
                    draft = response.content
                    break
                continue

            messages.append(response.assistant_message)
            for call in response.tool_calls:
                if call.name != "retrieve_relevant_content":
                    messages.append(
                        _tool_error(
                            call.id,
                            "unknown_generation_tool",
                            "Only retrieve_relevant_content is available in generation.",
                        )
                    )
                    continue
                if retrieval_count >= self._settings.max_generation_retrievals:
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
                    messages.append(
                        _tool_error(call.id, "invalid_retrieval_query", str(exc))
                    )
                    continue

                retrieval_count += 1
                try:
                    run = self._retrieval.run(query.strip(), deadline=deadline)
                    last_outcome = run.outcome
                    retrieval_l2_calls += run.l2_calls
                    mcp_calls += run.outcome.mcp_calls
                    _sum_usage(usage, run.usage)
                    content = run.outcome.to_generation_payload(
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
                    content = json.dumps(
                        {
                            "status": "no_evidence",
                            "note": (
                                "Authoritative retrieval was unavailable within the bounded "
                                "evidence path. Answer conservatively without unsupported claims."
                            ),
                            "evidence": [],
                        },
                        ensure_ascii=False,
                    )
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
            messages.append(
                {
                    "role": "system",
                    "content": GENERATION_AFTER_RETRIEVAL_PROMPT,
                }
            )
            response = self._l2.complete(messages, deadline=deadline, tools=None)
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
            draft, review_l2_calls, reviewed, revised, review_usage = self._review(
                compiled,
                draft,
                last_outcome,
                deadline=deadline,
            )
            l2_calls += review_l2_calls
            _sum_usage(usage, review_usage)

        lane = "HIGH_RISK" if compiled.is_high_risk else "GROUNDED" if retrieval_count else "DIRECT"
        trace = {
            "request_id": request_id,
            "history_hash": compiled.history_hash,
            "lane": lane,
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
            "latency_ms": round((time.monotonic() - started) * 1000),
            "state": "completed",
        }
        LOGGER.info(json.dumps(trace, ensure_ascii=False, separators=(",", ":")))
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
