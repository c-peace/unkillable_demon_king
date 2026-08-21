from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping, Sequence

from app.clients.l2 import L2Client, parse_tool_arguments
from app.conversation import CompiledConversation
from app.deadline import Deadline
from app.errors import AppError
from app.prompts import REVISION_SYSTEM_PROMPT, review_system_prompt
from app.response_contracts import ResponseContract


class ReviewReason(StrEnum):
    ACTIVE_HIGH_RISK = "active_high_risk"
    PARTIAL_CRITICAL_EVIDENCE = "partial_critical_evidence"
    CONFLICTING_EVIDENCE = "conflicting_evidence"
    HISTORY_COMPACTED = "history_compacted"
    STRICT_MULTIPART_OUTPUT = "strict_multipart_output"
    UNCERTAIN_APPLICABILITY = "uncertain_applicability"


REVIEW_DECISION_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "review_response",
        "description": "Submit a bounded defect review of the draft.",
        "parameters": {
            "type": "object",
            "properties": {
                "decision": {"type": "string", "enum": ["pass", "revise"]},
                "issues": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "category": {"type": "string"},
                            "requirement_id": {"type": "string"},
                            "concrete_defect": {"type": "string"},
                        },
                        "required": ["category", "concrete_defect"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["decision", "issues"],
            "additionalProperties": False,
        },
    },
}


@dataclass(frozen=True, slots=True)
class ReviewRun:
    draft: str
    l2_calls: int
    reviewed: bool
    revised: bool
    state: str
    usage: dict[str, int]


def _sum_usage(total: dict[str, int], usage: Mapping[str, int]) -> None:
    for key, value in usage.items():
        if isinstance(value, int) and value >= 0:
            total[key] = total.get(key, 0) + value


def _parse_decision(response: Any) -> tuple[str, list[dict[str, str]]] | None:
    for call in response.tool_calls or ():
        if call.name != "review_response":
            continue
        try:
            arguments = parse_tool_arguments(call.arguments)
        except (ValueError, json.JSONDecodeError):
            return None
        decision = arguments.get("decision")
        raw_issues = arguments.get("issues")
        if decision not in {"pass", "revise"} or not isinstance(raw_issues, list):
            return None
        issues: list[dict[str, str]] = []
        for raw in raw_issues:
            if not isinstance(raw, Mapping):
                continue
            category = raw.get("category")
            defect = raw.get("concrete_defect")
            requirement_id = raw.get("requirement_id", "")
            if isinstance(category, str) and isinstance(defect, str) and defect.strip():
                issues.append(
                    {
                        "category": category.strip()[:80],
                        "requirement_id": (
                            requirement_id.strip()[:80]
                            if isinstance(requirement_id, str)
                            else ""
                        ),
                        "concrete_defect": defect.strip()[:1_000],
                    }
                )
        if decision == "revise" and not issues:
            return None
        return decision, issues

    # Backward-compatible parser for the earlier free-text reviewer contract.
    content = response.content.strip()
    if content.upper() == "PASS":
        return "pass", []
    if content.upper().startswith("REVISE:") and content[7:].strip():
        return "revise", [
            {
                "category": "legacy_review",
                "requirement_id": "",
                "concrete_defect": content[7:].strip()[:1_000],
            }
        ]
    return None


class ConditionalReviewer:
    def __init__(self, l2: L2Client) -> None:
        self._l2 = l2

    def run(
        self,
        compiled: CompiledConversation,
        draft: str,
        evidence: str,
        reasons: Sequence[str],
        *,
        deadline: Deadline,
        contract: ResponseContract | None = None,
    ) -> ReviewRun:
        usage: dict[str, int] = {}
        reason_text = ",".join(dict.fromkeys(str(reason) for reason in reasons))
        audit_input = (
            f"REVIEW REASONS:\n{reason_text}\n\n"
            f"CONVERSATION:\n{compiled.case_packet}\n\n"
            f"EVIDENCE:\n{evidence}\n\nDRAFT:\n{draft}"
        )
        try:
            review = self._l2.complete(
                [
                    {"role": "system", "content": review_system_prompt(contract)},
                    {"role": "user", "content": audit_input},
                ],
                deadline=deadline,
                tools=[REVIEW_DECISION_TOOL],
                tool_choice="required",
            )
            _sum_usage(usage, review.usage)
            parsed = _parse_decision(review)
            if parsed is None:
                return ReviewRun(draft, 1, False, False, "review_unavailable", usage)
            decision, issues = parsed
            if contract is not None and issues:
                allowed = set(contract.reviewer_categories)
                issues = [
                    issue
                    for issue in issues
                    if issue["category"] in allowed or issue["category"] == "legacy_review"
                ]
                if decision == "revise" and not issues:
                    return ReviewRun(draft, 1, False, False, "review_unavailable", usage)
            if decision == "pass":
                return ReviewRun(draft, 1, True, False, "passed", usage)
            if not deadline.can_start(1.0):
                return ReviewRun(draft, 1, True, False, "revision_skipped", usage)
            revision = self._l2.complete(
                [
                    {"role": "system", "content": REVISION_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"CONVERSATION:\n{compiled.case_packet}\n\n"
                            f"EVIDENCE:\n{evidence}\n\nDRAFT:\n{draft}\n\n"
                            "GROUNDED AUDIT ISSUES:\n"
                            + json.dumps(issues, ensure_ascii=False)
                        ),
                    },
                ],
                deadline=deadline,
            )
            _sum_usage(usage, revision.usage)
            if not revision.content:
                return ReviewRun(draft, 2, True, False, "revision_unavailable", usage)
            return ReviewRun(revision.content, 2, True, True, "revised", usage)
        except AppError:
            return ReviewRun(draft, 0, False, False, "review_unavailable", usage)
