from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from app.clients.l2 import L2Client, parse_tool_arguments
from app.config import Settings
from app.conversation import CompiledConversation
from app.deadline import Deadline
from app.errors import AppError
from app.orchestration.planning import ResponsePlan
from app.prompts import REVISION_SYSTEM_PROMPT, STRUCTURED_REVIEW_SYSTEM_PROMPT

ISSUE_CATEGORIES = {
    "conversation_conflict",
    "missing_clinical_obligation",
    "unsafe_or_disproportionate_action",
    "unsupported_claim",
    "citation_mismatch",
    "uncertainty_error",
    "instruction_or_format_miss",
    "clarity_or_language_miss",
}
MATERIAL_SEVERITIES = {"material", "critical"}
CITE_TOKEN_PATTERN = re.compile(r"\bcite[_-][A-Za-z0-9_-]+\b")
HANGUL_PATTERN = re.compile(r"[가-힣]")


SUBMIT_REVIEW_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "submit_review",
        "description": "Submit the bounded answer-level audit decision.",
        "parameters": {
            "type": "object",
            "properties": {
                "verdict": {"type": "string", "enum": ["PASS", "REVISE"]},
                "issues": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "category": {"type": "string", "enum": sorted(ISSUE_CATEGORIES)},
                            "severity": {
                                "type": "string",
                                "enum": ["minor", "material", "critical"],
                            },
                            "target_id": {"type": "string"},
                            "instruction": {"type": "string"},
                        },
                        "required": ["category", "severity", "target_id", "instruction"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["verdict", "issues"],
            "additionalProperties": False,
        },
    },
}


@dataclass(frozen=True, slots=True)
class ReviewIssue:
    category: str
    severity: str
    target_id: str
    instruction: str


@dataclass(frozen=True, slots=True)
class ReviewResult:
    status: str
    passed: bool
    issues: tuple[ReviewIssue, ...] = ()
    error_code: str = ""
    l2_calls: int = 0
    usage: dict[str, int] | None = None

    @property
    def material_issues(self) -> tuple[ReviewIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity in MATERIAL_SEVERITIES)


def deterministic_review_issues(
    *, draft: str, plan: ResponsePlan, evidence_payload: str
) -> tuple[ReviewIssue, ...]:
    try:
        evidence = json.loads(evidence_payload or "{}")
    except json.JSONDecodeError:
        evidence = {}
    allowed: set[str] = set()

    def collect(value: Any) -> None:
        if isinstance(value, dict):
            cite_uid = value.get("cite_uid")
            if isinstance(cite_uid, str):
                allowed.add(cite_uid)
            for nested in value.values():
                collect(nested)
        elif isinstance(value, list):
            for nested in value:
                collect(nested)

    collect(evidence)
    issues: list[ReviewIssue] = []
    unknown = sorted(set(CITE_TOKEN_PATTERN.findall(draft)) - allowed)
    if unknown:
        issues.append(
            ReviewIssue(
                category="citation_mismatch",
                severity="material",
                target_id="citation",
                instruction="Remove or replace citation identifiers not present in adjudicated evidence.",
            )
        )
    requested = plan.interaction.requested_format.lower()
    if plan.interaction.strict_format and "json" in requested:
        try:
            json.loads(draft)
        except json.JSONDecodeError:
            issues.append(
                ReviewIssue(
                    category="instruction_or_format_miss",
                    severity="material",
                    target_id="requested_format",
                    instruction="Return valid JSON in the requested shape without surrounding prose.",
                )
            )
    if plan.interaction.language.lower().startswith("ko") and not HANGUL_PATTERN.search(draft):
        issues.append(
            ReviewIssue(
                category="clarity_or_language_miss",
                severity="material",
                target_id="language",
                instruction="Return the answer in Korean as requested by the conversation.",
            )
        )
    return tuple(issues)


def _parse_review(arguments: dict[str, Any]) -> ReviewResult:
    verdict = arguments.get("verdict")
    if verdict not in {"PASS", "REVISE"}:
        raise ValueError("invalid review verdict")
    raw_issues = arguments.get("issues")
    if not isinstance(raw_issues, list) or len(raw_issues) > 8:
        raise ValueError("review issues must be a bounded array")
    issues: list[ReviewIssue] = []
    for raw in raw_issues:
        if not isinstance(raw, dict):
            raise TypeError("review issue must be an object")
        category = raw.get("category")
        severity = raw.get("severity")
        target_id = raw.get("target_id")
        instruction = raw.get("instruction")
        if category not in ISSUE_CATEGORIES:
            raise ValueError("invalid review issue category")
        if severity not in {"minor", "material", "critical"}:
            raise ValueError("invalid review issue severity")
        if not isinstance(target_id, str) or not isinstance(instruction, str):
            raise TypeError("review target and instruction must be strings")
        instruction = instruction.strip()[:800]
        if not instruction:
            raise ValueError("review instruction cannot be empty")
        issues.append(
            ReviewIssue(
                category=category,
                severity=severity,
                target_id=target_id.strip()[:100],
                instruction=instruction,
            )
        )
    if verdict == "PASS" and issues:
        raise ValueError("PASS review cannot contain issues")
    if verdict == "REVISE" and not any(
        issue.severity in MATERIAL_SEVERITIES for issue in issues
    ):
        raise ValueError("REVISE review needs a material issue")
    return ReviewResult(
        status="passed" if verdict == "PASS" else "issues_found",
        passed=verdict == "PASS",
        issues=tuple(issues),
        l2_calls=1,
        usage={},
    )


class StructuredReviewer:
    def __init__(self, settings: Settings, *, l2: L2Client) -> None:
        self._settings = settings
        self._l2 = l2

    def should_review(
        self,
        plan: ResponsePlan,
        *,
        retrieval_status: str,
        deterministic_issues: Sequence[ReviewIssue],
    ) -> bool:
        del retrieval_status  # Retrieval state alone is intentionally not a trigger.
        if deterministic_issues:
            return True
        if plan.requires_review or plan.interaction.strict_format:
            return True
        if any(fact.corrected for fact in plan.clinical_facts):
            return True
        if plan.interaction.unresolved_references:
            return True
        return any(
            requirement.criticality == "critical"
            and requirement.status in {"missing", "contradicted", "conflicted"}
            for requirement in plan.evidence_requirements
        )

    def review(
        self,
        *,
        compiled: CompiledConversation,
        plan: ResponsePlan,
        draft: str,
        evidence_payload: str,
        deadline: Deadline,
        deterministic_issues: Sequence[ReviewIssue] = (),
    ) -> ReviewResult:
        review_input = json.dumps(
            {
                "conversation": compiled.case_packet,
                "response_contract": json.loads(plan.to_generation_payload()),
                "evidence": json.loads(evidence_payload or "{}"),
                "draft": draft,
                "deterministic_issues": [
                    {
                        "category": issue.category,
                        "severity": issue.severity,
                        "target_id": issue.target_id,
                        "instruction": issue.instruction,
                    }
                    for issue in deterministic_issues
                ],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        try:
            response = self._l2.complete(
                [
                    {"role": "system", "content": STRUCTURED_REVIEW_SYSTEM_PROMPT},
                    {"role": "user", "content": review_input},
                ],
                deadline=deadline,
                tools=[SUBMIT_REVIEW_TOOL],
                tool_choice={
                    "type": "function",
                    "function": {"name": "submit_review"},
                },
            )
        except AppError as exc:
            return ReviewResult(
                status="review_unavailable",
                passed=False,
                error_code=exc.code,
                l2_calls=1,
            )
        if len(response.tool_calls) != 1 or response.tool_calls[0].name != "submit_review":
            return ReviewResult(
                status="review_unavailable",
                passed=False,
                error_code="missing_structured_call",
                l2_calls=1,
                usage=dict(response.usage),
            )
        try:
            arguments = parse_tool_arguments(response.tool_calls[0].arguments)
            parsed = _parse_review(arguments)
            return ReviewResult(
                status=parsed.status,
                passed=parsed.passed,
                issues=parsed.issues,
                l2_calls=1,
                usage=dict(response.usage),
            )
        except json.JSONDecodeError:
            error_code = "invalid_review_json"
        except TypeError:
            error_code = "invalid_review_types"
        except ValueError:
            error_code = "invalid_review_values"
        return ReviewResult(
            status="review_unavailable",
            passed=False,
            error_code=error_code,
            l2_calls=1,
            usage=dict(response.usage),
        )

    def revise(
        self,
        *,
        compiled: CompiledConversation,
        plan: ResponsePlan,
        draft: str,
        evidence_payload: str,
        issues: Sequence[ReviewIssue],
        deadline: Deadline,
    ) -> tuple[str, int, dict[str, int]]:
        material = [issue for issue in issues if issue.severity in MATERIAL_SEVERITIES]
        if not material:
            return draft, 0, {}
        revision_input = json.dumps(
            {
                "conversation": compiled.case_packet,
                "response_contract": json.loads(plan.to_generation_payload()),
                "evidence": json.loads(evidence_payload or "{}"),
                "draft": draft,
                "issues": [
                    {
                        "category": issue.category,
                        "target_id": issue.target_id,
                        "instruction": issue.instruction,
                    }
                    for issue in material
                ],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        try:
            response = self._l2.complete(
                [
                    {"role": "system", "content": REVISION_SYSTEM_PROMPT},
                    {"role": "user", "content": revision_input},
                ],
                deadline=deadline,
            )
        except AppError:
            return draft, 0, {}
        if response.tool_calls or not response.content:
            return draft, 1, dict(response.usage)
        return response.content, 1, dict(response.usage)
