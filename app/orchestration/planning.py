from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from app.clients.l2 import L2Client, parse_tool_arguments
from app.conversation import CompiledConversation
from app.deadline import Deadline
from app.errors import AppError
from app.evidence.models import EvidenceRequirement, EvidenceRequirementLedger
from app.prompts import PLANNING_SYSTEM_PROMPT

VALID_RESPONSE_MODES = {"ANSWER", "CLARIFY"}
VALID_RISK_LEVELS = {"ROUTINE", "HIGH"}

_SOURCE_FAMILY_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"가이드라인|진료지침|guideline|recommendation", re.IGNORECASE), "guideline"),
    (re.compile(r"급여|보험|reimbursement|coverage", re.IGNORECASE), "reimbursement"),
    (re.compile(r"법령|법률|statute", re.IGNORECASE), "law"),
    (re.compile(r"질병코드|상병코드|kcd", re.IGNORECASE), "coding"),
    (re.compile(r"허가|적응증|approval|indication", re.IGNORECASE), "approval"),
)


class PlanValidationError(ValueError):
    """Privacy-safe planner contract failure with a stable structural code."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _closed_object(
    properties: dict[str, Any], required: list[str]
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


SOURCE_TURNS_SCHEMA = {
    "type": "array",
    "minItems": 1,
    "items": {"type": "integer", "minimum": 1},
}

SUBMIT_RESPONSE_PLAN_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "submit_response_plan",
        "description": (
            "Submit exactly one response plan. This is the only valid planner output; "
            "do not answer the user in prose."
        ),
        "parameters": _closed_object(
            {
                "response_mode": {
                    "type": "string",
                    "enum": sorted(VALID_RESPONSE_MODES),
                    "description": (
                        "ANSWER unless one decision-changing clarification must come first."
                    ),
                },
                "risk_level": {
                    "type": "string",
                    "enum": sorted(VALID_RISK_LEVELS),
                    "description": (
                        "HIGH when answer-level review is warranted. This does not replace "
                        "evidence requirements; a HIGH plan may also require retrieval."
                    ),
                },
                "clinical_facts": {
                    "type": "array",
                    "maxItems": 16,
                    "items": _closed_object(
                        {
                            "kind": {"type": "string", "minLength": 1},
                            "value": {"type": "string", "minLength": 1},
                            "source_turns": SOURCE_TURNS_SCHEMA,
                            "negated": {"type": "boolean"},
                            "corrected": {"type": "boolean"},
                        },
                        ["kind", "value", "source_turns", "negated", "corrected"],
                    ),
                },
                "interaction": _closed_object(
                    {
                        "intent": {"type": "string", "minLength": 1},
                        "language": {"type": "string", "minLength": 1},
                        "requested_format": {"type": "string"},
                        "unresolved_references": {
                            "type": "array",
                            "maxItems": 8,
                            "items": {"type": "string"},
                        },
                        "strict_format": {"type": "boolean"},
                    },
                    [
                        "intent",
                        "language",
                        "requested_format",
                        "unresolved_references",
                        "strict_format",
                    ],
                ),
                "risk_signals": {
                    "type": "array",
                    "maxItems": 12,
                    "items": _closed_object(
                        {
                            "category": {"type": "string", "minLength": 1},
                            "source_turns": SOURCE_TURNS_SCHEMA,
                            "materiality": {"type": "string", "minLength": 1},
                        },
                        ["category", "source_turns", "materiality"],
                    ),
                },
                "missing_information": {
                    "type": "array",
                    "maxItems": 8,
                    "items": _closed_object(
                        {
                            "id": {"type": "string", "minLength": 1},
                            "question": {"type": "string", "minLength": 1},
                            "material": {"type": "boolean"},
                        },
                        ["id", "question", "material"],
                    ),
                },
                "evidence_requirements": {
                    "type": "array",
                    "maxItems": 4,
                    "description": (
                        "Atomic authoritative questions needed before answering. Include these "
                        "independently of risk_level."
                    ),
                    "items": _closed_object(
                        {
                            "id": {"type": "string", "minLength": 1},
                            "question": {"type": "string", "minLength": 1},
                            "source_family": {"type": "string", "minLength": 1},
                            "jurisdiction": {"type": "string"},
                            "criticality": {
                                "type": "string",
                                "enum": ["critical", "supporting"],
                            },
                        },
                        [
                            "id",
                            "question",
                            "source_family",
                            "jurisdiction",
                            "criticality",
                        ],
                    ),
                },
                "answer_obligations": {
                    "type": "array",
                    "maxItems": 12,
                    "items": {"type": "string", "minLength": 1},
                },
            },
            [
                "response_mode",
                "risk_level",
                "clinical_facts",
                "interaction",
                "risk_signals",
                "missing_information",
                "evidence_requirements",
                "answer_obligations",
            ],
        ),
    },
}

FORCED_PLAN_TOOL_CHOICE = {
    "type": "function",
    "function": {"name": "submit_response_plan"},
}


def _text(
    value: Any,
    *,
    code: str,
    maximum: int = 1_000,
    allow_empty: bool = False,
) -> str:
    if not isinstance(value, str):
        raise PlanValidationError(code)
    normalized = value.strip()[:maximum]
    if not normalized and not allow_empty:
        raise PlanValidationError(code)
    return normalized


def _bounded_array(value: Any, *, code: str, maximum: int) -> list[Any]:
    if not isinstance(value, list) or len(value) > maximum:
        raise PlanValidationError(code)
    return value


def _strict_bool(value: Any, *, code: str) -> bool:
    if not isinstance(value, bool):
        raise PlanValidationError(code)
    return value


def _source_turns(value: Any, *, compiled: CompiledConversation) -> tuple[int, ...]:
    turns = _bounded_array(value, code="invalid_source_turns", maximum=32)
    if not turns:
        raise PlanValidationError("invalid_source_turns")
    allowed = set(compiled.user_turn_ids)
    normalized: list[int] = []
    for raw in turns:
        if not isinstance(raw, int) or isinstance(raw, bool) or raw not in allowed:
            raise PlanValidationError("invalid_source_turns")
        normalized.append(raw)
    return tuple(dict.fromkeys(normalized))


@dataclass(frozen=True, slots=True)
class ClinicalFact:
    kind: str
    value: str
    source_turns: tuple[int, ...]
    negated: bool = False
    corrected: bool = False


@dataclass(frozen=True, slots=True)
class InteractionState:
    intent: str
    language: str
    requested_format: str = ""
    unresolved_references: tuple[str, ...] = ()
    strict_format: bool = False


@dataclass(frozen=True, slots=True)
class RiskSignal:
    category: str
    source_turns: tuple[int, ...]
    materiality: str = "material"


@dataclass(frozen=True, slots=True)
class MissingInformation:
    id: str
    question: str
    material: bool = True


@dataclass(frozen=True, slots=True)
class ResponsePlan:
    response_mode: str
    risk_level: str
    clinical_facts: tuple[ClinicalFact, ...]
    interaction: InteractionState
    risk_signals: tuple[RiskSignal, ...]
    missing_information: tuple[MissingInformation, ...]
    evidence_requirements: tuple[EvidenceRequirement, ...]
    answer_obligations: tuple[str, ...]

    @property
    def lane(self) -> str:
        """Compatibility/trace label derived from independent control decisions."""
        if self.response_mode == "CLARIFY":
            return "CLARIFY"
        if self.risk_level == "HIGH":
            return "HIGH_RISK"
        if self.evidence_requirements:
            return "GROUNDED"
        return "DIRECT"

    @property
    def requires_retrieval(self) -> bool:
        return bool(self.evidence_requirements)

    @property
    def requires_review(self) -> bool:
        return self.risk_level == "HIGH"

    @classmethod
    def empty(cls, *, lane: str, intent: str, language: str) -> ResponsePlan:
        if lane not in {"DIRECT", "CLARIFY", "HIGH_RISK"}:
            raise ValueError("empty response plan supports direct, clarify, or high risk")
        return cls(
            response_mode="CLARIFY" if lane == "CLARIFY" else "ANSWER",
            risk_level="HIGH" if lane == "HIGH_RISK" else "ROUTINE",
            clinical_facts=(),
            interaction=InteractionState(intent=intent, language=language),
            risk_signals=(),
            missing_information=(),
            evidence_requirements=(),
            answer_obligations=(),
        )

    def ledger(self) -> EvidenceRequirementLedger:
        return EvidenceRequirementLedger(requirements=self.evidence_requirements)

    def to_generation_payload(self) -> str:
        payload = {
            "response_mode": self.response_mode,
            "risk_level": self.risk_level,
            "derived_lane": self.lane,
            "clinical_facts": [
                {
                    "kind": item.kind,
                    "value": item.value,
                    "source_turns": list(item.source_turns),
                    "negated": item.negated,
                    "corrected": item.corrected,
                }
                for item in self.clinical_facts
            ],
            "interaction": {
                "intent": self.interaction.intent,
                "language": self.interaction.language,
                "requested_format": self.interaction.requested_format,
                "unresolved_references": list(self.interaction.unresolved_references),
                "strict_format": self.interaction.strict_format,
            },
            "risk_signals": [
                {
                    "category": item.category,
                    "source_turns": list(item.source_turns),
                    "materiality": item.materiality,
                }
                for item in self.risk_signals
            ],
            "missing_information": [
                {"id": item.id, "question": item.question, "material": item.material}
                for item in self.missing_information
            ],
            "evidence_requirements": [
                {
                    "id": item.id,
                    "question": item.claim_or_question,
                    "source_family": item.source_family,
                    "jurisdiction": item.jurisdiction,
                    "criticality": item.criticality,
                }
                for item in self.evidence_requirements
            ],
            "answer_obligations": list(self.answer_obligations),
        }
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class PlanningResult:
    plan: ResponsePlan | None
    mode: str
    fallback_reason: str = ""
    error_code: str = ""
    l2_calls: int = 0
    usage: dict[str, int] | None = None


def _mapping(value: Any, *, code: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PlanValidationError(code)
    return value


def parse_response_plan(
    arguments: Mapping[str, Any], *, compiled: CompiledConversation
) -> ResponsePlan:
    response_mode = arguments.get("response_mode")
    if response_mode not in VALID_RESPONSE_MODES:
        raise PlanValidationError("invalid_response_mode")
    risk_level = arguments.get("risk_level")
    if risk_level not in VALID_RISK_LEVELS:
        raise PlanValidationError("invalid_risk_level")

    facts: list[ClinicalFact] = []
    for raw_value in _bounded_array(
        arguments.get("clinical_facts"), code="invalid_clinical_facts", maximum=16
    ):
        raw = _mapping(raw_value, code="invalid_clinical_fact")
        facts.append(
            ClinicalFact(
                kind=_text(raw.get("kind"), code="invalid_clinical_fact_kind", maximum=80),
                value=_text(raw.get("value"), code="invalid_clinical_fact_value"),
                source_turns=_source_turns(raw.get("source_turns"), compiled=compiled),
                negated=_strict_bool(raw.get("negated"), code="invalid_clinical_fact_negated"),
                corrected=_strict_bool(
                    raw.get("corrected"), code="invalid_clinical_fact_corrected"
                ),
            )
        )

    raw_interaction = _mapping(arguments.get("interaction"), code="invalid_interaction")
    raw_refs = _bounded_array(
        raw_interaction.get("unresolved_references", []),
        code="invalid_unresolved_references",
        maximum=8,
    )
    if not all(isinstance(item, str) for item in raw_refs):
        raise PlanValidationError("invalid_unresolved_references")
    interaction = InteractionState(
        intent=_text(raw_interaction.get("intent"), code="invalid_interaction_intent"),
        language=_text(
            raw_interaction.get("language", ""),
            code="invalid_interaction_language",
            maximum=20,
        ),
        requested_format=_text(
            raw_interaction.get("requested_format", ""),
            code="invalid_requested_format",
            allow_empty=True,
        ),
        unresolved_references=tuple(item.strip()[:200] for item in raw_refs if item.strip()),
        strict_format=_strict_bool(
            raw_interaction.get("strict_format"), code="invalid_strict_format"
        ),
    )

    risks: list[RiskSignal] = []
    for raw_value in _bounded_array(
        arguments.get("risk_signals"), code="invalid_risk_signals", maximum=12
    ):
        raw = _mapping(raw_value, code="invalid_risk_signal")
        risks.append(
            RiskSignal(
                category=_text(raw.get("category"), code="invalid_risk_category", maximum=80),
                source_turns=_source_turns(raw.get("source_turns"), compiled=compiled),
                materiality=_text(
                    raw.get("materiality", "material"),
                    code="invalid_risk_materiality",
                    maximum=40,
                ),
            )
        )

    missing: list[MissingInformation] = []
    for raw_value in _bounded_array(
        arguments.get("missing_information"),
        code="invalid_missing_information",
        maximum=8,
    ):
        raw = _mapping(raw_value, code="invalid_missing_information_item")
        missing.append(
            MissingInformation(
                id=_text(raw.get("id"), code="invalid_missing_id", maximum=80),
                question=_text(raw.get("question"), code="invalid_missing_question"),
                material=_strict_bool(raw.get("material"), code="invalid_missing_material"),
            )
        )

    requirements: list[EvidenceRequirement] = []
    for raw_value in _bounded_array(
        arguments.get("evidence_requirements"),
        code="invalid_evidence_requirements",
        maximum=4,
    ):
        raw = _mapping(raw_value, code="invalid_evidence_requirement")
        requirements.append(
            EvidenceRequirement(
                id=_text(raw.get("id"), code="invalid_requirement_id", maximum=80),
                claim_or_question=_text(
                    raw.get("question"), code="invalid_requirement_question"
                ),
                source_family=_text(
                    raw.get("source_family"),
                    code="invalid_requirement_source_family",
                    maximum=80,
                ),
                jurisdiction=_text(
                    raw.get("jurisdiction"),
                    code="invalid_requirement_jurisdiction",
                    maximum=80,
                    allow_empty=True,
                ),
                criticality=_text(
                    raw.get("criticality"),
                    code="invalid_requirement_criticality",
                    maximum=20,
                ),
            )
        )
    if requirements:
        try:
            EvidenceRequirementLedger(tuple(requirements))
        except (TypeError, ValueError) as exc:
            raise PlanValidationError("invalid_requirement_ledger") from exc

    raw_obligations = _bounded_array(
        arguments.get("answer_obligations"),
        code="invalid_answer_obligations",
        maximum=12,
    )
    if not all(isinstance(item, str) and item.strip() for item in raw_obligations):
        raise PlanValidationError("invalid_answer_obligations")

    if response_mode == "CLARIFY" and requirements:
        raise PlanValidationError("clarify_plan_has_requirements")
    if response_mode == "ANSWER" and compiled.is_source_sensitive and not requirements:
        raise PlanValidationError("source_sensitive_without_requirement")
    if compiled.full_history_high_risk:
        risk_level = "HIGH"

    return ResponsePlan(
        response_mode=response_mode,
        risk_level=risk_level,
        clinical_facts=tuple(facts),
        interaction=interaction,
        risk_signals=tuple(risks),
        missing_information=tuple(missing),
        evidence_requirements=tuple(requirements),
        answer_obligations=tuple(
            item.strip()[:1_000] for item in raw_obligations if item.strip()
        ),
    )


def needs_structured_planning(compiled: CompiledConversation) -> bool:
    """Planning is unconditional while the semantic controller is under validation."""
    del compiled
    return True


def _sum_usage(total: dict[str, int], usage: Mapping[str, int]) -> None:
    for key, value in usage.items():
        if isinstance(value, int):
            total[key] = total.get(key, 0) + value


def _error_code(exc: Exception) -> str:
    if isinstance(exc, PlanValidationError):
        return exc.code
    if isinstance(exc, json.JSONDecodeError):
        return "invalid_tool_json"
    if isinstance(exc, TypeError):
        return "invalid_tool_types"
    if isinstance(exc, ValueError):
        return "invalid_tool_values"
    if isinstance(exc, AppError):
        return exc.code
    return "unknown"


def _bounded_source_requirement(
    compiled: CompiledConversation,
) -> dict[str, Any]:
    """Create the one safe field-level repair allowed for source-sensitive answers."""
    text = compiled.latest_user_text.strip()[:1_000]
    source_family = next(
        (
            family
            for pattern, family in _SOURCE_FAMILY_PATTERNS
            if pattern.search(text)
        ),
        "research",
    )
    jurisdiction = "KR" if re.search(r"[가-힣]|kcd", text, re.IGNORECASE) else ""
    return {
        "id": "source_requirement_1",
        "question": text,
        "source_family": source_family,
        "jurisdiction": jurisdiction,
        "criticality": "critical",
    }


class StructuredPlanner:
    def __init__(self, *, l2: L2Client, retry_reserve_sec: float = 1.0) -> None:
        self._l2 = l2
        self._retry_reserve_sec = retry_reserve_sec

    def plan(self, compiled: CompiledConversation, *, deadline: Deadline) -> PlanningResult:
        usage: dict[str, int] = {}
        last_code = ""
        calls = 0
        for attempt in range(2):
            if attempt and not deadline.can_start(self._retry_reserve_sec):
                break
            messages = [
                {"role": "system", "content": PLANNING_SYSTEM_PROMPT},
                {"role": "user", "content": compiled.case_packet},
            ]
            if attempt:
                messages.insert(
                    1,
                    {
                        "role": "system",
                        "content": (
                            "The previous plan failed structural validation with code "
                            f"{last_code}. Submit a new complete plan matching the tool schema."
                        ),
                    },
                )
            calls += 1
            try:
                response = self._l2.complete(
                    messages,
                    deadline=deadline,
                    tools=[SUBMIT_RESPONSE_PLAN_TOOL],
                    tool_choice=FORCED_PLAN_TOOL_CHOICE,
                )
                _sum_usage(usage, response.usage)
                if (
                    len(response.tool_calls) != 1
                    or response.tool_calls[0].name != "submit_response_plan"
                ):
                    raise PlanValidationError("missing_structured_call")
                arguments = parse_tool_arguments(response.tool_calls[0].arguments)
                try:
                    plan = parse_response_plan(arguments, compiled=compiled)
                except PlanValidationError as exc:
                    if exc.code != "source_sensitive_without_requirement":
                        raise
                    repaired_arguments = dict(arguments)
                    repaired_arguments["evidence_requirements"] = [
                        _bounded_source_requirement(compiled)
                    ]
                    plan = parse_response_plan(repaired_arguments, compiled=compiled)
                    return PlanningResult(
                        plan=plan,
                        mode="structured_bounded_repair",
                        error_code=exc.code,
                        l2_calls=calls,
                        usage=usage,
                    )
                return PlanningResult(
                    plan=plan,
                    mode="structured" if attempt == 0 else "structured_repaired",
                    l2_calls=calls,
                    usage=usage,
                )
            except AppError as exc:
                last_code = _error_code(exc)
                break
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                last_code = _error_code(exc)

        return PlanningResult(
            plan=None,
            mode="legacy_fallback",
            fallback_reason=f"planner_{last_code or 'repair_deadline'}",
            error_code=last_code or "repair_deadline",
            l2_calls=calls,
            usage=usage,
        )
