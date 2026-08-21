from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from app.clients.l2 import L2Client, parse_tool_arguments
from app.conversation import CompiledConversation
from app.deadline import Deadline
from app.errors import AppError
from app.evidence.models import EvidenceRequirement, EvidenceRequirementLedger
from app.prompts import PLANNING_SYSTEM_PROMPT

VALID_LANES = {"DIRECT", "CLARIFY", "GROUNDED", "HIGH_RISK"}


SUBMIT_RESPONSE_PLAN_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "submit_response_plan",
        "description": "Submit the grounded execution plan for the final response.",
        "parameters": {
            "type": "object",
            "properties": {
                "lane": {"type": "string", "enum": sorted(VALID_LANES)},
                "clinical_facts": {"type": "array", "items": {"type": "object"}},
                "interaction": {"type": "object"},
                "risk_signals": {"type": "array", "items": {"type": "object"}},
                "missing_information": {"type": "array", "items": {"type": "object"}},
                "evidence_requirements": {
                    "type": "array",
                    "maxItems": 4,
                    "items": {"type": "object"},
                },
                "answer_obligations": {"type": "array", "items": {"type": "string"}},
            },
            "required": [
                "lane",
                "clinical_facts",
                "interaction",
                "risk_signals",
                "missing_information",
                "evidence_requirements",
                "answer_obligations",
            ],
            "additionalProperties": False,
        },
    },
}


def _text(value: Any, *, field: str, maximum: int = 1_000) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    return value.strip()[:maximum]


def _source_turns(value: Any, *, compiled: CompiledConversation) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("source turns must be a non-empty array")
    turns: list[int] = []
    for raw in value:
        if not isinstance(raw, int) or isinstance(raw, bool):
            raise TypeError("source turn must be an integer")
        if raw < 1 or raw > len(compiled.messages):
            raise ValueError("source turn does not exist")
        turns.append(raw)
    return tuple(dict.fromkeys(turns))


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
    lane: str
    clinical_facts: tuple[ClinicalFact, ...]
    interaction: InteractionState
    risk_signals: tuple[RiskSignal, ...]
    missing_information: tuple[MissingInformation, ...]
    evidence_requirements: tuple[EvidenceRequirement, ...]
    answer_obligations: tuple[str, ...]

    @classmethod
    def empty(cls, *, lane: str, intent: str, language: str) -> ResponsePlan:
        if lane not in VALID_LANES:
            raise ValueError("invalid response-plan lane")
        return cls(
            lane=lane,
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
            "lane": self.lane,
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
    l2_calls: int = 0
    usage: dict[str, int] | None = None


def parse_response_plan(
    arguments: Mapping[str, Any], *, compiled: CompiledConversation
) -> ResponsePlan:
    lane = arguments.get("lane")
    if lane not in VALID_LANES:
        raise ValueError("invalid response-plan lane")

    raw_facts = arguments.get("clinical_facts")
    if not isinstance(raw_facts, list) or len(raw_facts) > 16:
        raise ValueError("clinical_facts must be a bounded array")
    facts: list[ClinicalFact] = []
    for raw in raw_facts:
        if not isinstance(raw, Mapping):
            raise TypeError("clinical fact must be an object")
        facts.append(
            ClinicalFact(
                kind=_text(raw.get("kind"), field="clinical fact kind", maximum=80),
                value=_text(raw.get("value"), field="clinical fact value"),
                source_turns=_source_turns(raw.get("source_turns"), compiled=compiled),
                negated=bool(raw.get("negated", False)),
                corrected=bool(raw.get("corrected", False)),
            )
        )

    raw_interaction = arguments.get("interaction")
    if not isinstance(raw_interaction, Mapping):
        raise TypeError("interaction must be an object")
    raw_refs = raw_interaction.get("unresolved_references", [])
    if not isinstance(raw_refs, list) or not all(isinstance(item, str) for item in raw_refs):
        raise ValueError("unresolved references must be an array of strings")
    interaction = InteractionState(
        intent=_text(raw_interaction.get("intent"), field="interaction intent"),
        language=_text(raw_interaction.get("language", ""), field="interaction language", maximum=20),
        requested_format=_text(
            raw_interaction.get("requested_format", ""), field="requested format"
        ),
        unresolved_references=tuple(item.strip()[:200] for item in raw_refs if item.strip())[:8],
        strict_format=bool(raw_interaction.get("strict_format", False)),
    )

    raw_risks = arguments.get("risk_signals")
    if not isinstance(raw_risks, list) or len(raw_risks) > 12:
        raise ValueError("risk_signals must be a bounded array")
    risks: list[RiskSignal] = []
    for raw in raw_risks:
        if not isinstance(raw, Mapping):
            raise TypeError("risk signal must be an object")
        risks.append(
            RiskSignal(
                category=_text(raw.get("category"), field="risk category", maximum=80),
                source_turns=_source_turns(raw.get("source_turns"), compiled=compiled),
                materiality=_text(
                    raw.get("materiality", "material"), field="risk materiality", maximum=40
                ),
            )
        )

    raw_missing = arguments.get("missing_information")
    if not isinstance(raw_missing, list) or len(raw_missing) > 8:
        raise ValueError("missing_information must be a bounded array")
    missing: list[MissingInformation] = []
    for index, raw in enumerate(raw_missing, start=1):
        if isinstance(raw, str):
            missing.append(MissingInformation(f"missing-{index}", raw.strip()[:1_000]))
        elif isinstance(raw, Mapping):
            missing.append(
                MissingInformation(
                    id=_text(raw.get("id", f"missing-{index}"), field="missing id", maximum=80),
                    question=_text(raw.get("question"), field="missing question"),
                    material=bool(raw.get("material", True)),
                )
            )
        else:
            raise TypeError("missing information must be a string or object")

    raw_requirements = arguments.get("evidence_requirements")
    if not isinstance(raw_requirements, list) or len(raw_requirements) > 4:
        raise ValueError("evidence requirements must contain at most four items")
    requirements: list[EvidenceRequirement] = []
    for index, raw in enumerate(raw_requirements, start=1):
        if not isinstance(raw, Mapping):
            raise TypeError("evidence requirement must be an object")
        requirements.append(
            EvidenceRequirement(
                id=_text(raw.get("id", f"req-{index}"), field="requirement id", maximum=80),
                claim_or_question=_text(raw.get("question"), field="requirement question"),
                source_family=_text(
                    raw.get("source_family", "general"), field="source family", maximum=80
                ),
                jurisdiction=_text(
                    raw.get("jurisdiction", ""), field="jurisdiction", maximum=80
                ),
                criticality=_text(
                    raw.get("criticality", "critical"), field="criticality", maximum=20
                ),
            )
        )
    if requirements:
        EvidenceRequirementLedger(tuple(requirements))
    if lane == "GROUNDED" and not requirements:
        raise ValueError("grounded plan requires evidence requirements")
    if lane in {"DIRECT", "CLARIFY"} and requirements:
        raise ValueError("direct or clarify plan cannot contain evidence requirements")

    raw_obligations = arguments.get("answer_obligations")
    if not isinstance(raw_obligations, list) or not all(
        isinstance(item, str) for item in raw_obligations
    ):
        raise ValueError("answer obligations must be an array of strings")
    plan = ResponsePlan(
        lane=lane,
        clinical_facts=tuple(facts),
        interaction=interaction,
        risk_signals=tuple(risks),
        missing_information=tuple(missing),
        evidence_requirements=tuple(requirements),
        answer_obligations=tuple(item.strip()[:1_000] for item in raw_obligations if item.strip())[:12],
    )
    if compiled.full_history_high_risk and plan.lane != "HIGH_RISK":
        plan = replace(plan, lane="HIGH_RISK")
    return plan


def needs_structured_planning(compiled: CompiledConversation) -> bool:
    return (
        len(compiled.user_turn_ids) > 1
        or compiled.full_history_high_risk
        or compiled.has_implicit_reference
        or compiled.is_source_sensitive
        or compiled.has_strict_format
    )


class StructuredPlanner:
    def __init__(self, *, l2: L2Client) -> None:
        self._l2 = l2

    def plan(self, compiled: CompiledConversation, *, deadline: Deadline) -> PlanningResult:
        if not needs_structured_planning(compiled):
            return PlanningResult(
                plan=ResponsePlan.empty(lane="DIRECT", intent="direct answer", language=""),
                mode="fast_path",
            )
        try:
            response = self._l2.complete(
                [
                    {"role": "system", "content": PLANNING_SYSTEM_PROMPT},
                    {"role": "user", "content": compiled.case_packet},
                ],
                deadline=deadline,
                tools=[SUBMIT_RESPONSE_PLAN_TOOL],
            )
            if len(response.tool_calls) != 1 or response.tool_calls[0].name != "submit_response_plan":
                return PlanningResult(
                    plan=None,
                    mode="legacy_fallback",
                    fallback_reason="planner_missing_structured_call",
                    l2_calls=1,
                    usage=dict(response.usage),
                )
            arguments = parse_tool_arguments(response.tool_calls[0].arguments)
            plan = parse_response_plan(arguments, compiled=compiled)
            return PlanningResult(
                plan=plan,
                mode="structured",
                l2_calls=1,
                usage=dict(response.usage),
            )
        except (AppError, TypeError, ValueError, json.JSONDecodeError) as exc:
            return PlanningResult(
                plan=None,
                mode="legacy_fallback",
                fallback_reason=f"planner_{type(exc).__name__.lower()}",
                l2_calls=0 if isinstance(exc, AppError) else 1,
                usage={},
            )
