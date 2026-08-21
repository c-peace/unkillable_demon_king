from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

from app.clients.l2 import L2Client, parse_tool_arguments
from app.clinical_state import (
    AssertionState,
    ClinicalFact,
    ClinicalState,
    TurnProvenance,
)
from app.deadline import Deadline
from app.errors import AppError


ENRICH_CLINICAL_STATE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "submit_clinical_state_enrichment",
        "description": (
            "Add only clinically relevant facts that are explicitly supported by named "
            "conversation turns. Do not diagnose or answer the user."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "facts": {
                    "type": "array",
                    "maxItems": 24,
                    "items": {
                        "type": "object",
                        "properties": {
                            "kind": {"type": "string"},
                            "value": {"type": "string"},
                            "subject": {"type": "string"},
                            "assertion": {
                                "type": "string",
                                "enum": [item.value for item in AssertionState],
                            },
                            "turn_indices": {
                                "type": "array",
                                "items": {"type": "integer"},
                                "minItems": 1,
                            },
                        },
                        "required": ["kind", "value", "subject", "assertion", "turn_indices"],
                        "additionalProperties": False,
                    },
                },
                "hard_risk_signals": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 8,
                },
            },
            "required": ["facts", "hard_risk_signals"],
            "additionalProperties": False,
        },
    },
}


@dataclass(frozen=True, slots=True)
class StateEnrichmentRun:
    state: ClinicalState
    l2_calls: int
    usage: dict[str, int]
    status: str


class ClinicalStateEnricher:
    def __init__(self, l2: L2Client) -> None:
        self._l2 = l2

    def run(
        self,
        state: ClinicalState,
        *,
        case_packet: str,
        turn_roles: tuple[str, ...],
        deadline: Deadline,
    ) -> StateEnrichmentRun:
        if not state.needs_semantic_enrichment:
            return StateEnrichmentRun(state, 0, {}, "not_needed")
        try:
            response = self._l2.complete(
                [
                    {
                        "role": "system",
                        "content": (
                            "You enrich a request-local clinical state for a medical harness. "
                            "Extract only facts explicitly present in the role-labelled turns, "
                            "preserve negation and uncertainty, cite source turn numbers, and "
                            "never infer a diagnosis or write an answer. Hard-risk signals may "
                            "only upgrade caution; omit them when the text is quoted for a pure "
                            "translation or rewrite task."
                        ),
                    },
                    {"role": "user", "content": case_packet},
                ],
                deadline=deadline,
                tools=[ENRICH_CLINICAL_STATE_TOOL],
                tool_choice="required",
            )
        except AppError:
            return StateEnrichmentRun(state, 0, {}, "unavailable")
        parsed = _parse_enrichment(response, state, turn_roles)
        if parsed is None:
            return StateEnrichmentRun(state, 1, dict(response.usage), "invalid")
        return StateEnrichmentRun(parsed, 1, dict(response.usage), "enriched")


def _parse_enrichment(
    response: Any,
    state: ClinicalState,
    turn_roles: tuple[str, ...],
) -> ClinicalState | None:
    arguments: Mapping[str, Any] | None = None
    for call in response.tool_calls or ():
        if call.name != "submit_clinical_state_enrichment":
            continue
        try:
            arguments = parse_tool_arguments(call.arguments)
        except (ValueError, json.JSONDecodeError):
            return None
        break
    if arguments is None:
        return None
    raw_facts = arguments.get("facts")
    raw_risks = arguments.get("hard_risk_signals")
    if not isinstance(raw_facts, list) or not isinstance(raw_risks, list):
        return None
    facts = list(state.facts)
    existing = {
        (fact.kind, fact.value.lower(), fact.subject, fact.assertion.value)
        for fact in facts
    }
    for raw in raw_facts[:24]:
        if not isinstance(raw, Mapping):
            continue
        kind = raw.get("kind")
        value = raw.get("value")
        subject = raw.get("subject")
        assertion_value = raw.get("assertion")
        indices = raw.get("turn_indices")
        if not all(isinstance(item, str) and item.strip() for item in (kind, value, subject)):
            continue
        if assertion_value not in {item.value for item in AssertionState}:
            continue
        if not isinstance(indices, list):
            continue
        valid_indices = tuple(
            dict.fromkeys(
                index
                for index in indices
                if isinstance(index, int) and 1 <= index <= len(turn_roles)
            )
        )
        if not valid_indices:
            continue
        marker = (kind.strip(), value.strip().lower(), subject.strip(), assertion_value)
        if marker in existing:
            continue
        existing.add(marker)
        facts.append(
            ClinicalFact(
                kind=kind.strip()[:80],
                value=value.strip()[:300],
                subject=subject.strip()[:80],
                assertion=AssertionState(assertion_value),
                provenance=tuple(
                    TurnProvenance(index, turn_roles[index - 1], "model-linked")
                    for index in valid_indices
                ),
            )
        )
    risks = tuple(
        dict.fromkeys(
            (*state.hard_risk_signals,)
            + tuple(
                value.strip()[:80]
                for value in raw_risks[:8]
                if isinstance(value, str) and value.strip()
            )
        )
    )
    return ClinicalState(
        facts=tuple(facts),
        symptoms=state.symptoms,
        medications=state.medications,
        interaction=state.interaction,
        active_symptom_names=state.active_symptom_names,
        hard_risk_signals=risks,
        conflicts=state.conflicts,
        unresolved_references=state.unresolved_references,
        needs_semantic_enrichment=False,
    )
