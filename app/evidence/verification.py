from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Any, Mapping

from app.clients.l2 import L2Client, parse_tool_arguments
from app.config import Settings
from app.deadline import Deadline
from app.errors import AppError
from app.evidence.models import EvidenceItem, EvidenceRequirement, RetrievalOutcome


VALID_VERDICTS = {"entails", "contradicts", "insufficient", "inapplicable"}


VERIFY_EVIDENCE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "verify_evidence",
        "description": (
            "Verify only whether the registered evidence supports each named requirement "
            "for the stated population and jurisdiction. Do not answer the user."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "requirements": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "requirement_id": {"type": "string"},
                            "verdict": {
                                "type": "string",
                                "enum": sorted(VALID_VERDICTS),
                            },
                            "cite_uids": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "applicability_note": {"type": "string"},
                            "conflict_note": {"type": "string"},
                        },
                        "required": [
                            "requirement_id",
                            "verdict",
                            "cite_uids",
                            "applicability_note",
                            "conflict_note",
                        ],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["requirements"],
            "additionalProperties": False,
        },
    },
}


@dataclass(frozen=True, slots=True)
class VerificationRun:
    outcome: RetrievalOutcome
    l2_calls: int
    usage: dict[str, int]
    state: str
    issues: tuple[str, ...]


def _source_family(item: EvidenceItem) -> str:
    tool = item.source_tool.lower()
    source_type = item.source_type.lower()
    if tool.startswith("adr_") or "dailymed" in source_type:
        return "drug_safety"
    if tool.startswith("openapi_mfds"):
        return "drug"
    if tool.startswith("openapi_hira") or tool.startswith("hira_"):
        return "hira"
    if tool.startswith("kcd_") or source_type == "kcd":
        return "coding"
    if tool.startswith("openapi_law"):
        return "law"
    if tool.startswith("index_"):
        return source_type if source_type in {"guideline", "hira"} else "index"
    if tool.startswith("rag_"):
        return "faers" if "faers" in source_type else "research"
    return source_type or "unknown"


def _jurisdiction(item: EvidenceItem) -> str:
    if item.jurisdiction:
        return item.jurisdiction.upper()
    tool = item.source_tool.lower()
    if tool.startswith(("openapi_mfds", "openapi_hira", "hira_", "kcd_", "openapi_law")):
        return "KR"
    if tool.startswith("adr_"):
        return "US"
    return ""


def _compatible_source(expected: str, actual: str) -> bool | None:
    if not expected:
        return True
    aliases = {
        "drug": {"drug", "drug_safety"},
        "drug_safety": {"drug_safety", "drug"},
        "guideline": {"guideline", "index"},
        "hira": {"hira", "index"},
        "coding": {"coding"},
        "law": {"law"},
        "research": {"research"},
        "faers": {"faers"},
    }
    if actual == "unknown":
        return None
    return actual in aliases.get(expected, {expected})


def structural_verify(outcome: RetrievalOutcome) -> tuple[RetrievalOutcome, tuple[str, ...]]:
    items_by_id = {item.cite_uid: item for item in outcome.evidence}
    issues: list[str] = []
    verified: list[EvidenceRequirement] = []
    for requirement in outcome.requirements:
        if requirement.status not in {"supported", "contradicted"}:
            verified.append(
                replace(
                    requirement,
                    verification_status="unverified",
                    applicability_status=(
                        requirement.applicability_status
                        if requirement.applicability_status != "unchecked"
                        else "unknown"
                    ),
                )
            )
            continue
        linked = tuple(items_by_id[cite_uid] for cite_uid in requirement.cite_uids if cite_uid in items_by_id)
        if not linked:
            issues.append(f"{requirement.id}:missing_registered_evidence")
            verified.append(
                replace(
                    requirement,
                    status="unresolved",
                    cite_uids=(),
                    gap_reason="No selected registered evidence was linked to this requirement.",
                    verification_status="structural_rejected",
                    applicability_status="unknown",
                )
            )
            continue

        source_checks = [
            _compatible_source(requirement.source_preference, _source_family(item))
            for item in linked
        ]
        if source_checks and all(check is False for check in source_checks):
            issues.append(f"{requirement.id}:source_family_mismatch")
            verified.append(
                replace(
                    requirement,
                    status="unresolved",
                    gap_reason="The selected evidence came from the wrong source family.",
                    verification_status="structural_rejected",
                    applicability_status="inapplicable",
                )
            )
            continue

        if requirement.jurisdiction:
            known_jurisdictions = {_jurisdiction(item) for item in linked if _jurisdiction(item)}
            if known_jurisdictions and requirement.jurisdiction.upper() not in known_jurisdictions:
                issues.append(f"{requirement.id}:jurisdiction_mismatch")
                verified.append(
                    replace(
                        requirement,
                        status="unresolved",
                        gap_reason="The selected evidence did not match the requested jurisdiction.",
                        verification_status="structural_rejected",
                        applicability_status="inapplicable",
                    )
                )
                continue

        applicability = "applicable"
        if any(check is None for check in source_checks):
            applicability = "unknown"
        if requirement.time_sensitive and not any(item.version_or_date for item in linked):
            applicability = "unknown"
            issues.append(f"{requirement.id}:date_unknown")
        verified.append(
            replace(
                requirement,
                verification_status="structural_verified",
                applicability_status=applicability,
            )
        )

    status = outcome.status
    unresolved_critical = any(
        requirement.criticality == "critical" and requirement.status != "supported"
        for requirement in verified
    )
    if status == "sufficient" and unresolved_critical:
        status = "partial" if outcome.evidence else "no_evidence"
    return replace(outcome, status=status, requirements=tuple(verified)), tuple(issues)


class EvidenceVerifier:
    def __init__(self, settings: Settings, l2: L2Client) -> None:
        self._settings = settings
        self._l2 = l2

    def run(
        self,
        outcome: RetrievalOutcome,
        *,
        clinical_risk: str,
        deadline: Deadline,
        allow_semantic: bool,
    ) -> VerificationRun:
        structurally_verified, structural_issues = structural_verify(outcome)
        policy = getattr(self._settings, "evidence_verification_policy", "structural")
        should_semantically_verify = bool(
            allow_semantic
            and structurally_verified.evidence
            and policy != "structural"
            and (policy == "always" or clinical_risk in {"elevated", "urgent", "emergency"})
        )
        if not should_semantically_verify:
            return VerificationRun(
                structurally_verified,
                0,
                {},
                "structural_only",
                structural_issues,
            )

        payload = structurally_verified.to_generation_payload(
            max_chars=self._settings.max_evidence_chars
        )
        try:
            response = self._l2.complete(
                [
                    {
                        "role": "system",
                        "content": (
                            "You verify evidence for a medical harness. Judge only whether each "
                            "registered citation entails or contradicts the named requirement for "
                            "the stated constraints and jurisdiction. Topic mention is not support. "
                            "Use only known requirement and cite IDs, introduce no new medical fact, "
                            "and call verify_evidence."
                        ),
                    },
                    {"role": "user", "content": payload},
                ],
                deadline=deadline.with_timeout_cap(
                    getattr(self._settings, "evidence_verifier_timeout_sec", 30.0)
                ),
                tools=[VERIFY_EVIDENCE_TOOL],
                tool_choice="required",
            )
        except AppError:
            return VerificationRun(
                _mark_semantic_unverified(structurally_verified, "semantic_verifier_unavailable"),
                1,
                {},
                "semantic_unavailable",
                (*structural_issues, "semantic_verifier_unavailable"),
            )

        parsed = _parse_semantic_response(response, structurally_verified)
        if parsed is None:
            return VerificationRun(
                _mark_semantic_unverified(structurally_verified, "semantic_verifier_invalid"),
                1,
                dict(response.usage),
                "semantic_invalid",
                (*structural_issues, "semantic_verifier_invalid"),
            )
        verified_requirements, semantic_issues = parsed
        status = structurally_verified.status
        unresolved_critical = any(
            requirement.criticality == "critical" and requirement.status != "supported"
            for requirement in verified_requirements
        )
        if unresolved_critical:
            status = "partial" if structurally_verified.evidence else "no_evidence"
        elif verified_requirements:
            status = "sufficient"
        return VerificationRun(
            replace(
                structurally_verified,
                status=status,
                requirements=verified_requirements,
            ),
            1,
            dict(response.usage),
            "semantic_verified",
            (*structural_issues, *semantic_issues),
        )


def _mark_semantic_unverified(
    outcome: RetrievalOutcome,
    reason: str,
) -> RetrievalOutcome:
    requirements = tuple(
        replace(
            requirement,
            status=(
                "unresolved"
                if requirement.criticality == "critical" and requirement.status == "supported"
                else requirement.status
            ),
            gap_reason=(
                "Semantic evidence verification was unavailable."
                if requirement.criticality == "critical" and requirement.status == "supported"
                else requirement.gap_reason
            ),
            verification_status=(
                reason
                if requirement.criticality == "critical" and requirement.status == "supported"
                else requirement.verification_status
            ),
        )
        for requirement in outcome.requirements
    )
    return replace(
        outcome,
        status="partial" if outcome.evidence else "no_evidence",
        requirements=requirements,
    )


def _parse_semantic_response(
    response: Any,
    outcome: RetrievalOutcome,
) -> tuple[tuple[EvidenceRequirement, ...], tuple[str, ...]] | None:
    arguments: Mapping[str, Any] | None = None
    for call in response.tool_calls or ():
        if call.name != "verify_evidence":
            continue
        try:
            arguments = parse_tool_arguments(call.arguments)
        except (ValueError, json.JSONDecodeError):
            return None
        break
    if arguments is None or not isinstance(arguments.get("requirements"), list):
        return None
    expected = {item.id: item for item in outcome.requirements}
    known_cites = {item.cite_uid for item in outcome.evidence}
    updates: dict[str, EvidenceRequirement] = {}
    issues: list[str] = []
    for raw in arguments["requirements"]:
        if not isinstance(raw, Mapping):
            continue
        identifier = raw.get("requirement_id")
        verdict = raw.get("verdict")
        raw_cites = raw.get("cite_uids")
        if not isinstance(identifier, str) or identifier not in expected:
            issues.append("unknown_requirement")
            continue
        if verdict not in VALID_VERDICTS or not isinstance(raw_cites, list):
            issues.append(f"{identifier}:invalid_verdict")
            continue
        cite_uids = tuple(
            dict.fromkeys(
                cite_uid
                for cite_uid in raw_cites
                if isinstance(cite_uid, str) and cite_uid in known_cites
            )
        )
        requirement = expected[identifier]
        applicability_note = raw.get("applicability_note")
        conflict_note = raw.get("conflict_note")
        note = " ".join(
            value.strip()
            for value in (applicability_note, conflict_note)
            if isinstance(value, str) and value.strip()
        )[:500]
        if verdict == "entails" and cite_uids and requirement.applicability_status != "inapplicable":
            updates[identifier] = replace(
                requirement,
                status="supported",
                cite_uids=cite_uids,
                gap_reason="",
                verification_status="semantic_verified",
                applicability_status=(
                    requirement.applicability_status
                    if requirement.applicability_status != "unchecked"
                    else "applicable"
                ),
            )
        elif verdict == "contradicts" and cite_uids:
            updates[identifier] = replace(
                requirement,
                status="contradicted",
                cite_uids=cite_uids,
                gap_reason=note or "The selected evidence contradicts the requirement.",
                verification_status="semantic_verified",
            )
        else:
            updates[identifier] = replace(
                requirement,
                status="unresolved",
                cite_uids=cite_uids,
                gap_reason=note or "The evidence did not establish this requirement.",
                verification_status="semantic_verified",
                applicability_status=(
                    "inapplicable" if verdict == "inapplicable" else requirement.applicability_status
                ),
            )
            issues.append(f"{identifier}:{verdict}")

    finalized = tuple(
        updates.get(
            requirement.id,
            replace(
                requirement,
                status="unresolved" if requirement.criticality == "critical" else requirement.status,
                gap_reason=(
                    "Semantic verification did not close this requirement."
                    if requirement.criticality == "critical"
                    else requirement.gap_reason
                ),
            ),
        )
        for requirement in outcome.requirements
    )
    return finalized, tuple(issues)
