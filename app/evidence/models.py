from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


VALID_RETRIEVAL_STATUSES = {"sufficient", "partial", "no_evidence"}
VALID_REQUIREMENT_STATUSES = {"missing", "supported", "contradicted", "unresolved"}


def _first_string(mapping: Mapping[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _compact_json(value: Any, maximum: int = 4_000) -> str:
    try:
        serialized = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        serialized = str(value)
    if len(serialized) <= maximum:
        return serialized
    return serialized[:maximum] + "…"


def _evidence_content(mapping: Mapping[str, Any]) -> str:
    direct = _first_string(
        mapping,
        (
            "content",
            "text",
            "page_text",
            "excerpt",
            "summary",
            "label",
            "article_text",
            "indication",
        ),
    )
    if direct:
        return direct
    pages = mapping.get("pages")
    if not isinstance(pages, list):
        return ""
    page_texts: list[str] = []
    for page in pages:
        if isinstance(page, str) and page.strip():
            page_texts.append(page.strip())
        elif isinstance(page, dict):
            text = page.get("text")
            if isinstance(text, str) and text.strip():
                number = page.get("page")
                prefix = f"[page {number}]\n" if isinstance(number, (int, str)) else ""
                page_texts.append(prefix + text.strip())
    return "\n\n".join(page_texts)[:12_000]


@dataclass(frozen=True, slots=True)
class EvidenceItem:
    cite_uid: str
    source_tool: str
    title: str = ""
    url: str = ""
    content: str = ""
    source_type: str = ""
    jurisdiction: str = ""
    version_or_date: str = ""
    content_hash: str = ""
    raw_preview: str = ""

    def to_generation_record(self, relevance_score: float) -> dict[str, Any]:
        return {
            "cite_uid": self.cite_uid,
            "relevance_score": relevance_score,
            "source_tool": self.source_tool,
            "source_type": self.source_type,
            "jurisdiction": self.jurisdiction,
            "title": self.title,
            "url": self.url,
            "version_or_date": self.version_or_date,
            "verification_status": "candidate",
            "content": self.content or self.raw_preview,
        }


@dataclass(slots=True)
class EvidenceRegistry:
    _items: dict[str, EvidenceItem] = field(default_factory=dict)

    def register_payload(self, payload: Any, *, source_tool: str) -> tuple[str, ...]:
        registered: list[str] = []

        def visit(value: Any) -> None:
            if isinstance(value, dict):
                cite_uid = value.get("cite_uid")
                if isinstance(cite_uid, str) and cite_uid.strip():
                    identifier = cite_uid.strip()
                    if identifier not in self._items:
                        content = _evidence_content(value)
                        self._items[identifier] = EvidenceItem(
                            cite_uid=identifier,
                            source_tool=source_tool,
                            title=_first_string(value, ("title", "name", "document_title")),
                            url=_first_string(value, ("url", "source_link", "link")),
                            content=content,
                            source_type=_first_string(
                                value, ("source_type", "source", "corpus_tag", "data_source")
                            ),
                            jurisdiction=_first_string(value, ("jurisdiction", "country")),
                            version_or_date=_first_string(
                                value,
                                ("version", "date", "updated_at", "effective_date"),
                            ),
                            content_hash=hashlib.sha256(
                                (content or _compact_json(value)).encode("utf-8")
                            ).hexdigest()[:16],
                            raw_preview=_compact_json(value),
                        )
                    registered.append(identifier)
                for nested in value.values():
                    visit(nested)
            elif isinstance(value, list):
                for nested in value:
                    visit(nested)
            elif isinstance(value, str):
                candidate = value.strip()
                if candidate.startswith(("{", "[")):
                    try:
                        decoded = json.loads(candidate)
                    except json.JSONDecodeError:
                        return
                    if not isinstance(decoded, str):
                        visit(decoded)

        visit(payload)
        return tuple(dict.fromkeys(registered))

    def get(self, cite_uid: str) -> EvidenceItem | None:
        return self._items.get(cite_uid)

    def all(self) -> tuple[EvidenceItem, ...]:
        return tuple(self._items.values())

    def __len__(self) -> int:
        return len(self._items)


@dataclass(frozen=True, slots=True)
class CitableItem:
    cite_uid: str
    relevance_score: float


@dataclass(frozen=True, slots=True)
class EvidenceRequirement:
    id: str
    claim_or_question: str
    source_preference: str = ""
    clinical_constraints: str = ""
    jurisdiction: str = ""
    time_sensitive: bool = False
    criticality: str = "critical"
    status: str = "missing"
    cite_uids: tuple[str, ...] = ()
    gap_reason: str = ""
    verification_status: str = "unverified"
    applicability_status: str = "unchecked"


@dataclass(frozen=True, slots=True)
class EvidenceRequirementLedger:
    requirements: tuple[EvidenceRequirement, ...]

    @property
    def primary(self) -> EvidenceRequirement:
        return self.requirements[0]

    def unresolved_critical(self) -> tuple[EvidenceRequirement, ...]:
        return tuple(
            requirement
            for requirement in self.requirements
            if requirement.criticality == "critical"
            and requirement.status != "supported"
        )


@dataclass(frozen=True, slots=True)
class RetrievalOutcome:
    status: str
    items: tuple[CitableItem, ...]
    note: str
    evidence: tuple[EvidenceItem, ...]
    requirements: tuple[EvidenceRequirement, ...]
    model_rounds: int
    mcp_calls: int

    @property
    def requirement(self) -> EvidenceRequirement:
        return self.requirements[0]

    @property
    def ledger(self) -> EvidenceRequirementLedger:
        return EvidenceRequirementLedger(self.requirements)

    def to_generation_payload(self, *, max_chars: int) -> str:
        score_by_id = {item.cite_uid: item.relevance_score for item in self.items}
        records = [
            item.to_generation_record(score_by_id.get(item.cite_uid, 0.0))
            for item in self.evidence
        ]
        requirement_records = [
            {
                "id": requirement.id,
                "claim_or_question": requirement.claim_or_question,
                "source_preference": requirement.source_preference,
                "clinical_constraints": requirement.clinical_constraints,
                "jurisdiction": requirement.jurisdiction,
                "time_sensitive": requirement.time_sensitive,
                "criticality": requirement.criticality,
                "status": requirement.status,
                "cite_uids": list(requirement.cite_uids),
                "gap_reason": requirement.gap_reason,
                "verification_status": requirement.verification_status,
                "applicability_status": requirement.applicability_status,
            }
            for requirement in self.requirements
        ]
        payload = {
            "status": self.status,
            "note": self.note,
            "requirements": requirement_records,
            "requirement": {
                "claim_or_question": self.requirement.claim_or_question,
                "status": self.requirement.status,
                "gap_reason": self.requirement.gap_reason,
            },
            "evidence": records,
            "instruction": (
                "These records are the only citable sources: cite nothing else and never "
                "fabricate support. They supplement your medical knowledge rather than "
                "limiting it, so when the status is partial or no_evidence, or the records "
                "do not address the question, answer from established medical knowledge and "
                "attach no citation to those parts. Do not refuse and do not describe the "
                "retrieval. These records are often Korean: reply in the user's own language "
                "and translate what you cite, never switching language to match the evidence. "
                "cite_uid is provenance, not answer confidence."
            ),
        }
        serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        if len(serialized) <= max_chars:
            return serialized
        reduced = dict(payload)
        reduced["evidence"] = []
        remaining = max_chars - len(
            json.dumps(reduced, ensure_ascii=False, separators=(",", ":"))
        )
        for record in records:
            compact = dict(record)
            content = compact.get("content")
            if isinstance(content, str) and len(content) > 1_200:
                compact["content"] = content[:1_200] + "…"
            candidate = [*reduced["evidence"], compact]
            reduced["evidence"] = candidate
            rendered = json.dumps(reduced, ensure_ascii=False, separators=(",", ":"))
            if len(rendered) > max_chars or remaining <= 0:
                reduced["evidence"].pop()
                break
        return json.dumps(reduced, ensure_ascii=False, separators=(",", ":"))


def default_requirement_ledger(
    query: str,
    *,
    requirements: Iterable[EvidenceRequirement] | None = None,
) -> EvidenceRequirementLedger:
    if requirements is None:
        return EvidenceRequirementLedger(
            (
                EvidenceRequirement(
                    id="req-1",
                    claim_or_question=query,
                ),
            )
        )

    normalized: list[EvidenceRequirement] = []
    seen: set[str] = set()
    for requirement in requirements:
        identifier = requirement.id.strip()
        if not identifier or identifier in seen:
            continue
        seen.add(identifier)
        normalized.append(
            EvidenceRequirement(
                id=identifier,
                claim_or_question=requirement.claim_or_question.strip() or query,
                source_preference=requirement.source_preference.strip(),
                clinical_constraints=requirement.clinical_constraints.strip(),
                jurisdiction=requirement.jurisdiction.strip(),
                time_sensitive=bool(requirement.time_sensitive),
                criticality=requirement.criticality.strip() or "critical",
                status=(
                    requirement.status
                    if requirement.status in VALID_REQUIREMENT_STATUSES
                    else "missing"
                ),
                cite_uids=tuple(
                    cite_uid.strip()
                    for cite_uid in requirement.cite_uids
                    if isinstance(cite_uid, str) and cite_uid.strip()
                ),
                gap_reason=requirement.gap_reason.strip(),
                verification_status=requirement.verification_status,
                applicability_status=requirement.applicability_status,
            )
        )
    if not normalized:
        return default_requirement_ledger(query)
    return EvidenceRequirementLedger(tuple(normalized))


def _outcome_requirement(
    requirement: EvidenceRequirement,
    *,
    status: str,
    cite_uids: tuple[str, ...],
    note: str,
) -> EvidenceRequirement:
    if status == "sufficient" and cite_uids:
        return EvidenceRequirement(
            id=requirement.id,
            claim_or_question=requirement.claim_or_question,
            source_preference=requirement.source_preference,
            clinical_constraints=requirement.clinical_constraints,
            jurisdiction=requirement.jurisdiction,
            time_sensitive=requirement.time_sensitive,
            criticality=requirement.criticality,
            status="supported",
            cite_uids=cite_uids,
            gap_reason="",
            verification_status=requirement.verification_status,
            applicability_status=requirement.applicability_status,
        )
    return EvidenceRequirement(
        id=requirement.id,
        claim_or_question=requirement.claim_or_question,
        source_preference=requirement.source_preference,
        clinical_constraints=requirement.clinical_constraints,
        jurisdiction=requirement.jurisdiction,
        time_sensitive=requirement.time_sensitive,
        criticality=requirement.criticality,
        status="unresolved",
        cite_uids=cite_uids,
        gap_reason=note or "Evidence remained incomplete.",
        verification_status=requirement.verification_status,
        applicability_status=requirement.applicability_status,
    )


def _append_note(note: str, message: str) -> str:
    if not message:
        return note
    if not note:
        return message
    return f"{note} {message}".strip()


def parse_final_selection(
    arguments: Mapping[str, Any],
    *,
    registry: EvidenceRegistry,
    query: str,
    model_rounds: int,
    mcp_calls: int,
    max_items: int,
    requirements: Iterable[EvidenceRequirement] | None = None,
) -> RetrievalOutcome:
    raw_status = arguments.get("status")
    if raw_status not in VALID_RETRIEVAL_STATUSES:
        raise ValueError("invalid finalize_retrieval status")
    note_value = arguments.get("note", "")
    note = note_value.strip()[:2_000] if isinstance(note_value, str) else ""
    ledger = default_requirement_ledger(query, requirements=requirements)
    raw_items = arguments.get("selected_items", arguments.get("items", []))
    if not isinstance(raw_items, list):
        raise ValueError("finalize_retrieval items must be an array")
    raw_requirements = arguments.get("requirements")
    if raw_requirements is not None and not isinstance(raw_requirements, list):
        raise ValueError("finalize_retrieval requirements must be an array")

    selected: list[CitableItem] = []
    unknown: list[str] = []
    seen: set[str] = set()
    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        cite_uid = raw.get("cite_uid")
        score = raw.get("relevance_score")
        if not isinstance(cite_uid, str) or not cite_uid or cite_uid in seen:
            continue
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            continue
        parsed_score = float(score)
        if not math.isfinite(parsed_score):
            continue
        seen.add(cite_uid)
        if registry.get(cite_uid) is None:
            unknown.append(cite_uid)
            continue
        selected.append(CitableItem(cite_uid=cite_uid, relevance_score=parsed_score))
        if len(selected) >= max_items:
            break

    status = raw_status
    if status == "no_evidence":
        selected = []
    elif not selected:
        status = "partial" if len(registry) else "no_evidence"
    if status == "sufficient" and not selected:
        status = "partial" if len(registry) else "no_evidence"
    if unknown:
        suffix = f"Unknown cite_uids were rejected: {', '.join(unknown[:3])}."
        note = _append_note(note, suffix)
        if status == "sufficient":
            status = "partial" if selected or len(registry) else "no_evidence"

    selected_cite_uids = tuple(item.cite_uid for item in selected)
    selected_cite_uid_set = set(selected_cite_uids)
    updates_by_id: dict[str, EvidenceRequirement] = {}
    unknown_requirement_ids: list[str] = []
    rejected_requirement_cites: list[str] = []
    duplicate_requirement_ids: list[str] = []
    if raw_requirements is None:
        finalized_requirements = tuple(
            _outcome_requirement(
                requirement,
                status=status if requirement.id == ledger.primary.id else "partial",
                cite_uids=selected_cite_uids if requirement.id == ledger.primary.id else (),
                note=note,
            )
            for requirement in ledger.requirements
        )
    else:
        expected_by_id = {requirement.id: requirement for requirement in ledger.requirements}
        for raw_requirement in raw_requirements:
            if not isinstance(raw_requirement, Mapping):
                continue
            identifier = raw_requirement.get("id")
            if not isinstance(identifier, str) or not identifier.strip():
                unknown_requirement_ids.append("<missing>")
                continue
            requirement_id = identifier.strip()
            if requirement_id in updates_by_id:
                duplicate_requirement_ids.append(requirement_id)
                continue
            expected = expected_by_id.get(requirement_id)
            if expected is None:
                unknown_requirement_ids.append(requirement_id)
                continue
            requirement_status = raw_requirement.get("status", expected.status)
            if requirement_status not in VALID_REQUIREMENT_STATUSES:
                requirement_status = "unresolved"
            raw_cite_uids = raw_requirement.get("cite_uids", list(expected.cite_uids))
            cite_uids: list[str] = []
            if raw_cite_uids is None:
                raw_cite_uids = []
            if not isinstance(raw_cite_uids, list):
                rejected_requirement_cites.append(f"{requirement_id}:<invalid-array>")
                raw_cite_uids = []
                requirement_status = "unresolved"
            for raw_cite_uid in raw_cite_uids:
                if not isinstance(raw_cite_uid, str) or not raw_cite_uid.strip():
                    continue
                cite_uid = raw_cite_uid.strip()
                if (
                    registry.get(cite_uid) is None
                    or cite_uid not in selected_cite_uid_set
                    or cite_uid in cite_uids
                ):
                    rejected_requirement_cites.append(f"{requirement_id}:{cite_uid}")
                    requirement_status = "unresolved"
                    continue
                cite_uids.append(cite_uid)
            if requirement_status in {"supported", "contradicted"} and not cite_uids:
                requirement_status = "unresolved"
            gap_reason_value = raw_requirement.get("gap_reason", expected.gap_reason)
            gap_reason = gap_reason_value.strip()[:500] if isinstance(gap_reason_value, str) else ""
            if requirement_status == "unresolved" and not gap_reason:
                gap_reason = note or "Evidence remained incomplete."
            if requirement_status == "supported":
                gap_reason = ""
            updates_by_id[requirement_id] = EvidenceRequirement(
                id=expected.id,
                claim_or_question=expected.claim_or_question,
                source_preference=expected.source_preference,
                clinical_constraints=expected.clinical_constraints,
                jurisdiction=expected.jurisdiction,
                time_sensitive=expected.time_sensitive,
                criticality=expected.criticality,
                status=requirement_status,
                cite_uids=tuple(cite_uids),
                gap_reason=gap_reason,
                verification_status=(
                    "model_selected" if requirement_status in {"supported", "contradicted"} else "unverified"
                ),
                applicability_status=expected.applicability_status,
            )

        finalized_requirements = tuple(
            updates_by_id.get(
                requirement.id,
                EvidenceRequirement(
                    id=requirement.id,
                    claim_or_question=requirement.claim_or_question,
                    source_preference=requirement.source_preference,
                    clinical_constraints=requirement.clinical_constraints,
                    jurisdiction=requirement.jurisdiction,
                    time_sensitive=requirement.time_sensitive,
                    criticality=requirement.criticality,
                    status="unresolved",
                    cite_uids=(),
                    gap_reason=note or "Requirement was not closed by finalize_retrieval.",
                    verification_status="unverified",
                    applicability_status=requirement.applicability_status,
                ),
            )
            for requirement in ledger.requirements
        )
        if unknown_requirement_ids:
            note = _append_note(
                note,
                "Unknown requirement ids were rejected: "
                + ", ".join(unknown_requirement_ids[:3])
                + ".",
            )
        if duplicate_requirement_ids:
            note = _append_note(
                note,
                "Duplicate requirement ids were ignored: "
                + ", ".join(duplicate_requirement_ids[:3])
                + ".",
            )
        if rejected_requirement_cites:
            note = _append_note(
                note,
                "Requirement cite_uids were rejected: "
                + ", ".join(rejected_requirement_cites[:3])
                + ".",
            )

    unresolved_critical = tuple(
        requirement
        for requirement in finalized_requirements
        if requirement.criticality == "critical"
        and requirement.status != "supported"
    )
    if status == "sufficient" and unresolved_critical:
        status = "partial" if selected or len(registry) else "no_evidence"
        note = _append_note(
            note,
            "Critical requirements remained unresolved: "
            + ", ".join(requirement.id for requirement in unresolved_critical[:3])
            + ".",
        )

    evidence = tuple(
        item
        for selection in selected
        if (item := registry.get(selection.cite_uid)) is not None
    )
    return RetrievalOutcome(
        status=status,
        items=tuple(selected),
        note=note,
        evidence=evidence,
        requirements=finalized_requirements,
        model_rounds=model_rounds,
        mcp_calls=mcp_calls,
    )


def fallback_selection(
    *,
    registry: EvidenceRegistry,
    query: str,
    requirements: Iterable[EvidenceRequirement] | None = None,
    note: str,
    model_rounds: int,
    mcp_calls: int,
    max_items: int,
) -> RetrievalOutcome:
    ledger = default_requirement_ledger(query, requirements=requirements)
    # Registry membership proves provenance, not relevance. On budget/finalizer failure,
    # expose only evidence already linked to a requirement; never promote the first page
    # merely because the bridge happened to retrieve it.
    linked_ids = tuple(
        dict.fromkeys(
            cite_uid
            for requirement in ledger.requirements
            if requirement.status in {"supported", "contradicted"}
            for cite_uid in requirement.cite_uids
            if registry.get(cite_uid) is not None
        )
    )[:max_items]
    evidence = tuple(
        item for cite_uid in linked_ids if (item := registry.get(cite_uid)) is not None
    )
    items = tuple(CitableItem(item.cite_uid, 0.0) for item in evidence)
    status = "partial" if evidence else "no_evidence"
    return RetrievalOutcome(
        status=status,
        items=items,
        note=note,
        evidence=evidence,
        requirements=tuple(
            requirement
            if requirement.status in {"supported", "contradicted"}
            else _outcome_requirement(
                requirement,
                status=status,
                cite_uids=(),
                note=note,
            )
            for requirement in ledger.requirements
        ),
        model_rounds=model_rounds,
        mcp_calls=mcp_calls,
    )
