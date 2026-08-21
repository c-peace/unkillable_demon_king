from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


VALID_RETRIEVAL_STATUSES = {"sufficient", "partial", "no_evidence"}


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
    criticality: str = "critical"
    status: str = "missing"
    cite_uids: tuple[str, ...] = ()
    gap_reason: str = ""


@dataclass(frozen=True, slots=True)
class RetrievalOutcome:
    status: str
    items: tuple[CitableItem, ...]
    note: str
    evidence: tuple[EvidenceItem, ...]
    requirement: EvidenceRequirement
    model_rounds: int
    mcp_calls: int

    def to_generation_payload(self, *, max_chars: int) -> str:
        score_by_id = {item.cite_uid: item.relevance_score for item in self.items}
        records = [
            item.to_generation_record(score_by_id.get(item.cite_uid, 0.0))
            for item in self.evidence
        ]
        payload = {
            "status": self.status,
            "note": self.note,
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


def parse_final_selection(
    arguments: Mapping[str, Any],
    *,
    registry: EvidenceRegistry,
    query: str,
    model_rounds: int,
    mcp_calls: int,
    max_items: int,
) -> RetrievalOutcome:
    raw_status = arguments.get("status")
    if raw_status not in VALID_RETRIEVAL_STATUSES:
        raise ValueError("invalid finalize_retrieval status")
    note_value = arguments.get("note", "")
    note = note_value.strip()[:2_000] if isinstance(note_value, str) else ""
    raw_items = arguments.get("items", [])
    if not isinstance(raw_items, list):
        raise ValueError("finalize_retrieval items must be an array")

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
        note = f"{note} {suffix}".strip()
        if status == "sufficient":
            status = "partial" if selected or len(registry) else "no_evidence"

    evidence = tuple(
        item
        for selection in selected
        if (item := registry.get(selection.cite_uid)) is not None
    )
    requirement_status = "supported" if status == "sufficient" and evidence else "unresolved"
    gap_reason = "" if requirement_status == "supported" else note or "Evidence remained incomplete."
    return RetrievalOutcome(
        status=status,
        items=tuple(selected),
        note=note,
        evidence=evidence,
        requirement=EvidenceRequirement(
            id="req-1",
            claim_or_question=query,
            status=requirement_status,
            cite_uids=tuple(item.cite_uid for item in selected),
            gap_reason=gap_reason,
        ),
        model_rounds=model_rounds,
        mcp_calls=mcp_calls,
    )


def fallback_selection(
    *,
    registry: EvidenceRegistry,
    query: str,
    note: str,
    model_rounds: int,
    mcp_calls: int,
    max_items: int,
) -> RetrievalOutcome:
    evidence = registry.all()[:max_items]
    items = tuple(CitableItem(item.cite_uid, 0.0) for item in evidence)
    status = "partial" if evidence else "no_evidence"
    return RetrievalOutcome(
        status=status,
        items=items,
        note=note,
        evidence=evidence,
        requirement=EvidenceRequirement(
            id="req-1",
            claim_or_question=query,
            status="unresolved",
            cite_uids=tuple(item.cite_uid for item in evidence),
            gap_reason=note,
        ),
        model_rounds=model_rounds,
        mcp_calls=mcp_calls,
    )
