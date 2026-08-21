from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable

from app.evidence.models import (
    CitableItem,
    EvidenceRegistry,
    EvidenceRequirement,
    RetrievalOutcome,
    default_requirement_ledger,
)


@dataclass(slots=True)
class RetrievalSession:
    """All evidence state for one `ConversationDriver.complete()` call."""

    registry: EvidenceRegistry = field(default_factory=EvidenceRegistry)
    requirements_by_id: dict[str, EvidenceRequirement] = field(default_factory=dict)
    selected_scores: dict[str, float] = field(default_factory=dict)
    exact_call_cache: dict[str, str] = field(default_factory=dict)
    semantic_call_history: list[tuple[str, frozenset[str]]] = field(default_factory=list)
    cumulative_model_rounds: int = 0
    cumulative_mcp_calls: int = 0
    cumulative_bridge_calls: int = 0
    duplicate_blocked: int = 0
    invalid_calls: int = 0
    no_progress: int = 0
    stop_reason: str = "not_used"

    @classmethod
    def create(
        cls,
        requirements: Iterable[EvidenceRequirement] = (),
    ) -> "RetrievalSession":
        session = cls()
        session.seed(requirements)
        return session

    def seed(self, requirements: Iterable[EvidenceRequirement]) -> None:
        for requirement in requirements:
            identifier = requirement.id.strip()
            if not identifier or identifier in self.requirements_by_id:
                continue
            self.requirements_by_id[identifier] = requirement

    def target_requirements(
        self,
        proposed: Iterable[EvidenceRequirement] = (),
    ) -> tuple[EvidenceRequirement, ...]:
        self.seed(proposed)
        return tuple(
            requirement
            for requirement in self.requirements_by_id.values()
            if requirement.status != "supported"
        )

    def apply_outcome(
        self,
        outcome: RetrievalOutcome,
        *,
        diagnostics: dict[str, object] | None = None,
    ) -> None:
        self.cumulative_model_rounds += outcome.model_rounds
        self.cumulative_mcp_calls += outcome.mcp_calls
        diagnostics = diagnostics or {}
        self.cumulative_bridge_calls += _nonnegative_int(diagnostics.get("bridge_calls"))
        self.duplicate_blocked += _nonnegative_int(diagnostics.get("duplicate_blocked"))
        self.invalid_calls += _nonnegative_int(diagnostics.get("invalid_calls"))
        self.no_progress = _nonnegative_int(diagnostics.get("no_progress"))
        reason = diagnostics.get("stop_reason")
        if isinstance(reason, str) and reason:
            self.stop_reason = reason

        for item in outcome.items:
            self.selected_scores[item.cite_uid] = max(
                item.relevance_score,
                self.selected_scores.get(item.cite_uid, float("-inf")),
            )
        for updated in outcome.requirements:
            existing = self.requirements_by_id.get(updated.id)
            if existing is None:
                self.requirements_by_id[updated.id] = updated
                continue
            merged_cites = tuple(dict.fromkeys((*existing.cite_uids, *updated.cite_uids)))
            if existing.status == "supported" and updated.status in {"missing", "unresolved"}:
                status = "supported"
                gap_reason = ""
                verification_status = existing.verification_status
                applicability_status = existing.applicability_status
            elif existing.status == "supported" and updated.status == "contradicted":
                status = "contradicted"
                gap_reason = updated.gap_reason or "Conflicting evidence was retrieved."
                verification_status = updated.verification_status
                applicability_status = updated.applicability_status
            else:
                status = updated.status
                gap_reason = updated.gap_reason
                verification_status = updated.verification_status
                applicability_status = updated.applicability_status
            self.requirements_by_id[updated.id] = replace(
                existing,
                status=status,
                cite_uids=merged_cites,
                gap_reason=gap_reason,
                verification_status=verification_status,
                applicability_status=applicability_status,
            )

    def update_requirement(self, requirement: EvidenceRequirement) -> None:
        if requirement.id not in self.requirements_by_id:
            raise KeyError(requirement.id)
        self.requirements_by_id[requirement.id] = requirement

    def snapshot(self, *, note: str = "") -> RetrievalOutcome:
        requirements = tuple(self.requirements_by_id.values())
        unresolved_critical = tuple(
            requirement
            for requirement in requirements
            if requirement.criticality == "critical" and requirement.status != "supported"
        )
        selected = tuple(
            CitableItem(cite_uid, score)
            for cite_uid, score in self.selected_scores.items()
            if self.registry.get(cite_uid) is not None
        )
        evidence = tuple(
            item
            for selection in selected
            if (item := self.registry.get(selection.cite_uid)) is not None
        )
        if requirements and not unresolved_critical:
            status = "sufficient"
        elif evidence:
            status = "partial"
        else:
            status = "no_evidence"
        if not requirements:
            ledger = default_requirement_ledger("")
            requirements = ledger.requirements
            status = "no_evidence"
        return RetrievalOutcome(
            status=status,
            items=selected,
            note=note,
            evidence=evidence,
            requirements=requirements,
            model_rounds=self.cumulative_model_rounds,
            mcp_calls=self.cumulative_mcp_calls,
        )

    def diagnostics(self) -> dict[str, object]:
        return {
            "stop_reason": self.stop_reason,
            "bridge_calls": self.cumulative_bridge_calls,
            "duplicate_blocked": self.duplicate_blocked,
            "invalid_calls": self.invalid_calls,
            "no_progress": self.no_progress,
        }


def _nonnegative_int(value: object) -> int:
    return value if isinstance(value, int) and value >= 0 else 0
